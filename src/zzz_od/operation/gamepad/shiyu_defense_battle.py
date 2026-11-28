"""剧变节点战斗沿用式舆识别与策略，并确认分队切换和真实结果。"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, ClassVar

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import cv2_utils
from zzz_od.application.shiyu_defense.shiyu_defense_battle import ShiyuDefenseBattle
from zzz_od.controller.macos_battle_input_guard import MacBattleHudGuard
from zzz_od.game_data.agent import AgentEnum
from zzz_od.operation.gamepad.battle_assistant import is_native_battle
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.shiyu_defense import GAMEPAD_SCREEN, _result_exit_button

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_base import OperationResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


def read_enemy_goal(words: dict[str, MatchResultList]) -> bool | None:
    """只读取与敌人目标同一行的完成比，排除时间目标的 1/1。"""
    label_rows: list[int] = []
    ratios: list[tuple[int, bool]] = []
    for text, matches in words.items():
        compact = ''.join(text.split())
        match = re.fullmatch(r'(消灭所有敌人)?([01])[/／]1', compact)
        for rect in matches:
            if compact == '消灭所有敌人' or (match is not None and match[1] is not None):
                label_rows.append(rect.center.y)
            if match is not None:
                ratios.append((rect.center.y, match[2] == '1'))
    if len(label_rows) != 1:
        return None
    values = [ok for y, ok in ratios if abs(y - label_rows[0]) < 22]
    return values[0] if len(values) == 1 else None


class GamepadShiyuBattle(GamepadOperation):
    """完成一队的战斗，确认下一队或式舆结算后才交还父操作。"""

    STATUS_NEXT_TEAM: ClassVar[str] = '已确认下半场编队'
    STATUS_VICTORY: ClassVar[str] = '已确认剧变节点胜利'
    STATUS_RESUME_BETWEEN: ClassVar[str] = '从层间入口继续'
    STATUS_ROOM_RESULT: ClassVar[str] = '已到第五防线房间结算'
    realtime_nodes: ClassVar[frozenset[str]] = frozenset({'接敌', '战斗', '层间移动'})

    def __init__(
        self, ctx: ZContext, team_idx: int, next_team_idx: int | None = None,
        resume_between: bool = False, room_result: bool = False,
    ) -> None:
        super().__init__(ctx, op_name='手柄式舆战斗', timeout_seconds=790)
        self.original: ShiyuDefenseBattle = ShiyuDefenseBattle(ctx, team_idx)
        self._next_team_idx: int | None = next_team_idx
        self._resume_between: bool = resume_between
        self._room_result: bool = room_result
        self._scope: bool = False
        self._running: bool = False
        self._timer_checked_at: float = 0
        self._timer_absent_since: float | None = None
        self._unknown_since: float | None = None
        self._interact_sent: bool = False
        self._between_moves: int = 0

    def handle_init(self) -> None:
        """复用操作时从新截图开始确认，不继承前一场计时。"""
        self._scope = False
        self._running = False
        self._timer_checked_at = 0
        self._timer_absent_since = None
        self._unknown_since = None
        self._interact_sent = False
        self._between_moves = 0
        self.original.move_times = 0
        self.original.distance_pos = None

    def execute(self) -> OperationResult:
        """异常、失败或正常换队后都关掉策略和本场输入。"""
        battle = self.ctx.auto_battle_context
        variants = battle.normal_attack_template_variants
        battle.normal_attack_template_variants = (*variants, ('ios_gamepad', 'lost_void_attack_inner'))
        try:
            return super().execute()
        finally:
            try:
                battle.stop_auto_battle()
                self._running = False
            finally:
                try:
                    if self._scope:
                        self.ctx.controller.end_battle_input_scope()
                        self._scope = False
                finally:
                    battle.normal_attack_template_variants = variants

    def _stop_strategy(self) -> None:
        if self._running:
            self._running = False
            self.ctx.auto_battle_context.stop_auto_battle()

    def handle_pause(self) -> None:
        """停掉当前策略，由战斗节点按新图恢复。"""
        self._stop_strategy()

    def handle_resume(self) -> None:
        """暂停前的战斗帧不能作为恢复策略的依据。"""
        if self._scope:
            self._unknown_since = time.monotonic()

    @operation_node(name='确认本场编队与策略', is_start_node=True, screenshot_before_round=False)
    def load_strategy(self) -> OperationRoundResult:
        """检查编队可用，调用原式舆策略选择。"""
        if self.original.team_config is None:
            return self.round_fail('未找到本场式舆预备编队')
        self.ctx.auto_battle_context.init_battle_context()
        self.original.load_auto_op()
        if self.ctx.auto_battle_context.auto_op is None:
            return self.round_fail('未加载本场式舆战斗策略')
        return self.round_success()

    def _team_matches(self, screen: MatLike, team_idx: int) -> bool:
        """用原战斗头像模板核对已计划的三个成员。"""
        team = self.ctx.team_config.get_team_by_idx(team_idx)
        if team is None:
            return False
        expected = set(team.agent_id_list)
        candidates = [(member.value, None) for member in AgentEnum if member.value.agent_id in expected]
        if len(candidates) != 3:
            return False
        matcher = self.ctx.auto_battle_context.agent_context
        actual: set[str] = set()
        for idx, area in enumerate((matcher.area_agent_3_1, matcher.area_agent_3_2, matcher.area_agent_3_3)):
            if area is None:
                return False
            image = cv2_utils.crop_image_only(screen, area.rect)
            matched, _ = matcher._match_agent_in(image, idx == 0, candidates, area.rect)
            if matched is None:
                return False
            actual.add(matched.agent_id)
        return actual == expected

    @node_from(from_name='确认本场编队与策略')
    @operation_node(name='等待本场战斗', node_max_retry_times=30, timeout_seconds=40)
    def wait_for_battle(self) -> OperationRoundResult:
        """先核实手柄战斗与本半场队员，再允许移动。"""
        screen = self.last_screenshot
        team = self.original.team_config
        if team is None or screen is None:
            return self.round_retry('等待式舆战斗画面', wait=1)
        at_portal = self._resume_between and self._has_layer_interact(screen)
        if not at_portal and not is_native_battle(self.ctx, screen):
            return self.round_retry('等待式舆战斗画面', wait=1)
        if self._resume_between and self._is_next_team(screen):
            return self.round_success(self.STATUS_NEXT_TEAM)
        if not self._team_matches(screen, team.idx):
            return self.round_retry('等待本半场预备编队的战斗头像', wait=1)
        if self._resume_between:
            if self.original.check_shiyu_countdown(screen):
                return self.round_fail('当前仍是计时战斗，无法从层间继续')
            if not at_portal and self._layer_target(screen) is None:
                return self.round_retry('等待可核对的式舆层间目标', wait=1)
        if not at_portal:
            self._begin_battle_scope()
        if self._resume_between:
            return self.round_success(self.STATUS_RESUME_BETWEEN)
        return self.round_success()

    def _begin_battle_scope(self) -> None:
        controller = self.ctx.controller
        controller.resume_input()
        controller.begin_battle_input_scope(MacBattleHudGuard(self.ctx.auto_battle_context))
        self._scope = True

    @node_from(from_name='等待本场战斗', ignore_status=False)
    @operation_node(name='接敌', timeout_seconds=65)
    def approach(self) -> OperationRoundResult:
        """保持原接敌速度，只有确认战斗画面才复用移动。"""
        if self.last_screenshot is None or not self._refresh_scope():
            return self._wait_for_known_screen()
        self._unknown_since = None
        self.original.last_screenshot = self.last_screenshot
        result = self.original.start_move()
        if result.is_success:
            self._running = True
        return result

    def _refresh_scope(self) -> bool:
        """护栏恢复按最新帧核对，不在失焦画面继续输出。"""
        controller = self.ctx.controller
        if self._unknown_since is not None or controller.battle_input_blocked:
            return controller.refresh_battle_input_scope()
        return True

    def _read_transition(self) -> OperationRoundResult | None:
        """下半场换队与最终结算都需要当前截图的正向依据。"""
        if self.last_screenshot is None:
            return None
        if self.round_by_find_area(self.last_screenshot, '战斗画面', '手柄-挑战结果').is_success:
            self._stop_strategy()
            if self._room_result:
                footer = self.ctx.screen_loader.get_area(GAMEPAD_SCREEN, '手柄-结果退出')
                if footer is not None and _result_exit_button(
                    self.ctx.ocr.crop_and_run_ocr(self.last_screenshot, footer.rect),
                ) is not None:
                    self.save_screenshot(prefix='macos_shiyu_room_result')
                    return self.round_success(self.STATUS_ROOM_RESULT)
                return self._wait_for_known_screen()
            area = self.ctx.screen_loader.get_area(GAMEPAD_SCREEN, '手柄-结算目标')
            if area is not None:
                result = read_enemy_goal(self.ctx.ocr.crop_and_run_ocr(self.last_screenshot, area.rect))
                if result is False:
                    return self.round_fail('未消灭所有敌人')
                if result is True and self._next_team_idx is None:
                    self.save_screenshot(prefix='macos_shiyu_victory')
                    return self.round_success(self.STATUS_VICTORY)
            return self._wait_for_known_screen()
        if self._is_next_team(self.last_screenshot):
            self._stop_strategy()
            return self.round_success(self.STATUS_NEXT_TEAM)
        return None

    def _is_next_team(self, screen: MatLike) -> bool:
        return (
            self._next_team_idx is not None
            and is_native_battle(self.ctx, screen)
            and self._team_matches(screen, self._next_team_idx)
        )

    @node_from(from_name='接敌')
    @node_from(from_name='层间移动', status='本半场继续战斗')
    @operation_node(name='战斗', mute=True, timeout_seconds=600)
    def fight(self) -> OperationRoundResult:
        """保留原战斗状态识别和五秒倒计时判定。"""
        screen = self.last_screenshot
        interval = self.ctx.battle_assistant_config.screenshot_interval
        battle = self.ctx.auto_battle_context
        if screen is None or self._unknown_since is not None or self.ctx.controller.battle_input_blocked:
            self._stop_strategy()
            transition = self._read_transition()
            if transition is not None:
                return transition
            if screen is not None and self._has_layer_interact(screen):
                self._interact_sent = False
                return self.round_success('进入层间移动')
            if screen is None or not self._refresh_scope():
                return self._wait_for_known_screen()
        self._unknown_since = None
        if not self._running:
            battle.resume_auto_battle()
            self._running = True
        if battle.check_battle_state(screen, self.last_screenshot_time):
            now = self.last_screenshot_time
            if now - self._timer_checked_at >= 1:
                self._timer_checked_at = now
                if self.original.check_shiyu_countdown(screen):
                    self._timer_absent_since = None
                elif self._timer_absent_since is None:
                    self._timer_absent_since = now
                elif now - self._timer_absent_since >= 5:
                    self._stop_strategy()
                    self._interact_sent = False
                    return self.round_success('进入层间移动')
        else:
            self._timer_absent_since = None
        return self.round_wait(wait=interval)

    @node_from(from_name='等待本场战斗', status=STATUS_RESUME_BETWEEN)
    @node_from(from_name='战斗', status='进入层间移动')
    @operation_node(name='层间移动', timeout_seconds=80)
    def move_between(self) -> OperationRoundResult:
        """确认换队、结果或原层间移动目标，再决定是否输入。"""
        transition = self._read_transition()
        if transition is not None:
            return transition
        screen = self.last_screenshot
        if screen is None:
            return self._wait_for_known_screen()
        if self._interact_sent:
            team = self.original.team_config
            if (
                team is not None and is_native_battle(self.ctx, screen)
                and self.original.check_shiyu_countdown(screen) and self._team_matches(screen, team.idx)
            ):
                self._begin_battle_scope()
                self._interact_sent = False
            else:
                return self._wait_for_known_screen()
        elif self._has_layer_interact(screen):
            # 这里的 A 已替代普通攻击。先撤掉战斗保护和策略，再发送一次明确交互。
            self._stop_strategy()
            if self._scope:
                self.ctx.controller.end_battle_input_scope()
                self._scope = False
            self.ctx.controller.resume_input()
            self.ctx.controller.interact(press=True, press_time=0.2, release=True)
            self._interact_sent = True
            self._unknown_since = None
            return self._wait_for_known_screen()
        if not self._refresh_scope():
            return self._wait_for_known_screen()
        if self.original.check_shiyu_countdown(screen):
            self._timer_absent_since = None
            self._unknown_since = None
            self._between_moves = 0
            return self.round_success('本半场继续战斗')
        target = self._layer_target(screen)
        if target is None:
            return self._wait_for_known_screen()
        target_x, distance = target
        if distance <= 2:
            return self._wait_for_known_screen()
        if self._between_moves >= 36:
            self.save_screenshot(prefix='macos_shiyu_layer_target_pending')
            return self.round_fail('未能到达式舆层间入口')
        self._unknown_since = None
        offset = target_x - self.ctx.project_config.screen_standard_width // 2
        if abs(offset) > 70:
            # 屏边目标需要实质转动，再由新画面校正剩余偏差。
            steps = max(65, min(abs(offset) * 0.65, 230))
            self.ctx.controller.turn_by_distance(steps if offset > 0 else -steps)
        else:
            self.ctx.controller.move_w(press=True, press_time=min(distance / 7.2, 1.3), release=True)
        self._between_moves += 1
        return self.round_wait(wait=0.5)

    def _has_layer_interact(self, screen: MatLike) -> bool:
        return (
            self.round_by_find_area(screen, '战斗画面', '按键-交互').is_success
            and self.round_by_find_area(screen, GAMEPAD_SCREEN, '手柄-层间交互键').is_success
        )

    def _layer_target(self, screen: MatLike) -> tuple[int, float] | None:
        """使用当前距离框的数值，缺距离时沿原式舆传送点检测。"""
        marker = self.ctx.auto_battle_context.check_battle_distance(screen)
        if marker is not None and isinstance(marker.data, (float, int)) and 0 < marker.data < 60:
            return marker.center.x, marker.data
        portal = self.original.check_teleport_point(screen)
        return (portal.center.x, 5) if portal is not None else None

    def _wait_for_known_screen(self) -> OperationRoundResult:
        now = time.monotonic()
        if self._unknown_since is None:
            self._unknown_since = now
        elif now - self._unknown_since >= 25:
            self.save_screenshot(prefix='macos_shiyu_battle_pending')
            return self.round_fail('未确认式舆战斗、换队或结算画面')
        return self.round_wait('等待式舆战斗、换队或结算画面', wait=0.5)
