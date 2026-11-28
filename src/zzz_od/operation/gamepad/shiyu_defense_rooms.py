"""第五防线三间按新成绩推进，并保留游戏已有成绩。"""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING

import cv2

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import cv2_utils
from zzz_od.application.shiyu_defense.shiyu_defense_team_utils import (
    DefensePhaseTeamInfo,
    select_teams,
)
from zzz_od.game_data.agent import AgentEnum
from zzz_od.operation.gamepad.predefined_team import GamepadChoosePredefinedTeam
from zzz_od.operation.gamepad.shiyu_defense import (
    GAMEPAD_SCREEN,
    ORDINALS,
    GamepadShiyuDeploy,
    _result_exit_button,
    _ShiyuFocusPredefinedTeam,
    _ShiyuMenu,
    read_critical_progress,
    read_team_order,
)
from zzz_od.operation.gamepad.shiyu_defense_battle import GamepadShiyuBattle
from zzz_od.operation.gamepad.transport import find_yellow_focus_frames
from zzz_od.operation.predefined_team_recognizer import (
    find_team_name,
    recognize_predefined_team,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.geometry.rectangle import Rect
    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


def read_room_score(words: dict[str, MatchResultList]) -> int | None:
    """把待挑战视为零，多个矛盾分值视为未知。"""
    scores: set[int] = set()
    for word in words:
        text = ''.join(word.split()).replace(',', '').replace('，', '')
        if text == '待挑战':
            scores.add(0)
        elif match := re.fullmatch(r'(?:(?:当前得分|总得分)[:：]?)?(?:[SAB][+＋]?)?(\d{1,5})(?:分)?', text):
            value = int(match[1])
            if value <= 50_000:
                scores.add(value)
    return scores.pop() if len(scores) == 1 else None


class _ShiyuRoomMenu(_ShiyuMenu):
    """复用式舆菜单识别，并按房间顺序读取真实分数。"""

    def rooms_visible(self) -> bool:
        return self._found('手柄-本期最佳总分') and self._found('手柄-防线选择')

    def read_scores(self) -> tuple[int, ...] | None:
        values: list[int] = []
        for room in (1, 2, 3):
            area = self._read(f'手柄-房间{room}状态')
            if area is None or (value := read_room_score(area[1])) is None:
                return None
            values.append(value)
        return tuple(values)

    def shown_room(self, room: int) -> bool:
        title = self._read('手柄-房间标题')
        return title is not None and any(
            ''.join(text.split()) == f'房间{ORDINALS[room - 1]}' for text in title[1]
        )


class GamepadShiyuOpenRoom(_ShiyuRoomMenu):
    """在三间列表中进入指定的未挑战房间，停在单队准备页。"""

    def __init__(self, ctx: ZContext, room: int) -> None:
        super().__init__(ctx, op_name='手柄打开式舆房间', timeout_seconds=55)
        self.room: int = room
        self._selected: bool = False
        self._rule_closed: bool = False
        self._prepare_requested: bool = False
        self._last_focus: int | None = None

    def handle_init(self) -> None:
        """按本次状态处理首次说明与焦点。"""
        self._selected = self._rule_closed = self._prepare_requested = False
        self._last_focus = None

    @operation_node(name='选择未挑战的房间', is_start_node=True, node_max_retry_times=16)
    def select_room(self) -> OperationRoundResult:
        """房间 A、说明 B、详情 Start 各自都需要对应的页面特征。"""
        if not 1 <= self.room <= 3:
            return self.round_fail('房间编号须为1至3')
        if self._prepare_requested and self._found('手柄-出战准备详情'):
            return self.round_success('已进入单队出战准备')
        if self._selected:
            if self._found('手柄-第五防线规则') and self._found('手柄-规则返回'):
                if not self._rule_closed:
                    self.ctx.controller.btn_tap('east')
                    self._rule_closed = True
                return self.round_retry('等待首次房间规则关闭', wait=0.8)
            if not self._prepare_requested and self.shown_room(self.room) and self._found('手柄-房间下一步'):
                score = self._read('手柄-房间详情总得分')
                if score is not None and read_room_score(score[1]) == 0:
                    self.save_screenshot(prefix='macos_shiyu_room_before')
                    self.ctx.controller.btn_tap('start')
                    self._prepare_requested = True
            return self.round_retry('等待目标房间详情或单队准备', wait=0.8)
        scores = self.read_scores() if self.rooms_visible() else None
        if scores is None:
            return self.round_retry('等待第五防线三间与已有成绩', wait=0.8)
        if scores[self.room - 1] != 0:
            return self.round_fail('目标房间已有成绩，保留并停止重复挑战')
        area = self._read('手柄-三间框')
        if area is None:
            return self.round_retry('等待三间焦点', wait=0.8)
        frames = find_yellow_focus_frames(area[0], 1025, 1120, 195, 232, min_edge_presence=0.65)
        if len(frames) != 1:
            return self.round_retry('等待唯一房间黄框', wait=0.8)
        focus = min((1, 2, 3), key=lambda idx: abs(frames[0].center.y - (116 + 224 * (idx - 1))))
        if focus == self.room:
            self.ctx.controller.btn_tap('south')
            self._selected = True
        elif focus != self._last_focus:
            self.ctx.controller.btn_tap('dpad_down' if focus < self.room else 'dpad_up')
        self._last_focus = focus
        return self.round_retry('等待房间选择反馈', wait=0.8)


class _ShiyuChooseRoomTeam(GamepadChoosePredefinedTeam):
    """单房复用原手柄选队，同时核对式舆的 TEAM 01 徽标。"""

    @node_from(from_name='打开编队列表')
    @operation_node(name='定位目标编队', timeout_seconds=115)
    def locate(self) -> OperationRoundResult:
        """当前页能核对目标就直接定位，减少锁定页不必要的滚动。"""
        if self.last_screenshot is not None:
            words = self.ctx.ocr.run_ocr(self.last_screenshot)
            if any('预备出战' in text for text in words):
                slot = self._visible_matching_team(words)
                if slot is not None:
                    self._visible_slot = slot
                    return self.round_success('已在当前页核对目标编队')
        result = _ShiyuFocusPredefinedTeam(self.ctx, self.target_team_idx).execute()
        if result.success:
            self._visible_slot = result.data['visible_slot']
        return self.round_by_op_result(result)

    def _visible_matching_team(self, words: dict[str, MatchResultList]) -> int | None:
        """原配置和当前页都必须只有一个同名队，再匹配实际成员顺序。"""
        team = self.ctx.team_config.get_team_by_idx(self.target_team_idx)
        if team is None or not team.name or sum(
            item.name == team.name for item in self.ctx.team_config.team_list
        ) != 1:
            return None
        areas = [self.ctx.screen_loader.get_area('编队选择', f'编队槽位{idx}') for idx in range(6)]
        matching = [(slot, area) for slot, area in enumerate(areas)
                    if area is not None and find_team_name(words, area.rect) == team.name]
        if len(matching) != 1:
            return None
        slot, area = matching[0]
        actual = recognize_predefined_team(self.ctx, self.last_screenshot, words, area.rect)
        members = [match.data.agent_id for match in actual.agent_match_result_list]
        return slot if len(members) == 3 and members == team.agent_id_list else None

    def is_target_selected(self, words: dict[str, MatchResultList], card: Rect) -> bool:
        return super().is_target_selected(words, card) or read_team_order(words, card) == 1


class GamepadShiyuRoomReturn(_ShiyuRoomMenu):
    """完成一间后退出结算，读取更新后的三间成绩。"""

    def __init__(self, ctx: ZContext, room: int, previous_scores: tuple[int, ...]) -> None:
        super().__init__(ctx, op_name='手柄核对式舆房间得分', timeout_seconds=50)
        self.room: int = room
        self._previous_scores: tuple[int, ...] = previous_scores
        self._exit_sent: bool = False
        self._detail_closed: bool = False

    @operation_node(name='退出房间并核对新增得分', is_start_node=True, node_max_retry_times=22)
    def exit_and_read(self) -> OperationRoundResult:
        """本房间必须得到新分，已完成房间的分数保持一致。"""
        if self._exit_sent:
            if self.rooms_visible():
                scores = self.read_scores()
                if scores is not None:
                    other = tuple(score for index, score in enumerate(scores, 1) if index != self.room)
                    expected = tuple(score for index, score in enumerate(self._previous_scores, 1) if index != self.room)
                    if other != expected:
                        return self.round_fail('其他房间分数与出战前不一致')
                    if scores[self.room - 1] > self._previous_scores[self.room - 1]:
                        self.save_screenshot(prefix='macos_shiyu_room_score')
                        return self.round_success('已核对第五防线房间得分', data={'scores': scores})
            if not self._detail_closed and self.shown_room(self.room) and self._found('手柄-房间下一步'):
                score = self._read('手柄-房间详情总得分')
                if score is not None and (value := read_room_score(score[1])) is not None and value > 0:
                    self.ctx.controller.btn_tap('east')
                    self._detail_closed = True
        elif self.last_screenshot is not None and self.round_by_find_area(
            self.last_screenshot, '战斗画面', '手柄-挑战结果',
        ).is_success:
            action = self._read('手柄-结果退出')
            key = _result_exit_button(action[1]) if action is not None else None
            if key is not None:
                self.ctx.controller.resume_input()
                self.ctx.controller.btn_tap(key)
                self._exit_sent = True
        if self.node_retry_times == 16:
            self.save_screenshot(prefix='macos_shiyu_room_score_pending')
        return self.round_retry('等待房间结算退出与新增真实得分', wait=1)


class GamepadShiyuRooms(_ShiyuRoomMenu):
    """规划互斥的三队，逐间处理并回到剧变选关。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄式舆第五防线', timeout_seconds=3200)
        self._scores: tuple[int, ...] = ()
        self._team_indices: dict[int, int] = {}
        self._returned: bool = False

    @cached_property
    def _room_avatar_templates(self) -> list[tuple[str, MatLike, MatLike]]:
        """把原连携头像和遮罩缩放到房间头像大小，保留皮肤变体。"""
        portraits: list[tuple[str, MatLike, MatLike]] = []
        for entry in AgentEnum:
            agent = entry.value
            for template_id in agent.template_id_list:
                template = self.ctx.template_loader.get_template('battle', f'avatar_chain_{template_id}')
                if template is None or template.raw is None or template.mask is None:
                    continue
                for size in (43, 44, 45, 47, 49, 51):
                    image = cv2.resize(template.raw, (size, size), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(template.mask, (size, size), interpolation=cv2.INTER_NEAREST)
                    portraits.append((agent.agent_id, image, mask))
        return portraits

    def _read_locked_room_agents(self, scores: tuple[int, ...]) -> set[str] | None:
        """每个已得分房间须认出三个互异角色，第四格邦布不参与。"""
        locked: set[str] = set()
        for room, score in enumerate(scores, 1):
            if score == 0:
                continue
            area = self.ctx.screen_loader.get_area(GAMEPAD_SCREEN, f'手柄-房间{room}角色头像')
            if area is None or self.last_screenshot is None:
                return None
            strip = cv2_utils.crop_image_only(self.last_screenshot, area.rect)
            width = area.width // 3
            for slot in range(3):
                agent_id = self._identify_room_avatar(strip[:, slot * width:(slot + 1) * width])
                if agent_id is None or agent_id in locked:
                    return None
                locked.add(agent_id)
        return locked

    def _identify_room_avatar(self, tile: MatLike) -> str | None:
        """独立比对角色模板。置信度或相对第二名的优势不足时等待。"""
        scores: dict[str, float] = {}
        for agent_id, image, mask in self._room_avatar_templates:
            match = cv2_utils.match_template(tile, image, 0.8, mask=mask, ignore_inf=True).max
            if match is not None:
                scores[agent_id] = max(scores.get(agent_id, 0), match.confidence)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        next_score = scores[ranked[1]] if len(ranked) > 1 else 0.8
        if scores[best] < 0.92 or scores[best] - next_score < 0.10:
            return None
        return best

    @operation_node(name='核对三间成绩与编队', is_start_node=True, node_max_retry_times=12)
    def plan(self) -> OperationRoundResult:
        """只给零分房间配队，并依据当前游戏头像排除已锁角色。"""
        scores = self.read_scores() if self.rooms_visible() else None
        if scores is None:
            return self.round_retry('等待第五防线三间分数', wait=0.8)
        self._scores = scores
        if all(scores):
            return self.round_success('核对第五防线完成')
        locked = self._read_locked_room_agents(scores)
        if locked is None:
            if self.node_retry_times >= 5:
                self.save_screenshot(prefix='macos_shiyu_locked_agents_pending')
                return self.round_fail('既有成绩已保留，未能核对已得分房间的锁定角色')
            return self.round_retry('等待已得分房间的锁定角色完整显示', wait=0.8)
        pending = [room for room, score in enumerate(scores, 1) if score == 0]
        targets: list[DefensePhaseTeamInfo] = []
        for room in pending:
            weak = self.read_attributes(f'手柄-房间{room}弱点', '推荐属性')
            resistant = self.read_attributes(f'手柄-房间{room}抗性', '强敌抗性')
            if not weak or resistant is None:
                return self.round_retry('等待三间完整推荐属性和抗性', wait=0.8)
            targets.append(DefensePhaseTeamInfo(weak, resistant))
        candidates = [
            team.idx for team in self.ctx.team_config.team_list
            if len(set(team.agent_id_list)) == 3 and 'unknown' not in team.agent_id_list
            and locked.isdisjoint(team.agent_id_list)
        ]
        choices = select_teams(self.ctx, targets, candidates)
        if len(choices) != len(pending):
            return self.round_fail('未挑战房间找不到彼此及已锁角色均不重叠的三人预备编队')
        self._team_indices = {room: team.team_idx for room, team in zip(pending, choices, strict=True)}
        return self.round_success('开始三间')

    @node_from(from_name='核对三间成绩与编队', status='开始三间')
    @operation_node(name='按实际成绩完成未打房间', screenshot_before_round=False, timeout_seconds=3080)
    def run_room(self) -> OperationRoundResult:
        """每间重新识别准备和结果，失败时保留已获分数。"""
        if all(self._scores):
            return self.round_success('核对第五防线完成')
        room = self._scores.index(0) + 1
        team = self._team_indices[room]
        entries = (
            GamepadShiyuOpenRoom(self.ctx, room),
            _ShiyuChooseRoomTeam(self.ctx, team),
            GamepadShiyuDeploy(self.ctx, single_room=True),
            GamepadShiyuBattle(self.ctx, team, room_result=True),
        )
        for task in entries:
            result = task.execute()
            if not result.success:
                return self.round_by_op_result(result)
        if result.status != GamepadShiyuBattle.STATUS_ROOM_RESULT:
            return self.round_fail('未到可核对的第五防线房间结算')
        result = GamepadShiyuRoomReturn(self.ctx, room, self._scores).execute()
        if not result.success:
            return self.round_by_op_result(result)
        self._scores = result.data['scores']
        return self.round_wait('已保留本房间实际分数')

    @node_from(from_name='核对三间成绩与编队', status='核对第五防线完成')
    @node_from(from_name='按实际成绩完成未打房间', status='核对第五防线完成')
    @operation_node(name='返回选关核对第五防线进度', node_max_retry_times=15)
    def verify_completion(self) -> OperationRoundResult:
        """确认三间均有成绩，并以实际剧变 5/5 作为完成依据。"""
        if self._returned and self._found('手柄-前哨档案'):
            tab = self._read('手柄-剧变进度')
            progress = read_critical_progress(tab[1]) if tab is not None else None
            if progress == (5, 5):
                return self.round_success('已核对第五防线和三间成绩', data={'scores': self._scores})
            if progress == (4, 5) and self.node_retry_times >= 3:
                return self.round_fail('三间均有分数，游戏未记第五防线完成')
        elif not self._returned and self.rooms_visible():
            self.ctx.controller.btn_tap('east')
            self._returned = True
        return self.round_retry('等待第五防线进度确认', wait=1)
