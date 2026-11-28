"""用原枯萎之都模型识别走格，并在新棋盘核对手柄移动。"""

from __future__ import annotations

import re
import time
from collections import deque
from statistics import median
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from one_dragon.base.operation.application import application_const
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import gpu_executor
from one_dragon.utils.log_utils import log
from zzz_od.application.hollow_zero.hollow_eval_speed import hollow_eval_speed_const
from zzz_od.application.hollow_zero.hollow_eval_speed.hollow_eval_speed_config import (
    HollowEvalSpeedConfig,
)
from zzz_od.operation.gamepad.battle_assistant import is_native_battle
from zzz_od.operation.gamepad.focus import has_filled_highlight
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.hollow_eval_battle import (
    GamepadHollowEvalBattle,
    GamepadHollowEvalBattleResult,
)
from zzz_od.operation.gamepad.hollow_eval_door import GamepadHollowEvalDoor
from zzz_od.operation.gamepad.hollow_eval_event import GamepadHollowEvalEvent
from zzz_od.operation.gamepad.hollow_eval_path import (
    neighbor_direction,
    route_around_loop,
)
from zzz_od.operation.gamepad.hollow_eval_receipt import (
    GamepadHollowEvalMeritReceipt,
    confirms_weekly_merit_cap,
)
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext
    from zzz_od.hollow_zero.hollow_map.hollow_zero_map import (
        HollowZeroMap,
        HollowZeroMapNode,
    )

BoardState = tuple[int | None, int, tuple[tuple[tuple[int, int], str], ...], tuple[int, int] | None]


def _board_cells(board: HollowZeroMap) -> dict[tuple[int, int], str]:
    """以相对行列给可识别格定位，同类格可出现在不同列。"""
    if board.current_idx is None:
        return {}
    current = board.nodes[board.current_idx].pos.center
    horizontal: list[int] = []
    vertical: list[int] = []
    for start, neighbors in board.edges.items():
        first = board.nodes[start]
        for end in neighbors:
            second = board.nodes[end]
            direction = neighbor_direction(first, second)
            if direction is None:
                continue
            if direction[0]:
                horizontal.append(abs(first.pos.center.x - second.pos.center.x))
            else:
                vertical.append(abs(first.pos.center.y - second.pos.center.y))
    spacing_x = median(horizontal) if horizontal else None
    spacing_y = median(vertical) if vertical else None
    cells: dict[tuple[int, int], str] = {}
    ambiguous: set[tuple[int, int]] = set()
    for index, node in enumerate(board.nodes):
        name = node.entry.entry_name
        if (index == board.current_idx or node.confidence < 0.65
                or name in ('当前', '未知', '空白', '空白已通行')):
            continue
        cell = (
            round((node.pos.center.x - current.x) / spacing_x) if spacing_x else 0,
            round((node.pos.center.y - current.y) / spacing_y) if spacing_y else 0,
        )
        if cell in cells and cells[cell] != name:
            ambiguous.add(cell)
        cells[cell] = name
    return {cell: name for cell, name in cells.items() if cell not in ambiguous}


def board_moved_one_tile(
    before: dict[tuple[int, int], str], after: dict[tuple[int, int], str], direction: tuple[int, int],
    reward_confirmed: bool = False,
) -> bool:
    """对齐预计位移和其他方向，至少两类实见棋格都支持唯一最佳位移。"""
    def evidence(delta_x: int, delta_y: int) -> tuple[int, int]:
        overlap = [
            (before[(x + delta_x, y + delta_y)], name)
            for (x, y), name in after.items() if (x + delta_x, y + delta_y) in before
        ]
        matches = [name for old_name, name in overlap if name == old_name]
        return 2 * len(matches) - len(overlap), len(set(matches))

    score, kinds = evidence(*direction)
    minimum = 1 if reward_confirmed else 2
    if score < minimum or kinds < minimum:
        return False
    alternatives = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    return all(score > evidence(*other)[0] for other in alternatives if other != direction)


def _has_white_tiles(screen: MatLike, board: HollowZeroMap) -> bool:
    """展开动画会把电视格中央填成近纯白，模型偶尔误判成有效格。"""
    for node in board.nodes:
        rect = node.pos
        image = screen[
            max(0, rect.y1 + rect.height // 6):rect.y2 - rect.height // 3,
            max(0, rect.x1 + rect.width // 6):rect.x2 - rect.width // 6,
        ]
        if image.size and np.all(image >= 225, axis=2).mean() >= 0.8:
            return True
    return False


class GamepadHollowEvalStep(GamepadOperation):
    """用原速刷寻路选择并核对一格，只发送一次方向输入。"""

    SCREEN: ClassVar[str] = '零号空洞-事件'
    STATUS_READY_TO_LEAVE: ClassVar[str] = '已核对本次业绩停止条件'
    OBSERVABLE_EVENTS: ClassVar[frozenset[str]] = frozenset({'邦布商人', '呼叫增援', '业绩考察点'})
    BATTLE_ENTRIES: ClassVar[frozenset[str]] = frozenset({
        '危机', '双重危机', '限时战斗', '守门人', '门扉禁闭-善战', '门扉禁闭-侵蚀',
    })
    MAINLINE_EXITS: ClassVar[tuple[str, ...]] = ('传送点', '守门人', '不宜久留')
    LANDMARKS: ClassVar[frozenset[str]] = frozenset({
        '业绩考察点', '业绩考察点空', '传送点', '守门人', '门扉禁闭-财富', '门扉禁闭-善战',
        '门扉禁闭-侵蚀', '零号银行', '邦布商人', '失联',
    })
    KEYS: ClassVar[dict[tuple[int, int], str]] = {
        (1, 0): 'dpad_right', (-1, 0): 'dpad_left', (0, -1): 'dpad_up', (0, 1): 'dpad_down',
    }

    def __init__(self, ctx: ZContext, reuse_run: bool = False, stop_on_merit: bool = False) -> None:
        super().__init__(ctx, op_name='手柄零号空洞移动一格', timeout_seconds=25)
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', '周期征讨')
        self.reuse_run: bool = reuse_run
        self.stop_on_merit: bool = stop_on_merit
        self.before: dict[tuple[int, int], str] = {}
        self.before_path: str | None = None
        self._before_reward: tuple[int, int] | None = None
        self._before_resonium_count: int | None = None
        self.direction: tuple[int, int] | None = None
        self.moved_node: HollowZeroMapNode | None = None
        self.before_board: HollowZeroMap | None = None
        self.floor: int | None = None
        self._before_floor: int | None = None
        self._confirmed_merit_floor: int | None = None
        self.merit_award_floor: int | None = None
        self._merit_sample: tuple[int, tuple[tuple[int, int], ...], bool] | None = None
        self._weekly_cap_proof: str | None = None
        self._last_frame_time: float = -1
        self._recent_routes: deque[tuple[BoardState, tuple[int, int]]] = deque(maxlen=24)
        self._mainline_phase: tuple[int | None, int] | None = None
        self._expanding_since: float | None = None
        self._layout_to_settle: tuple[int | None, frozenset[tuple[int, int]]] | None = None

    def handle_init(self) -> None:
        self._confirmed_merit_floor = None
        self.merit_award_floor = None
        self._weekly_cap_proof = None
        self._recent_routes.clear()
        self._mainline_phase = None
        self._expanding_since = None
        self._layout_to_settle = None
        self.reset_move()

    def reset_move(self) -> None:
        """保留本局已核对的考察点记录，重新识别下一步。"""
        self.before.clear()
        self.before_path = None
        self._before_reward = None
        self._before_resonium_count = None
        self.direction = None
        self.moved_node = None
        self.before_board = None
        self.floor = None
        self._before_floor = None
        self._merit_sample = None
        self._last_frame_time = -1

    def active_board(self) -> HollowZeroMap | None:
        """画面需有层数和背包标记，地图只用本帧结果。"""
        if self.last_screenshot is None:
            return None
        self.navigation.last_screenshot = self.last_screenshot
        level = self.navigation._read_area(self.SCREEN, '手柄-走格层数')
        bag = self.navigation._read_area(self.SCREEN, '手柄-走格背包')
        if level is None or bag is None or '背包' not in bag[1]:
            return None
        floors = {
            int(match[1]) for text in level[1]
            if (match := re.search(r'当前层数\s*[：:]?\s*([1-3])\s*[/／]\s*3', text)) is not None
        }
        if len(floors) != 1:
            return None
        self.floor = next(iter(floors))
        return gpu_executor.submit(
            self.ctx.withered_domain.map_service.cal_current_map_by_screen,
            self.last_screenshot,
            self.last_screenshot_time,
        ).result()

    def _board_number(self, area_name: str) -> int | None:
        if self.last_screenshot is None:
            return None
        area = self.ctx.screen_loader.get_area('零号空洞-入口', area_name)
        if area is None:
            return None
        rect = area.rect
        image = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
        value = self.ctx.ocr.run_ocr_single_line(image).strip()
        return int(value) if re.fullmatch(r'[0-9]{1,6}', value) else None

    def _reward_hud(self) -> tuple[int, int] | None:
        """只读取固定的压力和齿轮硬币余额，忽略动画增量数字。"""
        pressure = self._board_number('手柄-棋盘压力')
        coins = self._board_number('手柄-棋盘硬币')
        return (pressure, coins) if pressure is not None and pressure <= 100 and coins is not None else None

    def _narrow_reward_receipt(self, cells: dict[tuple[int, int], str]) -> bool:
        """单锚点时用本步明确的同层 HUD 变化补足回执。"""
        if self._before_reward is None or self.moved_node is None or self.direction is None:
            return False
        after = self._reward_hud()
        if after is None:
            return False
        if self.moved_node.entry.entry_name == '齿轮硬币箱':
            rewarded = all(old < new for old, new in zip(self._before_reward, after, strict=True))
        elif self.moved_node.entry.entry_name == '扭蛋机' and self._before_resonium_count is not None:
            # 扭蛋机关格消失后只余一只相邻危机，压力加十、鸣徽加一，硬币不动。
            pressure, coins = self._before_reward
            rewarded = (
                self._before_floor is not None and self.floor == self._before_floor and cells == {self.direction: '危机'}
                and after == (pressure + 10, coins)
                and self._board_number('手柄-棋盘鸣徽数量') == self._before_resonium_count + 1
            )
        else:
            return False
        return rewarded and board_moved_one_tile(self.before, cells, self.direction, reward_confirmed=True)

    def waiting_for_expansion(self) -> bool:
        return self._expanding_since is not None and time.monotonic() - self._expanding_since < 20

    def _board_settled(self, board: HollowZeroMap) -> bool:
        """发现亮白格后，等到相同楼层和棋盘行列连续稳定。"""
        if self.last_screenshot is not None and _has_white_tiles(self.last_screenshot, board):
            if self._expanding_since is None:
                self._expanding_since = time.monotonic()
                self.save_screenshot(prefix='macos_hollow_eval_board_expanding')
            self._layout_to_settle = None
            return False
        if self._expanding_since is None:
            return True
        layout = frozenset(_board_cells(board))
        previous = self._layout_to_settle
        if previous is None or previous[0] != self.floor or len(previous[1] ^ layout) > 2:
            self._layout_to_settle = self.floor, layout
            return False
        self._expanding_since = None
        self._layout_to_settle = None
        log.info('第 %s 层展开后确认 %s 个可识别棋格', self.floor, len(layout))
        return True

    def _record_progress(self) -> None:
        assert self.before_board is not None and self.moved_node is not None
        self.ctx.withered_domain.update_context_after_move(self.before_board, self.moved_node, update_current=False)
        if self.moved_node.entry.entry_name == '业绩考察点':
            self._confirmed_merit_floor = self.floor

    def confirm_event_entry(self, entry_name: str) -> bool:
        """窄棋盘最后一格可凭同名事件中的新鲜画面确认。"""
        if (self.direction is None or self.moved_node is None or entry_name not in self.OBSERVABLE_EVENTS
                or self.moved_node.entry.entry_name != entry_name or self.last_screenshot_time <= self._last_frame_time):
            return False
        self._record_progress()
        return True

    def waits_for_battle(self) -> bool:
        return self.moved_node is not None and self.moved_node.entry.entry_name in self.BATTLE_ENTRIES

    def visible_shop_tile(self) -> bool:
        """窄棋盘只剩一个格子时，用当前商格的入口提示复核到达。"""
        if self.moved_node is None or self.moved_node.entry.entry_name != '邦布商人':
            return False
        self.navigation.last_screenshot = self.last_screenshot
        area = self.navigation._read_area('零号空洞-入口', '手柄-棋盘商人入口')
        if area is None:
            return False
        image, words = area
        entrance, prompt = words.get('进入商店'), words.get('A')
        if entrance is None or entrance.max is None or not has_filled_highlight(
            image, entrance.max.rect, 'yellow',
        ):
            return False
        target = entrance.max
        if prompt is not None and prompt.max is not None:
            return prompt.max.x < target.x and abs(prompt.max.center.y - target.center.y) <= 24
        crop = image[max(0, target.y - 20):target.y + target.h + 20, :max(0, target.x - 45)]
        return self.ctx.tm.match_template(crop, 'ios_gamepad', 'dialog_a', threshold=0.7).max is not None

    def confirm_battle_entry(self) -> bool:
        """调用方已看到真实手柄战斗 HUD 时确认进入预期战斗格。"""
        if not self.waits_for_battle() or self.last_screenshot_time <= self._last_frame_time:
            return False
        self._record_progress()
        return True

    def _merit_stop(self, board: HollowZeroMap) -> OperationRoundResult | None:
        """复用第三层考察点后的退出语义，结算前只交付观察结果。"""
        if not self.stop_on_merit or self.floor is None:
            return None
        hollow = self.ctx.withered_domain

        empty_cells = ()
        if any(node.entry.entry_name == '业绩考察点空' and node.confidence >= 0.75 for node in board.nodes):
            empty_cells = tuple(sorted(cell for cell, name in _board_cells(board).items() if name == '业绩考察点空'))
        point_visible = board.contains_entry('业绩考察点')
        phase_unconfirmed = self.floor == 3 and hollow.level_info.phase > 1 and not point_visible
        merit_taken = self.floor == 3 and not point_visible and (
            self.merit_award_floor == 3 or (phase_unconfirmed and self.merit_award_floor is not None)
        )
        if not empty_cells and not merit_taken and not phase_unconfirmed:
            self._merit_sample = None
            return None
        if (self._weekly_cap_proof is None and self.last_screenshot is not None
                and confirms_weekly_merit_cap(self.ctx.ocr.run_ocr(self.last_screenshot))):
            self._weekly_cap_proof = self.save_screenshot(prefix='macos_hollow_eval_weekly_cap')
        sample = self.floor, empty_cells, merit_taken
        if sample != self._merit_sample:
            self._merit_sample = sample
            return self.round_retry('复核业绩考察点与退出条件', wait=0.8)
        path = self.save_screenshot(prefix='macos_hollow_eval_stop_ready')
        evidence = {
            'floor': self.floor, 'empty_merit_tile': bool(empty_cells), 'screenshot': path,
            'weekly_cap_screenshot': self._weekly_cap_proof,
        }
        if self._weekly_cap_proof is None and not merit_taken:
            return self.round_fail('零号业绩领取或周上限尚未核实，已保留本局', data=evidence)
        return self.round_success(self.STATUS_READY_TO_LEAVE, data=evidence)

    @operation_node(name='加载原速刷地图', is_start_node=True, screenshot_before_round=False)
    def initialize(self) -> OperationRoundResult:
        """独立执行一格时加载原速刷配置和地图检测。"""
        if self.reuse_run:
            return self.round_success()
        config = self.ctx.run_context.get_config(
            app_id=hollow_eval_speed_const.APP_ID,
            instance_idx=self.ctx.current_instance_idx,
            group_id=application_const.DEFAULT_GROUP_ID,
        )
        if not isinstance(config, HollowEvalSpeedConfig):
            return self.round_fail('原零号业绩速刷配置未加载')
        context = self.ctx.withered_domain
        context.init_before_run(
            app_id=hollow_eval_speed_const.APP_ID, challenge_config=config.build_challenge_config(), speed_run=True,
        )
        context.init_before_hollow_start('施工废墟', config.mission_name, level=-1)
        return self.round_success()

    @node_from(from_name='加载原速刷地图')
    @operation_node(name='核对一格移动', node_max_retry_times=8)
    def move(self) -> OperationRoundResult:
        if self.last_screenshot_time <= self._last_frame_time:
            return self.round_retry('等待新的棋盘画面', wait=0.7)
        board = self.active_board()
        if board is None or board.current_idx is None:
            return self.round_retry('等待棋盘和唯一当前格', wait=0.7)
        if not self._board_settled(board):
            return self.round_wait('等待棋盘展开和稳定', wait=0.7)
        if self.direction is not None:
            assert self.before_board is not None and self.moved_node is not None
            next_floor = (
                self._before_floor is not None and self.floor == self._before_floor + 1
                and self.moved_node.entry.entry_name in ('守门人', '传送点', '诡雾')
            )
            cells = _board_cells(board)
            if not next_floor and not board_moved_one_tile(self.before, cells, self.direction):
                if not self._narrow_reward_receipt(cells):
                    return self.round_retry('等待棋盘行列或本格奖励回执', wait=0.7)
            self._record_progress()
            path = self.save_screenshot(prefix='macos_hollow_eval_step')
            return self.round_success('已确认实际走格', data={
                'key': self.KEYS[self.direction], 'before_screenshot': self.before_path, 'screenshot': path,
            })

        hollow = self.ctx.withered_domain
        if self.floor is not None and hollow.level_info.level != self.floor:
            hollow.level_info.level = self.floor
            hollow.level_info.phase = 1
            self._recent_routes.clear()
        milestone = self._merit_stop(board)
        if milestone is not None:
            return milestone
        cells = _board_cells(board)
        landmarks = tuple(sorted((cell, name) for cell, name in cells.items() if name in self.LANDMARKS))
        resources = self._reward_hud() if landmarks else None
        route_key: BoardState | None = None
        tried: set[tuple[int, int]] = set()
        if landmarks:
            route_key = self.floor, hollow.level_info.phase, landmarks, resources
            tried = {heading for state, heading in self._recent_routes if state == route_key}
        phase = self.floor, hollow.level_info.phase
        point_visible = board.contains_entry('业绩考察点')
        prioritize_exit = (
            self.floor in (1, 2) and self.merit_award_floor == self.floor
        ) or (self._mainline_phase == phase and not point_visible)
        mainline_goals = [node for name in self.MAINLINE_EXITS for node in board.nodes if node.entry.entry_name == name]
        if prioritize_exit:
            goals = mainline_goals
        else:
            goals = [
                node for node in board.nodes
                if node.entry.entry_name == '业绩考察点' and self._confirmed_merit_floor != self.floor
            ]
        priority = route_around_loop(board, goals, tried, set(hollow._get_avoid())) if goals else None
        if priority is not None:
            neighbor, direction, goal = priority
            log.info('手柄优先前往%s：%s', goal.entry.entry_name, self.KEYS[direction])
        elif prioritize_exit:
            path = self.save_screenshot(prefix='macos_hollow_eval_exit_route_unconfirmed')
            return self.round_fail('本层已有业绩回执，暂未核实可达的主线出口，已保留本局', data={'screenshot': path})
        else:
            target = hollow.get_next_to_move(board)
            neighbor = target.path_first_node if target is not None else None
            if neighbor is None or not any(board.nodes[idx] is neighbor for idx in board.edges.get(board.current_idx, ())):
                self.save_screenshot(prefix='macos_hollow_eval_unconfirmed_path')
                return self.round_fail('原寻路未确认相邻目标')
            direction = neighbor_direction(board.nodes[board.current_idx], neighbor)
            if direction is None:
                return self.round_retry('等待明确的邻格方向', wait=0.7)
            if direction in tried:
                avoid = set(hollow._get_avoid())
                bypass = route_around_loop(board, [target], tried, avoid)
                if (bypass is None and target.entry.entry_name == '诡雾' and not point_visible
                        and self._mainline_phase != phase):
                    bypass = route_around_loop(board, mainline_goals, set(), avoid)
                    if bypass is not None:
                        # 换目标只解锁一次旧退路，此后仍由当前阶段的已走方向限制。
                        self._mainline_phase = phase
                        self._recent_routes.clear()
                if bypass is None:
                    path = self.save_screenshot(prefix='macos_hollow_eval_route_loop')
                    return self.round_fail('同一棋盘的原方向已经走过，当前没有可达目标的其他方向', data={'screenshot': path})
                neighbor, direction, goal = bypass
                log.info('手柄离开重复路线，%s 朝 %s 移动', self.KEYS[direction], goal.entry.entry_name)
        can_confirm_event = neighbor.entry.entry_name in self.OBSERVABLE_EVENTS | self.BATTLE_ENTRIES
        if not can_confirm_event and len(set(cells.values())) < 2:
            return self.round_retry('等待可区分的棋盘格', wait=0.7)
        if route_key is not None:
            self._recent_routes.append((route_key, direction))
        self.before_path = self.save_screenshot(prefix='macos_hollow_eval_step_before')
        dx, dy = direction
        capsule_lane = neighbor.entry.entry_name == '扭蛋机' and cells == {direction: '扭蛋机', (dx * 2, dy * 2): '危机'}
        if capsule_lane or neighbor.entry.entry_name == '齿轮硬币箱':
            self._before_reward = resources if landmarks else self._reward_hud()
        if capsule_lane:
            self._before_resonium_count = self._board_number('手柄-棋盘鸣徽数量')
        self.before = cells
        self.before_board = board
        self._before_floor = self.floor
        self.moved_node = neighbor
        self.direction = direction
        self._last_frame_time = self.last_screenshot_time
        self.ctx.controller.btn_tap(self.KEYS[direction])
        return self.round_retry('等待走格后的新画面', wait=0.7)


class GamepadHollowEvalExplore(GamepadOperation):
    """在同一施工废墟中逐批确认走格，处理已确认的事件。"""

    def __init__(self, ctx: ZContext, max_steps: int = 12, continue_batches: bool = False) -> None:
        super().__init__(ctx, op_name='手柄探索施工废墟', timeout_seconds=900)
        self.max_steps: int = max_steps
        self.continue_batches: bool = continue_batches
        self.step: GamepadHollowEvalStep = GamepadHollowEvalStep(ctx, stop_on_merit=True)
        self.event: GamepadHollowEvalEvent = GamepadHollowEvalEvent(ctx)
        self.door: GamepadHollowEvalDoor = GamepadHollowEvalDoor(ctx)
        self.merit_receipt: GamepadHollowEvalMeritReceipt = GamepadHollowEvalMeritReceipt(ctx)
        self.merit_receipts: list[tuple[int, str]] = []
        self.completed: int = 0
        self._stalled_frames: int = 0
        self._combat_wait_until: float | None = None
        self._transition_wait_until: float | None = None

    def handle_init(self) -> None:
        self.completed = 0
        self._stalled_frames = 0
        self._combat_wait_until = None
        self._transition_wait_until = None
        self.step.handle_init()
        self.event.handle_init()
        self.door.handle_init()
        self.merit_receipt.handle_init()
        self.merit_receipts.clear()

    @operation_node(name='准备施工废墟模型', is_start_node=True, screenshot_before_round=False)
    def initialize(self) -> OperationRoundResult:
        if self.max_steps < 1:
            return self.round_fail('走格验证次数必须为正数')
        return self.step.initialize()

    def _count_step(self, keep_event: bool = False) -> OperationRoundResult | None:
        self.completed += 1
        self._stalled_frames = 0
        self.step.reset_move()
        self.door.handle_init()
        if not keep_event:
            self.event.handle_init()
        if self.completed % self.max_steps:
            return None
        if self.continue_batches:
            return self.round_wait(f'已核对{self.completed}格，继续本局')
        return self.round_success('已核对本批走格次数', data={'steps': self.completed})

    def _handle_merit_receipt(self) -> OperationRoundResult | None:
        receipt = self.merit_receipt
        receipt.last_screenshot = self.last_screenshot
        receipt.last_screenshot_time = self.last_screenshot_time
        result = receipt.try_confirm()
        if result is not None:
            if receipt.pending is None and not result.is_fail:
                self._stalled_frames += 1
                if self._stalled_frames >= 8:
                    path = self.save_screenshot(prefix='macos_hollow_eval_merit_award_unconfirmed')
                    return self.round_fail('局内零号业绩回执尚未核实，已保留现场', data={'screenshot': path})
            return result
        if receipt.pending is None:
            return None
        if self.last_screenshot_time <= receipt.requested_frame:
            return receipt.wait_for_close()
        board = self.step.active_board()
        if board is None or board.current_idx is None or self.step.floor is None or not self.step._board_settled(board):
            return receipt.wait_for_close()
        floor = self.step.floor
        amount, proof = receipt.pending
        self.merit_receipts.append((amount, proof))
        receipt.handle_init()
        checkpoint = self._count_step() if self.step.confirm_event_entry('业绩考察点') else None
        self.step.reset_move()
        self.step.merit_award_floor = floor
        self.step._confirmed_merit_floor = floor
        self._stalled_frames = 0
        return checkpoint if checkpoint is not None else self.round_wait(f'已核实本次局内获得零号业绩 {amount}')

    @node_from(from_name='准备施工废墟模型')
    @operation_node(name='处理事件并走格')
    def explore(self) -> OperationRoundResult:
        """一格完成再重新寻路，事件打开期间不继续走格。"""
        self.event.last_screenshot = self.step.last_screenshot = self.last_screenshot
        self.event.last_screenshot_time = self.step.last_screenshot_time = self.last_screenshot_time
        merit_result = self._handle_merit_receipt()
        if merit_result is not None:
            return merit_result
        self.door.last_screenshot = self.last_screenshot
        door_result = self.door.try_open()
        if door_result is not None:
            if not door_result.is_success:
                return door_result
            self.step.reset_move()
            self._combat_wait_until = None
            self._stalled_frames = 0
            return self.round_wait(door_result.status, wait=0.8)
        if is_native_battle(self.ctx, self.last_screenshot):
            checkpoint = self._count_step() if self.step.confirm_battle_entry() else None
            self._combat_wait_until = None
            result = GamepadHollowEvalBattle(self.ctx).execute()
            if not result.success:
                return self.round_by_op_result(result)
            self.ctx.controller.resume_input()
            self._stalled_frames = 0
            return checkpoint if checkpoint is not None else self.round_wait(result.status, wait=0.8)
        if self.last_screenshot is not None and self.round_by_find_area(
            self.last_screenshot, '迷失之地-挑战结果', '标题-挑战结果',
        ).is_success:
            result = GamepadHollowEvalBattleResult(self.ctx).execute()
            if not result.success:
                return self.round_by_op_result(result)
            self.ctx.withered_domain.update_to_next_level()
            self._stalled_frames = 0
            self._transition_wait_until = time.monotonic() + 30
            return self.round_wait('已确认道中胜利，等待新棋盘', wait=0.8)
        event_name = self.event.identify_entry()
        if event_name is None and self.step.visible_shop_tile():
            event_name = '邦布商人'
        if event_name is not None and self.step.confirm_event_entry(event_name):
            checkpoint = self._count_step(keep_event=True)
            if checkpoint is not None:
                return checkpoint
        previous_event = self.event._last_action
        self.event.node_retry_times = self._stalled_frames
        result = self.event.try_respond()
        if result is not None:
            if result.is_fail:
                return result
            changed = previous_event != self.event._last_action
        else:
            if self._combat_wait_until is not None:
                if time.monotonic() < self._combat_wait_until:
                    return self.round_wait('已请求进入战斗格，等待过场与手柄 HUD', wait=0.8)
                path = self.save_screenshot(prefix='macos_hollow_eval_battle_loading_unconfirmed')
                return self.round_fail('战斗格后未核实手柄 HUD', data={
                    'steps': self.completed, 'before_screenshot': self.step.before_path, 'screenshot': path,
                })
            previous_move = self.step.direction
            result = self.step.move()
            if previous_move is None and self.step.waits_for_battle():
                self._combat_wait_until = time.monotonic() + 30
            if result.is_fail:
                return result
            if result.is_success:
                if result.status == self.step.STATUS_READY_TO_LEAVE:
                    receipts = [{'amount': amount, 'screenshot': path} for amount, path in self.merit_receipts]
                    return self.round_success(result.status, data={
                        **result.data, 'steps': self.completed, 'merit_receipts': receipts,
                    })
                checkpoint = self._count_step()
                if checkpoint is not None:
                    return checkpoint
            changed = result.is_success or previous_move != self.step.direction
        self._stalled_frames = 0 if changed else self._stalled_frames + 1
        if changed:
            self._transition_wait_until = None
        if self._stalled_frames >= 8:
            if self.step.waiting_for_expansion():
                return self.round_wait('已观察到棋盘展开，等待新区域稳定', wait=0.8)
            if self._transition_wait_until is not None and time.monotonic() < self._transition_wait_until:
                return self.round_wait('已确认道中胜利，等待层间过场', wait=0.8)
            path = self.save_screenshot(prefix='macos_hollow_eval_explore_unconfirmed')
            return self.round_fail(
                result.status or '未确认施工废墟的进展', data={
                    'steps': self.completed, 'before_screenshot': self.step.before_path, 'screenshot': path,
                },
            )
        return self.round_wait(result.status)
