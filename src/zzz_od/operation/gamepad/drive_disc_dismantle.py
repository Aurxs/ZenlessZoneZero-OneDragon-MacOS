"""按原配置在驱动仓库快选并拆解未锁定的驱动盘。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.application.drive_disc_dismantle.drive_disc_dismantle_config import (
    DismantleLevelEnum,
)
from zzz_od.operation.gamepad.focus import has_filled_highlight, highlight_mask
from zzz_od.operation.gamepad.function_guide import GamepadOpenFunction
from zzz_od.operation.gamepad.rewards import _GamepadRewardOperation
from zzz_od.operation.gamepad.transport import find_yellow_focus_frames

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.geometry.rectangle import Rect
    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


@dataclass(frozen=True)
class QuickDiscSelection:
    """本帧快速选择弹窗的焦点、已选条件和匹配数量。"""

    focus: str
    selected: frozenset[str]
    count: int


class GamepadDriveDiscDismantle(_GamepadRewardOperation):
    """每次输入前核对当前页面，结算后核对实际驱动盘库存减少量。"""

    STORAGE: ClassVar[str] = '仓库-驱动仓库'
    WENGINE: ClassVar[str] = '仓库-音擎仓库'
    DISMANTLE: ClassVar[str] = '仓库-驱动仓库-驱动盘拆解'
    ABANDON: ClassVar[str] = '全选已弃置'
    OPTIONS: ClassVar[tuple[str, ...]] = (ABANDON, 'B', 'A及以下', 'S及以下')

    def __init__(
        self, ctx: ZContext, level: str, abandon: bool, resume_receipt: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(ctx, op_name='手柄拆解驱动盘', timeout_seconds=180)
        self.level: str = level
        self.abandon: bool = abandon
        self.resume_receipt: tuple[int, int] | None = resume_receipt
        self.warehouse: GamepadOpenFunction = GamepadOpenFunction(ctx, '仓库', scroll_to_missing=True)
        self._sent: set[str] = set()
        self._action_frame: float = -1
        self._receipt_frame: float = -1
        self._before: int | None = None
        self._after: int | None = None
        self._limit: int | None = None
        self._expected: int = 0
        self._filter_move: tuple[str, str] | None = None

    def handle_init(self) -> None:
        """按运行重新建立前态证据和一次性操作记录。"""
        self.evidence_paths.clear()
        self._sent.clear()
        self._action_frame = -1
        self._receipt_frame = -1
        self._before = None
        self._after = None
        self._limit = None
        self._expected = 0
        self._filter_move = None

    def _fresh_frame(self) -> bool:
        return self.last_screenshot is not None and self.last_screenshot_time > self._action_frame

    def _tap(self, key: str) -> None:
        self._action_frame = self.last_screenshot_time
        self.ctx.controller.btn_tap(key)

    def _tap_once(self, step: str, key: str) -> None:
        if step not in self._sent:
            self._sent.add(step)
            self._tap(key)

    def _wait_for(self, phase: str, reason: str, attempts: int = 6) -> OperationRoundResult:
        if self.node_retry_times >= attempts:
            return self._pending_evidence(phase, reason)
        return self.round_retry(reason, wait=0.6)

    def _labels(self, screen: str, area_name: str) -> set[str]:
        area = self._read_area(screen, area_name)
        return {''.join(label.split()).upper() for label in area[1]} if area else set()

    def _quantity(self, screen: str, area_name: str, name: str) -> tuple[int, int] | None:
        """只接受带准确页面名称、完整分数和有效容量的库存数字。"""
        expected = rf'{name}\[?(\d{{1,4}})/(\d{{1,4}})\]?'
        values = [re.fullmatch(expected, word) for word in self._labels(screen, area_name)]
        found = [(int(item[1]), int(item[2])) for item in values if item]
        if len(found) == 1 and 0 <= found[0][0] <= found[0][1] and found[0][1] > 0:
            return found[0]
        return None

    def _selected_quantity(self) -> tuple[int, int] | None:
        return self._quantity(self.DISMANTLE, '标题-驱动盘拆解', '驱动盘拆解')

    def _equipped_hidden(self) -> bool:
        words = self._labels(self.DISMANTLE, '手柄-装备显示状态')
        return {'显示已装备', 'OFF'} <= words or '显示已装备OFF' in words

    @operation_node(name='打开仓库', is_start_node=True, screenshot_before_round=False)
    def open_warehouse(self) -> OperationRoundResult:
        """从功能导览打开仓库，音擎页只允许切换页签。"""
        if self.resume_receipt is not None:
            count, capacity = self.resume_receipt
            if not 0 < count <= capacity:
                return self.round_fail('拆解续接数量无效')
            self._expected, self._limit = count, capacity
            return self.round_success('续接拆解获得')
        if self.level not in {level.value.value for level in DismantleLevelEnum}:
            return self.round_fail('驱动盘拆解等级设置无效')
        result = self.warehouse.execute()
        return self.round_success('已打开仓库') if result.success else self.round_by_op_result(result)

    @node_from(from_name='打开仓库', status='已打开仓库')
    @operation_node(name='进入驱动盘拆解', node_max_retry_times=12)
    def enter_dismantle(self) -> OperationRoundResult:
        """只有驱动仓库的拆解入口可以收到菜单键。"""
        if not self._fresh_frame():
            return self._wait_for('disc_entry', '等待仓库画面的新帧', 12)
        quantity = self._selected_quantity()
        if quantity is not None and '进入拆解' in self._sent:
            if quantity[0] != 0 or not self._equipped_hidden():
                return self._pending_evidence('disc_entry', '驱动盘拆解已有选择或未确认已装备隐藏')
            self._limit = quantity[1]
            return self.round_success()
        inventory = self._quantity(self.STORAGE, '标题-驱动仓库', '驱动仓库')
        if inventory is not None and '拆解' in self._labels(self.STORAGE, '手柄-拆解入口'):
            self._before = inventory[0]
            self._save_evidence('disc_pre_inventory')
            self._tap_once('进入拆解', 'start')
        elif (self._quantity(self.WENGINE, '标题-音擎仓库', '音擎仓库') is not None
                and 'RB' in self._labels(self.WENGINE, '手柄-右肩键提示')):
            self._tap_once('切换驱动仓库', 'right_shoulder')
        return self._wait_for('disc_entry', '等待驱动盘拆解页', 12)

    @staticmethod
    def _label_rects(words: dict[str, MatchResultList], label: str) -> list[Rect]:
        """逐个保留匹配框，同名按键与选项不得合并。"""
        return [match.rect for word, matches in words.items()
                if ''.join(word.split()).upper().rstrip('：:') == label for match in matches]

    @classmethod
    def _has_prompt(cls, words: dict[str, MatchResultList], button: str, action: str) -> bool:
        """按键字母必须在操作文字同一行的左侧。"""
        hints = [(key, label) for key in cls._label_rects(words, button)
                 for label in cls._label_rects(words, action)
                 if abs(key.center.y - label.center.y) <= 15 and 0 < label.x1 - key.x2 < 120]
        return len(hints) == 1

    def _lock_rule_visible(self) -> bool:
        """从游戏说明核对未装备、未锁定两个条件及快选容量。"""
        result = self._read_area(self.DISMANTLE, '手柄-拆解规则正文')
        if result is None:
            return False
        _, words = result
        text = {re.sub(r'[\s\[\]【】.,，。:：、]', '', line) for line in words}
        conditions = {
            '驱动拆解说明',
            '拆解条件',
            '驱动盘在满足以下条件时可以被拆解',
            '1驱动盘未被装备',
            '2驱动盘未被标记为锁定状态',
            '处于锁定状态的驱动盘不可直接拆解如需拆解需要取消锁定状态',
        }
        return (conditions <= text
                and any(f'每次最多可以选择{self._limit}个驱动盘' in line for line in text)
                and self._has_prompt(words, 'B', '返回'))

    def _option_state(self, image: MatLike, rect: Rect) -> bool | None:
        """以蓝色实心和灰色底分别确认已选与未选。"""
        if has_filled_highlight(image, rect, 'blue'):
            return True
        y, x = rect.center.y, rect.x2
        probe = image[y - 4:y + 4, x + 20:x + 45]
        if probe.size and np.mean(np.max(probe, axis=2) < 80) >= 0.9:
            return False
        return None

    @classmethod
    def _matching_count(cls, words: dict[str, MatchResultList]) -> tuple[int, int] | None:
        """兼容候选数字独立识别或与完整数量标签同框。"""
        matches = [(count, match.rect) for label, items in words.items()
                   if (count := re.fullmatch(
                       r'符合以上条件的驱动盘数量[：:]?(\d{0,4})', ''.join(label.split()),
                   )) is not None
                   for match in items]
        if len(matches) != 1:
            return None
        match, rect = matches[0]
        if match[1]:
            return int(match[1]), rect.center.y
        values = [int(label) for label, items in words.items()
                  if re.fullmatch(r'\d{1,4}', label)
                  for item in items
                  if abs(item.center.y - rect.center.y) < 12 and abs(item.x - rect.x2) < 24]
        return (values[0], rect.center.y) if len(values) == 1 else None

    def _quick_selection(self) -> QuickDiscSelection | None:
        """核对弹窗标识、手柄提示、四项状态和唯一焦点。"""
        result = self._read_area(self.DISMANTLE, '手柄-快速选择弹窗')
        if result is None:
            return None
        image, words = result
        required = ('快速选择', '在已弃置的驱动盘中选择', '在未弃置的驱动盘中选择')
        if not all(self._label_rects(words, label) for label in required):
            return None
        if not all(self._has_prompt(words, button, action)
                   for button, action in (('X', '确认'), ('A', '选择'), ('B', '返回'))):
            return None
        count = self._matching_count(words)
        if count is None:
            return None
        minimum_y = self._label_rects(words, '在已弃置的驱动盘中选择')[0].center.y
        frames = find_yellow_focus_frames(image, 300, 345, 38, 62, min_edge_presence=0.57)
        if len(frames) != 1:
            return None
        focus: list[str] = []
        selected: set[str] = set()
        for label in self.OPTIONS:
            rects = [rect for rect in self._label_rects(words, label) if minimum_y < rect.center.y < count[1]]
            if len(rects) != 1 or (state := self._option_state(image, rects[0])) is None:
                return None
            rect = rects[0]
            if frames[0].x1 < rect.center.x < frames[0].x2 and frames[0].y1 < rect.center.y < frames[0].y2:
                focus.append(label)
            if state:
                selected.add(label)
        if len(focus) == 1:
            return QuickDiscSelection(focus[0], frozenset(selected), count[0])
        return None

    @classmethod
    def _filter_direction(cls, focus: str, target: str) -> str:
        """沿已确认的左列和同一行移焦，右侧选项先回到左列。"""
        if focus == 'A及以下':
            return 'dpad_left'
        if target == 'A及以下' and focus == 'B':
            return 'dpad_right'
        left = (cls.ABANDON, 'B', 'S及以下')
        row = left.index('B' if target == 'A及以下' else target)
        return 'dpad_up' if left.index(focus) > row else 'dpad_down'

    @node_from(from_name='进入驱动盘拆解')
    @operation_node(name='按原配置快速选择', node_max_retry_times=18)
    def choose_discs(self) -> OperationRoundResult:
        """应用前核对所有选项和候选数，零候选从弹窗返回。"""
        reason = '等待快速选择及唯一焦点'
        if not self._fresh_frame():
            return self._wait_for('disc_filter', reason, 18)
        choices = self._quick_selection()
        if choices is None:
            if (self._selected_quantity() == (0, self._limit) and self._equipped_hidden()
                    and {'RT', '快速选择'} <= self._labels(self.DISMANTLE, '手柄-快速选择提示')):
                self._tap_once('打开筛选', 'right_trigger')
            return self._wait_for('disc_filter', reason, 18)

        desired = {self.level}
        if self.abandon:
            desired.add(self.ABANDON)
        change = next((option for option in self.OPTIONS if (option in desired) != (option in choices.selected)), None)
        if change is not None:
            if choices.focus == change:
                self._tap_once(f'切换筛选 {change}', 'south')
                self._filter_move = None
            else:
                key = self._filter_direction(choices.focus, change)
                signature = (choices.focus, key)
                if signature != self._filter_move:
                    self._tap(key)
                    self._filter_move = signature
            return self._wait_for('disc_filter', reason, 18)
        if self._limit is None:
            return self._pending_evidence('disc_filter', '未确认单次拆解容量')
        self._expected = min(choices.count, self._limit)
        if self._expected and not self._save_evidence('disc_quick_nonempty'):
            return self.round_fail('未保存非零驱动盘候选')
        self._tap('west' if choices.count else 'east')
        return self.round_success(wait=0.7)

    @node_from(from_name='按原配置快速选择')
    @operation_node(name='核对拆解候选', node_max_retry_times=10)
    def check_selected(self) -> OperationRoundResult:
        """确认选中数，只有正向操作需要打开并核对锁定保护说明。"""
        reason = '等待选中数、已装备状态和锁定保护说明'
        if not self._fresh_frame():
            return self._wait_for('disc_selected', reason, 10)
        if '打开拆解说明' in self._sent and '关闭拆解说明' not in self._sent:
            if self._lock_rule_visible():
                self._tap_once('关闭拆解说明', 'east')
            return self._wait_for('disc_selected', reason, 10)
        if self._selected_quantity() == (self._expected, self._limit) and self._equipped_hidden():
            if self._expected == 0:
                return self.round_success('无候选')
            if '关闭拆解说明' in self._sent:
                return self.round_success('有候选')
            if 'LT' in self._labels(self.DISMANTLE, '手柄-规则提示'):
                self._tap_once('打开拆解说明', 'left_trigger')
        return self._wait_for('disc_selected', reason, 10)

    def _is_dismantle_enabled(self) -> bool:
        """有清晰的白色拆解按钮才可打开确认。"""
        result = self._read_area(self.DISMANTLE, '手柄-拆解按钮')
        if result is None:
            return False
        image, words = result
        rects = self._label_rects(words, '拆解')
        if len(rects) != 1:
            return False
        rect = rects[0]
        text = image[rect.y1:rect.y2, rect.x1:rect.x2]
        return bool(text.size and np.mean(np.min(text, axis=2) >= 190) >= 0.12)

    @node_from(from_name='核对拆解候选', status='有候选')
    @operation_node(name='打开拆解确认', node_max_retry_times=6)
    def open_dismantle_confirmation(self) -> OperationRoundResult:
        """在说明正向通过后再核对选中数与启用的驱动盘拆解键。"""
        if not self._fresh_frame():
            return self._wait_for('disc_confirm', '等待打开拆解确认')
        if ('关闭拆解说明' not in self._sent or self._before is None or not 0 < self._expected <= self._before
                or self._selected_quantity() != (self._expected, self._limit)
                or not self._equipped_hidden() or not self._is_dismantle_enabled()):
            return self._wait_for('disc_confirm', '等待已核对的驱动盘拆解按钮')
        if not self._save_evidence('disc_candidates'):
            return self.round_fail('未保存驱动盘拆解前态')
        self._tap_once('打开拆解确认', 'start')
        return self.round_success(wait=0.7)

    def _is_a_dismantle_confirmation(self) -> bool:
        """确认只接受已建档的A级文案、黄色确认按钮及 A 提示。"""
        words = self._labels(self.DISMANTLE, '手柄-拆解确认正文')
        body = {text.rstrip('，,。！？?!') for text in words}
        if body != {'选中的驱动盘稀有度包含A级', '确认要分解吗'}:
            return False
        button = self._read_area(self.DISMANTLE, '手柄-拆解确认文字')
        prompt = self._read_area(self.DISMANTLE, '手柄-拆解确认选择提示')
        return bool(
            button is not None and prompt is not None
            and len(self._label_rects(button[1], '确认')) == 1
            and np.mean(highlight_mask(button[0], 'yellow') > 0) >= 0.70
            and self._has_prompt(prompt[1], 'A', '选择')
        )

    @node_from(from_name='打开拆解确认')
    @operation_node(name='检查拆解确认', node_max_retry_times=8)
    def inspect_dismantle_confirmation(self) -> OperationRoundResult:
        """仅确认本次打开的驱动盘分解弹窗，未知品质停止留图。"""
        reason = '等待已建档的A级拆解确认或结果'
        if not self._fresh_frame() or '打开拆解确认' not in self._sent:
            return self._wait_for('disc_confirm', reason, 8)
        if self._is_dismantle_receipt() or (
            self._selected_quantity() == (0, self._limit) and self._equipped_hidden()
        ):
            return self.round_success()
        if self._is_a_dismantle_confirmation():
            if not self._save_evidence('disc_confirm'):
                return self.round_fail('未保存驱动盘确认前态')
            self._tap_once('确认拆解', 'south')
            return self.round_success(wait=0.7)
        return self._wait_for('disc_confirm', reason, 8)

    def _is_dismantle_receipt(self) -> bool:
        """按返还母盘核对数量，获得标题淡入时等待白色前景稳定。"""
        result = self._read_area(self.DISMANTLE, '手柄-拆解获得正文')
        if result is None:
            return False
        image, words = result
        titles = self._label_rects(words, '获得')
        if len(titles) != 1:
            return False
        title = titles[0]
        text = image[title.y1:title.y2, title.x1:title.x2]
        if not text.size or np.mean(np.min(text, axis=2) >= 202) < 0.15:
            return False
        mothers = [item.rect for label, items in words.items() if label.endswith('母盘') for item in items]
        counts = [(int(label), item.rect) for label, items in words.items()
                  if re.fullmatch(r'\d{1,4}', label) for item in items]
        returned = 0
        for mother in mothers:
            quantities = [quantity for quantity, rect in counts if abs(rect.center.x - mother.center.x) <= 25
                          and 0 <= mother.y1 - rect.y2 <= 35]
            if len(quantities) != 1 or quantities[0] <= 0:
                return False
            returned += quantities[0]
        if not mothers or returned != self._expected:
            return False
        button = self._read_area(self.DISMANTLE, '手柄-拆解获得确认')
        prompt = self._read_area(self.DISMANTLE, '手柄-拆解获得选择提示')
        return bool(button is not None and prompt is not None
                    and len(self._label_rects(button[1], '确认')) == 1
                    and np.mean(highlight_mask(button[0], 'yellow') > 0) >= 0.60
                    and self._has_prompt(prompt[1], 'A', '选择'))

    @node_from(from_name='检查拆解确认')
    @node_from(from_name='打开仓库', status='续接拆解获得')
    @operation_node(name='核对拆解结算', node_max_retry_times=10)
    def check_dismantle_result(self) -> OperationRoundResult:
        """处理获得页，回到拆解页后要求选中数已经归零。"""
        reason = '等待拆解获得与选中数归零'
        if not self._fresh_frame() or (self.resume_receipt is None and '打开拆解确认' not in self._sent):
            return self._wait_for('disc_result', reason, 10)
        if self._is_dismantle_receipt():
            if self._receipt_frame < 0:
                self._receipt_frame = self.last_screenshot_time
            elif self.last_screenshot_time - self._receipt_frame >= 0.25:
                if not self._save_evidence('disc_reward'):
                    return self.round_fail('未保存驱动盘拆解获得')
                self._tap_once('关闭拆解获得', 'south')
        else:
            self._receipt_frame = -1
            may_finish = self.resume_receipt is None or '关闭拆解获得' in self._sent
            if may_finish and self._selected_quantity() == (0, self._limit) and self._equipped_hidden():
                return self.round_success()
        return self._wait_for('disc_result', reason, 10)

    @node_from(from_name='核对拆解候选', status='无候选')
    @node_from(from_name='核对拆解结算')
    @operation_node(name='核对库存并返回', node_max_retry_times=10)
    def leave_storage(self) -> OperationRoundResult:
        """只在辨认出驱动仓库、功能导览或大世界后推进返回。"""
        reason = '等待驱动仓库、功能导览或大世界'
        if not self._fresh_frame():
            return self._wait_for('disc_return', reason, 10)
        if '退出驱动仓库' in self._sent:
            if self.warehouse.is_guide_visible(self.last_screenshot):
                self._tap_once('关闭功能导览', 'east')
            elif self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
                if self.resume_receipt is not None:
                    status = f'已核对 {self._expected} 个驱动盘的拆解返还'
                else:
                    status = f'已拆解 {self._expected} 个驱动盘' if self._expected else '无可拆解驱动盘'
                return self.round_success(status, data={'dismantled': self._expected, 'inventory_after': self._after})
        elif (self._selected_quantity() == (0, self._limit)
              and {'B', '返回'} <= self._labels(self.DISMANTLE, '手柄-返回提示')):
            self._tap_once('退出拆解', 'east')
        elif '退出拆解' in self._sent:
            stock = self._quantity(self.STORAGE, '标题-驱动仓库', '驱动仓库')
            if stock is not None:
                if self.resume_receipt is None and (self._before is None or stock[0] != self._before - self._expected):
                    return self._pending_evidence('disc_return', '实际库存减少量与拆解数量不符')
                if {'B', '返回'} <= self._labels(self.STORAGE, '手柄-返回提示'):
                    self._after = stock[0]
                    self._save_evidence('disc_post_inventory')
                    self._tap_once('退出驱动仓库', 'east')
        return self._wait_for('disc_return', reason, 10)
