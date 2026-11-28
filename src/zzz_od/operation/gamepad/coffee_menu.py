"""六分街咖啡列表的手柄识别与移焦。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.focus import has_filled_highlight
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.shop import fail_shop_state, save_shop_frame

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation import Operation
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


def _read_menu_area(
    operation: Operation, area_name: str,
) -> tuple[MatLike, dict[str, MatchResultList]] | None:
    area = operation.ctx.screen_loader.get_area('咖啡店', area_name)
    if area is None or operation.last_screenshot is None:
        return None
    rect = area.rect
    image = operation.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
    if not image.size:
        return None
    texts = operation.ctx.ocr.crop_and_run_ocr(operation.last_screenshot, rect)
    return image, {''.join(text.split()): matches for text, matches in texts.items()}


def read_coffee_selection(operation: Operation) -> tuple[str, dict[str, int]] | None:
    """右侧详情必须与下方已解锁的同名杯位对应。"""
    cards = _read_menu_area(operation, '手柄-咖啡卡名')
    details = _read_menu_area(operation, '手柄-当前咖啡详情')
    if cards is None or details is None:
        return None
    known_coffees = operation.ctx.compendium_service.name_2_coffee
    positions = {
        text: matches.max.center.x
        for text, matches in cards[1].items()
        if text in known_coffees and len(matches) == 1 and matches.max is not None
    }
    selected = [text for text in details[1] if text in positions and len(details[1][text]) == 1]
    return (selected[0], positions) if len(selected) == 1 else None


def can_order_free_coffee(operation: Operation) -> bool:
    """点单同帧需要每日免费一杯、点单文字及绿色 A。"""
    row = _read_menu_area(operation, '手柄-免费点单提示')
    if row is None or not {'每日免费一杯', '点单', 'A'}.issubset(row[1]):
        return False
    prompts = row[1]['A']
    if len(prompts) != 1 or prompts.max is None:
        return False
    rect = prompts.max.rect
    prompt = row[0][rect.y1:rect.y2, rect.x1:rect.x2]
    return bool(prompt.size and cv2.countNonZero(
        cv2.inRange(cv2.cvtColor(prompt, cv2.COLOR_RGB2HSV), (54, 75, 95), (88, 255, 255)),
    ) >= 45)


def can_leave_coffee_menu(operation: Operation) -> bool:
    """已知的咖啡列表底栏明确标出 B 返回。"""
    row = _read_menu_area(operation, '手柄-咖啡列表返回')
    if row is None or '返回' not in row[1]:
        return False
    accent = row[0][:, :55]
    return bool(accent.size and cv2.countNonZero(
        cv2.inRange(cv2.cvtColor(accent, cv2.COLOR_RGB2HSV), (0, 90, 125), (18, 255, 255)),
    ) >= 45)


def coffee_already_drunk_prompt(operation: Operation) -> bool:
    """核对咖啡专属提示、单个黄色确认以及绿色 A 选择。"""
    title = _read_menu_area(operation, '手柄-咖啡已喝文案')
    if title is None or not any(
        text.rstrip('～~。！!') == '咖啡不可贪杯哦' and len(matches) == 1
        for text, matches in title[1].items()
    ):
        return False
    button = _read_menu_area(operation, '手柄-咖啡已喝确认')
    choice = _read_menu_area(operation, '手柄-咖啡已喝选择')
    if button is None or choice is None or '确认' not in button[1] or not {'A', '选择'}.issubset(choice[1]):
        return False
    if len(button[1]['确认']) != 1 or len(choice[1]['A']) != 1:
        return False
    confirm = button[1]['确认'].max
    prompt = choice[1]['A'].max
    if confirm is None or prompt is None or not has_filled_highlight(button[0], confirm.rect, 'yellow'):
        return False
    rect = prompt.rect
    accent = choice[0][rect.y1:rect.y2, rect.x1:rect.x2]
    return bool(accent.size and cv2.countNonZero(
        cv2.inRange(cv2.cvtColor(accent, cv2.COLOR_RGB2HSV), (54, 75, 95), (88, 255, 255)),
    ) >= 45)


class GamepadChooseCafeCoffee(GamepadOperation):
    """按原候选顺序将焦点移到已解锁的咖啡。"""

    def __init__(self, ctx: ZContext, candidates: list[str]) -> None:
        super().__init__(ctx, op_name='手柄选六分街咖啡', timeout_seconds=14)
        self.candidates: list[str] = candidates
        self._last_focus: str | None = None

    def handle_init(self) -> None:
        self._last_focus = None

    @operation_node(name='核对咖啡详情并移焦', is_start_node=True, node_max_retry_times=8)
    def choose(self) -> OperationRoundResult:
        state = read_coffee_selection(self)
        if state is None:
            return self.round_retry('等待咖啡卡名和当前详情', wait=0.6)
        selected, positions = state
        target = next((name for name in self.candidates if name in positions), None)
        if target is None:
            return fail_shop_state(self, '咖啡店', '已解锁列表未识别出原配置要求的咖啡')
        if selected == target:
            return self.round_success(target)
        if selected != self._last_focus:
            self._last_focus = selected
            self.ctx.controller.btn_tap('dpad_right' if positions[target] > positions[selected] else 'dpad_left')
        return self.round_retry('等待咖啡详情切换', wait=0.6)


class GamepadFinishAlreadyDrunkCoffee(GamepadOperation):
    """从明确已喝提示返回，过程中不会重新点单。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄收尾六分街已喝咖啡', timeout_seconds=20)
        self._confirmed: bool = False
        self._exit_sent: bool = False

    def handle_init(self) -> None:
        self._confirmed = self._exit_sent = False

    @operation_node(name='确认已喝提示并返回', is_start_node=True, node_max_retry_times=9)
    def finish(self) -> OperationRoundResult:
        if coffee_already_drunk_prompt(self):
            if not self._confirmed:
                if save_shop_frame(self, '咖啡店', '六分街已喝提示') is None:
                    return self.round_fail('未能保存六分街已喝提示')
                self._confirmed = True
                self.ctx.controller.btn_tap('south')
            return self.round_retry('等待已喝提示关闭', wait=0.7)
        if self._confirmed and self.last_screenshot is not None:
            if self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
                save_shop_frame(self, '咖啡店', '六分街已喝后返回')
                return self.round_success('今日已喝')
            if not self._exit_sent and read_coffee_selection(self) is not None and can_leave_coffee_menu(self):
                self._exit_sent = True
                self.ctx.controller.btn_tap('east')
        if self.node_retry_times >= 7:
            return fail_shop_state(self, '咖啡店', '六分街已喝提示或返回页面未识别')
        return self.round_retry('等待六分街已喝提示或返回页面', wait=0.7)
