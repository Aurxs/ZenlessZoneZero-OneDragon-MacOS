"""施工废墟红 S 门的原生手柄开门提示。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.focus import has_filled_highlight
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadHollowEvalDoor(GamepadOperation):
    """只有侧邻时游戏显示已知 S 门选项，A 只请求开门一次。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄打开施工废墟 S 门', timeout_seconds=12)
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', '周期征讨')
        self._requested_at: float | None = None

    def handle_init(self) -> None:
        self._requested_at = None

    def try_open(self) -> OperationRoundResult | None:
        """核实开门项、守门人条件、黄焦及左侧绿色 A。"""
        if self.last_screenshot is None:
            return None
        self.navigation.last_screenshot = self.last_screenshot
        area = self.navigation._read_area('零号空洞-入口', '手柄-棋盘S门提示')
        if area is None:
            return None
        image, words = area
        title = words.get('开门')
        condition = next((
            matches.max for text, matches in words.items()
            if text.replace('『', '').replace('』', '').replace('「', '').replace('」', '').replace(' ', '')
            == '上一场守门人战斗获得S评价可通过'
        ), None)
        if title is None or title.max is None or condition is None:
            return None
        if self._requested_at is not None:
            if time.monotonic() - self._requested_at >= 8:
                path = self.save_screenshot(prefix='macos_hollow_eval_door_open_unconfirmed')
                return self.round_fail('S 门开门请求后提示仍未关闭', data={'screenshot': path})
            return self.round_wait('等待 S 门开门回应', wait=0.7)
        target = title.max
        if not (0 < condition.center.y - target.center.y <= 40 and has_filled_highlight(image, target.rect, 'yellow')):
            return self.round_retry('等待 S 门开门黄焦', wait=0.7)
        prompt = image[max(0, target.y - 6):condition.y + condition.h, :max(0, target.x - 28)]
        key = self.ctx.tm.match_template(prompt, 'ios_gamepad', 'dialog_a', threshold=0.70).max if prompt.size else None
        if key is None:
            return self.round_retry('等待 S 门开门的同排 A 提示', wait=0.7)
        path = self.save_screenshot(prefix='macos_hollow_eval_door_open_request')
        self._requested_at = time.monotonic()
        self.ctx.controller.btn_tap('south')
        return self.round_success('已请求打开 S 门，等待新画面', data={'screenshot': path})

    @operation_node(name='核实并请求打开 S 门', is_start_node=True, node_max_retry_times=6)
    def open(self) -> OperationRoundResult:
        result = self.try_open()
        return result if result is not None else self.round_retry('等待侧邻 S 门的开门提示', wait=0.7)
