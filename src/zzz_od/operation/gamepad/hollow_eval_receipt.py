"""仅从单条明确的游戏回执确认本周零号业绩上限。"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cv2

from zzz_od.operation.gamepad.focus import has_filled_highlight
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

_CAP_TEXT = re.compile(
    r'(?:(?:本周|每周)零号业绩(?:获取|获得)?已达(?:到)?上限'
    r'|零号业绩(?:获取|获得)?已达(?:到)?(?:本周|每周)上限'
    r'|已达(?:到)?(?:本周|每周)零号业绩(?:获取|获得)?上限)[。！!]?'
)
_WEEKLY_PROGRESS = re.compile(r'(?:本周|每周)零号业绩(?:获取|获得)?[：:]?(\d{1,6})[/／](\d{1,6})')


def confirms_weekly_merit_cap(texts: Iterable[str]) -> bool:
    """要求明确已满文案或单条具名周进度，拒绝条件说明和其他货币。"""
    progress: set[tuple[int, int]] = set()
    for text in texts:
        normalized = ''.join(text.split())
        if _CAP_TEXT.fullmatch(normalized):
            return True
        if match := _WEEKLY_PROGRESS.fullmatch(normalized):
            progress.add((int(match[1]), int(match[2])))
    if len(progress) != 1:
        return False
    current, limit = progress.pop()
    return limit > 0 and current == limit


class GamepadHollowEvalMeritReceipt(GamepadOperation):
    """核对局内已到账的零号业绩，先留图，再请求关闭回执。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄核实局内零号业绩')
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', '周期征讨')
        self.pending: tuple[int, str] | None = None
        self.requested_frame: float = -1
        self.requested_at: float = 0

    def handle_init(self) -> None:
        self.pending = None
        self.requested_frame = -1
        self.requested_at = 0

    def try_confirm(self) -> OperationRoundResult | None:
        if self.last_screenshot is None:
            return None
        self.navigation.last_screenshot = self.last_screenshot
        panel = self.navigation._read_area('零号空洞-入口', '手柄-局内业绩获得')
        if panel is None:
            return None
        image, words = panel
        title, label = words.get('获得'), words.get('零号业绩')
        if title is None or label is None or title.max is None or label.max is None:
            return None
        if self.pending is not None:
            return self.wait_for_close()
        item = label.max
        # 获得页使用较大的物品图标。按实际牌面缩放复用原业绩模板。
        card = image[max(0, item.y - 154):item.y - 31, max(0, item.center.x - 79):item.center.x + 79]
        normalized = cv2.resize(card, None, fx=0.85, fy=0.85) if card.size else None
        icon = self.ctx.tm.match_template(normalized, 'hollow', 'challenge_reward_eval', threshold=0.74).max if normalized is not None else None
        if icon is None or abs(icon.center.x - normalized.shape[1] / 2) > 20:
            return self.round_retry('等待核实局内零号业绩图标', wait=0.6)
        counts = {
            int(text) for text, matches in words.items() if re.fullmatch(r'[1-9][0-9]{0,3}', text)
            and any(abs(word.center.x - item.center.x) <= 25 and 1 <= item.y - word.y - word.h <= 35
                    for word in matches)
        }
        confirm, prompt, select = words.get('确认'), words.get('A'), words.get('选择')
        if len(counts) != 1 or confirm is None or confirm.max is None or select is None or select.max is None:
            return self.round_retry('等待局内零号业绩数量及确认提示', wait=0.6)
        if not has_filled_highlight(image, confirm.max.rect, 'yellow'):
            return self.round_retry('等待局内零号业绩的黄确认焦点', wait=0.6)
        if not (prompt is not None and prompt.max is not None and prompt.max.x + prompt.max.w < select.max.x
                and abs(prompt.max.center.y - select.max.center.y) <= 20):
            return self.round_retry('等待局内零号业绩确认的同排 A', wait=0.6)
        path = self.save_screenshot(prefix='macos_hollow_eval_merit_award')
        if not path:
            return self.round_fail('未能留存局内零号业绩回执，已保留获得页')
        self.pending = next(iter(counts)), path
        self.requested_frame = self.last_screenshot_time
        self.requested_at = time.monotonic()
        self.ctx.controller.btn_tap('south')
        return self.round_wait('已留存局内零号业绩回执，等待获得页关闭', wait=0.7)

    def wait_for_close(self) -> OperationRoundResult:
        if self.pending is not None and time.monotonic() - self.requested_at >= 8:
            return self.round_fail('未核实零号业绩获得页关闭，已保留现场', data={'screenshot': self.pending[1]})
        return self.round_wait('等待零号业绩回执关闭并返回棋盘', wait=0.7)
