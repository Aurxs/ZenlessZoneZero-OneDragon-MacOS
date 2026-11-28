"""大世界功能导览的手柄网格导航。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.focus import highlight_mask
from zzz_od.operation.gamepad.rewards import _GamepadRewardOperation
from zzz_od.operation.gamepad.transport import find_yellow_focus_frames

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


def themed_intel_focus(image: MatLike) -> list[Rect]:
    """粉色导览的焦点包围图标；NEW 徽章可能压住上边，仍须核对其余三边。"""
    mask = highlight_mask(image, 'yellow')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frames: list[Rect] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not (112 <= width <= 145 and 55 <= height <= 90):
            continue
        part = mask[y:y + height, x:x + width]
        if (np.count_nonzero(np.any(part[-8:], axis=0)) >= width * 0.65
                and np.count_nonzero(np.any(part[:, :8], axis=1)) >= height * 0.50
                and np.count_nonzero(np.any(part[:, -8:], axis=1)) >= height * 0.50):
            frames.append(Rect(x, y, x + width, y + height))
    return frames


class GamepadOpenFunction(_GamepadRewardOperation):
    """按功能名与唯一黄色框移焦，调用方继续确认目的页面。"""

    def __init__(self, ctx: ZContext, target: str, scroll_to_missing: bool = False) -> None:
        super().__init__(ctx, op_name=f'手柄功能导览 {target}', timeout_seconds=35)
        self.target: str = target
        self.scroll_to_missing: bool = scroll_to_missing
        self._scan_count: int = 0
        self._scan_signature: tuple[int, int, tuple[str, ...]] | None = None
        self._opened: bool = False
        self._entered: bool = False
        self._focus_unknown_count: int = 0
        self._themed_last_move: tuple[int, int, str] | None = None

    def handle_init(self) -> None:
        """重新执行时清理本次输入标记。"""
        self.evidence_paths.clear()
        self._opened = False
        self._entered = False
        self._scan_count = 0
        self._scan_signature = None
        self._focus_unknown_count = 0
        self._themed_last_move = None

    def _themed_card(self, screenshot: MatLike | None, name: str) -> tuple[MatLike, Rect] | None:
        """用同一画面的手柄提示、分组和卡片文字确认粉色导览。"""
        if name not in ('情报板', '仓库') or screenshot is None:
            return None
        cards = Rect(1225, 350, 1800, 945)
        footer = Rect(52, 979, 585, 1057)
        words = self.ctx.ocr.crop_and_run_ocr(screenshot, cards)
        if not {'COMMUNITY', '仓库', '个人主页', '情报板'}.issubset(words):
            return None
        hints = self.ctx.ocr.crop_and_run_ocr(screenshot, footer)
        if not {'导览套件', '选择', '返回'}.issubset(hints):
            return None
        target = words[name]
        if len(target) != 1 or target.max is None:
            return None
        return screenshot[cards.y1:cards.y2, cards.x1:cards.x2], target.max.rect

    def is_guide_visible(self, screenshot: MatLike | None) -> bool:
        """从当前帧识别普通或主题导览，供仓库等功能返回时核对。"""
        if screenshot is None:
            return False
        return (self.round_by_find_area(screenshot, '功能导览', '标题').is_success
                or self._themed_card(screenshot, '仓库') is not None)

    def _select_themed_card(self, image: MatLike, target: Rect) -> OperationRoundResult:
        """粉色导览文字在图标下方，只在唯一焦点和目标确切对齐时确认。"""
        frames = themed_intel_focus(image)
        if len(frames) != 1:
            return self.round_retry('等待粉色导览唯一卡片焦点，未发送输入', wait=0.8)
        focus = frames[0]
        if focus.x1 < target.center.x < focus.x2 and focus.y1 < target.center.y <= focus.y2 + 35:
            self._save_evidence('function_selected')
            self.ctx.controller.btn_tap('south')
            self._entered = True
            return self.round_retry(f'等待{self.target}打开', wait=0.7)
        if target.center.y > focus.y2 + 35:
            key = 'dpad_down'
        elif target.center.y < focus.y1:
            key = 'dpad_up'
        else:
            key = 'dpad_right' if target.center.x > focus.center.x else 'dpad_left'
        move = (focus.center.x, focus.center.y, key)
        if move != self._themed_last_move:
            self.ctx.controller.btn_tap(key)
            self._themed_last_move = move
        return self.round_retry('等待粉色导览卡片移焦', wait=0.7)

    @operation_node(name='选择功能入口', is_start_node=True, node_max_retry_times=18)
    def select_function(self) -> OperationRoundResult:
        """LT 与菜单键同时保持打开导览，每次移焦后重新读图。"""
        visible = self.round_by_find_area(self.last_screenshot, '功能导览', '标题').is_success
        themed_card = None if visible else self._themed_card(self.last_screenshot, self.target)
        if self._entered:
            if not visible and themed_card is None:
                return self.round_success('功能入口已选择')
            return self.round_retry('等待功能入口切换', wait=1)
        if themed_card is not None:
            return self._select_themed_card(*themed_card)
        if visible:
            area = self._read_area('功能导览', '手柄-卡片')
            if area is None:
                return self.round_retry('等待功能卡片', wait=0.6)
            image, words = area
            frames = find_yellow_focus_frames(image, 100, 175, 100, 180, min_edge_presence=0.65)
            target = self._text_rect(words, self.target)
            if len(frames) == 1 and target is None and self.scroll_to_missing and self._scan_count < 6:
                signature = (frames[0].x1, frames[0].y1, tuple(sorted(words)))
                if signature != self._scan_signature:
                    self.ctx.controller.btn_tap('dpad_down')
                    self._scan_signature = signature
                    self._scan_count += 1
                return self.round_retry('等待下方功能卡片显示', wait=0.8)
            if len(frames) != 1 or target is None:
                self._focus_unknown_count += 1
                if self._focus_unknown_count <= 5:
                    return self.round_retry('等待功能卡片动画及唯一焦点，未发送输入', wait=0.8)
                return self._pending_evidence('function_focus', f'未识别{self.target}或唯一功能焦点')
            self._focus_unknown_count = 0
            focus = frames[0]
            if focus.x1 <= target.center.x <= focus.x2 and focus.y1 <= target.center.y <= focus.y2:
                self._save_evidence('function_selected')
                self.ctx.controller.btn_tap('south')
                self._entered = True
            elif target.center.y > focus.y2:
                self.ctx.controller.btn_tap('dpad_down')
            elif target.center.y < focus.y1:
                self.ctx.controller.btn_tap('dpad_up')
            else:
                self.ctx.controller.btn_tap('dpad_right' if target.center.x > focus.center.x else 'dpad_left')
            return self.round_retry('等待功能焦点或页面切换', wait=0.7)
        if self.round_by_find_area(self.last_screenshot, '菜单-更多功能', '标题-更多功能').is_success:
            self.ctx.controller.btn_tap('east')
            return self.round_retry('从更多功能返回菜单', wait=1)
        menu = self._read_area('菜单', '手柄-底部应用')
        if menu is not None and all(self._text_rect(menu[1], name) is not None for name in ('代理人', '更多')):
            self.ctx.controller.btn_tap('east')
            return self.round_retry('从菜单返回大世界', wait=1)
        if self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            if not self._opened:
                self.ctx.controller.gamepad.tap_combo(['left_trigger', 'start'])
                self._opened = True
            return self.round_retry('等待功能导览', wait=1)
        return self._pending_evidence('function_entry', '未识别功能导览入口')
