"""根据当前交互目标接近店员，避免进入相邻 NPC 的对话。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.shop import fail_shop_state, save_shop_frame

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


def has_interaction_arrows(image: MatLike, label: Rect) -> bool:
    """识别同一行名字两侧向内的白色箭头。OCR 文本框可能已经含箭头。"""
    x1, y1 = max(0, label.x1 - 22), max(0, label.y1 - 5)
    crop = image[y1:label.y2 + 5, x1:label.x2 + 22]
    if not crop.size:
        return False
    white = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV), (0, 0, 175), (179, 44, 255))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = label.center
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (9 <= w <= 21 and 10 <= h <= 26 and 0.35 <= cv2.contourArea(contour) / (w * h) <= 0.68):
            continue
        dx, dy = x1 + x + w // 2 - center.x, y1 + y + h // 2 - center.y
        if not (16 <= abs(dx) <= 82 and abs(dy) <= 7):
            continue
        # 近箭头尖的一侧亮区行高明显小于另一侧。
        span = white[y:y + h, x:x + w]
        # 黑描边与抗锯齿会让宽端离开最外三列，按箭头宽度取两侧三分之一。
        edge_width = max(3, w // 3)
        rows_on_left = np.count_nonzero(np.any(span[:, :edge_width], axis=1))
        rows_on_right = np.count_nonzero(np.any(span[:, -edge_width:], axis=1))
        if dx < 0 and rows_on_left >= rows_on_right + 3:
            left.append((-dx, dy))
        elif dx > 0 and rows_on_right >= rows_on_left + 3:
            right.append((dx, dy))
    # OCR 有时只把一侧箭头读入文字框，允许该偏移，但仍核对两侧形状和同行。
    return sum(abs(lx - rx) <= 28 and abs(ly - ry) <= 5 for lx, ly in left for rx, ry in right) == 1


class GamepadApproachShop(GamepadOperation):
    """短步接近指定店员，直到同一名字出现可交互的双箭头。"""

    def __init__(
        self, ctx: ZContext, target: str, aliases: tuple[str, ...] = (),
        fallback_label_area: str | None = None,
    ) -> None:
        super().__init__(ctx, op_name=f'手柄接近{target}', timeout_seconds=24)
        self.target: str = target
        self.aliases: tuple[str, ...] = (target, *aliases)
        self.fallback_label_area: str | None = fallback_label_area
        self._last_action_frame: float = -1
        self._steps: int = 0
        self._turns: int = 0
        self._interacted: bool = False

    def handle_init(self) -> None:
        self._last_action_frame = -1
        self._steps = self._turns = 0
        self._interacted = False

    def _target(self) -> tuple[MatLike, Rect] | None:
        """只使用大世界店员的完整名字，不读取对话框姓名。"""
        area = self.ctx.screen_loader.get_area('大世界', '手柄-附近互动目标')
        if area is None or self.last_screenshot is None:
            return None
        rect = area.rect
        words = self.ctx.ocr.crop_and_run_ocr(self.last_screenshot, rect)
        targets = self._matching_labels(words)
        image = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
        if len(targets) == 1:
            return image, targets[0]
        if not targets and self.fallback_label_area is not None:
            label = self._read_larger_label(rect, self.fallback_label_area)
            if label is not None:
                return image, label
        return None

    def _matching_labels(self, words: dict[str, MatchResultList]) -> list[Rect]:
        return [
            match.rect
            for text, matches in words.items()
            if ''.join(text.split()).strip('><›‹▶◀') in self.aliases
            for match in matches
        ]

    def _read_larger_label(self, nearby_rect: Rect, area_name: str) -> Rect | None:
        """在已建档的小范围放大名牌，双箭头仍针对原始图像核对。"""
        area = self.ctx.screen_loader.get_area('大世界', area_name)
        if area is None or self.last_screenshot is None:
            return None
        rect = area.rect
        crop = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
        if not crop.size:
            return None
        enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        labels = self._matching_labels(self.ctx.ocr.run_ocr(enlarged))
        if len(labels) != 1:
            return None
        label = labels[0]
        x_offset, y_offset = rect.x1 - nearby_rect.x1, rect.y1 - nearby_rect.y1
        return Rect(
            x_offset + label.x1 // 2, y_offset + label.y1 // 2,
            x_offset + label.x2 // 2, y_offset + label.y2 // 2,
        )

    def _interact_ready(self) -> bool:
        """互动提示须在该按钮位置显示明亮的绿色。"""
        area = self.ctx.screen_loader.get_area('大世界', '手柄-启用互动提示')
        if area is None or self.last_screenshot is None:
            return False
        rect = area.rect
        roi = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
        if not roi.size:
            return False
        green = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV), (56, 88, 95), (82, 230, 235))
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(
            11 <= w <= 22 and 12 <= h <= 23 and 32 <= x <= 43 and 12 <= y <= 22
            for x, y, w, h in (cv2.boundingRect(contour) for contour in contours)
        )

    @operation_node(name='确认店员交互', is_start_node=True, node_max_retry_times=13)
    def approach(self) -> OperationRoundResult:
        """每帧只执行一个动作，交互后交由原应用确认实际营业或点单页面。"""
        if self.last_screenshot is None:
            return self.round_retry('等待当前店员画面', wait=0.5)
        world = self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success
        if self._interacted:
            if not world:
                return self.round_success(f'已发送{self.target}交互，等待业务核验')
            return self.round_retry('等待交互画面切换', wait=0.6)
        if not world or self.last_screenshot_time <= self._last_action_frame:
            return self.round_retry('等待新的大世界画面', wait=0.5)
        target = self._target()
        if target is None:
            if self.node_retry_times >= 3:
                return fail_shop_state(self, self.target, '未找到唯一可见的店员')
            return self.round_retry('等待可见店员姓名', wait=0.5)
        image, label = target
        if has_interaction_arrows(image, label) and self._interact_ready():
            if save_shop_frame(self, self.target, '交互前') is None:
                return self.round_fail('未能保存店员交互前画面')
            self._interacted = True
            self.ctx.controller.interact(press=True, press_time=0.2, release=True)
            return self.round_retry('等待店员交互画面', wait=0.7)

        area = self.ctx.screen_loader.get_area('大世界', '手柄-附近互动目标')
        dx = area.rect.x1 + label.center.x - self.ctx.controller.standard_width / 2
        self._last_action_frame = self.last_screenshot_time
        if abs(dx) > 115 and self._turns < 6:
            self._turns += 1
            self.ctx.controller.move_camera(0.65 if dx > 0 else -0.65, 0, min(0.21, abs(dx) / 2200))
        elif abs(dx) <= 115 and self._steps < 4:
            self._steps += 1
            self.ctx.controller.move_w(press=True, press_time=0.28, release=True)
        else:
            return fail_shop_state(self, self.target, '未能取得指定店员的交互焦点')
        return self.round_retry('等待接近店员后的新帧', wait=0.55)
