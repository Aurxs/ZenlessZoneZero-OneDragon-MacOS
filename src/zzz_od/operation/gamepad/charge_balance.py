"""用手册和仓库的独立读数核对体力资源，不操作货币图标。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict
from typing import TYPE_CHECKING, ClassVar

import cv2

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.function_guide import GamepadOpenFunction
from zzz_od.operation.gamepad.reserve_charge import ChargeResourceSnapshot
from zzz_od.operation.gamepad.rewards import (
    GamepadOpenDailyCompendium,
    _GamepadRewardOperation,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadReadChargeBalances(_GamepadRewardOperation):
    """读取手册中的电量、储蓄和电池，再读仓库顶栏的菲林。"""

    SCREEN: ClassVar[str] = '体力资源核账-手柄'

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄读取体力资源余额', timeout_seconds=110)
        self._guide: GamepadOpenFunction = GamepadOpenFunction(ctx, '仓库', scroll_to_missing=True)
        self._handbook: tuple[int, int, int] | None = None
        self._balances: ChargeResourceSnapshot | None = None
        self._guide_back_sent: bool = False

    def handle_init(self) -> None:
        self.evidence_paths.clear()
        self._handbook = None
        self._balances = None
        self._guide_back_sent = False

    def _image(self, area_name: str) -> MatLike | None:
        area = self.ctx.screen_loader.get_area(self.SCREEN, area_name)
        if area is None or self.last_screenshot is None:
            return None
        rect = area.rect
        image = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
        return image if image.size else None

    def _text(self, area_name: str) -> str:
        image = self._image(area_name)
        if image is None:
            return ''
        text = self.ctx.ocr.run_ocr_single_line(image, threshold=0.85)
        return ''.join(unicodedata.normalize('NFKC', text).split())

    def _energy(self, location: str) -> int | None:
        match = re.fullmatch(r'([0-9]{1,4})/240', self._text(f'{location}-当前电量'))
        return int(match[1]) if match else None

    def _read_handbook(self) -> tuple[int, int, int] | None:
        if not self._daily_visible():
            return None
        energy = self._energy('手册')
        reserve = self._text('手册-储蓄电量')
        batteries = self._text('手册-以太电池')
        if (energy is None or not re.fullmatch(r'[0-9]{1,4}', reserve)
                or not re.fullmatch(r'[0-9]{1,3}', batteries) or int(reserve) > 2400):
            return None
        return energy, int(reserve), int(batteries)

    def _read_storage(self) -> tuple[int, int] | None:
        title = self._text('仓库-标题')
        if title != '材料道具' and not re.fullmatch(r'(音擎仓库|驱动仓库|材料道具)\[[0-9]+/[0-9]+\]', title):
            return None
        if not self._film_icon_visible():
            return None
        energy = self._energy('仓库')
        premium = self._text('仓库-菲林')
        # 菲林栏有八位补零，缺字时不把部分读数当余额。
        if energy is None or not re.fullmatch(r'[0-9]{8}', premium):
            return None
        return energy, int(premium)

    def _film_icon_visible(self) -> bool:
        """菲林图标是银灰边框包围彩色内页，内页颜色可随动画变化。"""
        image = self._image('仓库-菲林图标')
        if image is None or image.shape[:2] != (51, 54):
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation, brightness = hsv[:, :, 1], hsv[:, :, 2]
        border = (saturation < 85) & (brightness >= 160)
        center = (saturation >= 110) & (brightness >= 155)
        return bool(
            border[1:11, 14:47].mean() >= 0.40
            and border[9:40, 40:52].mean() >= 0.12
            and border[35:47, :41].mean() >= 0.24
            and center[13:32, 8:37].mean() >= 0.45
        )

    @operation_node(name='打开日常手册核账', is_start_node=True, screenshot_before_round=False, timeout_seconds=30)
    def open_handbook(self) -> OperationRoundResult:
        return self.round_by_op_result(GamepadOpenDailyCompendium(self.ctx).execute())

    @node_from(from_name='打开日常手册核账')
    @operation_node(name='读取手册余额', node_max_retry_times=6)
    def read_handbook(self) -> OperationRoundResult:
        self._handbook = self._read_handbook()
        if self._handbook is None:
            return self.round_retry('等待手册电量、储蓄和电池读数', wait=0.8)
        if not self._save_evidence('charge_balance_handbook'):
            return self.round_fail('未能留存手册余额')
        self.ctx.controller.btn_tap('east')
        return self.round_success(wait=1)

    @node_from(from_name='读取手册余额')
    @operation_node(name='打开仓库核账', node_max_retry_times=6, timeout_seconds=55)
    def open_storage(self) -> OperationRoundResult:
        if not self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            return self.round_retry('等待手册退出', wait=0.8)
        return self.round_by_op_result(self._guide.execute())

    @node_from(from_name='打开仓库核账')
    @operation_node(name='核对仓库余额', node_max_retry_times=6)
    def read_storage(self) -> OperationRoundResult:
        storage = self._read_storage()
        if self._handbook is None or storage is None:
            return self.round_retry('等待仓库及菲林读数', wait=0.8)
        energy, reserve, batteries = self._handbook
        storage_energy, premium = storage
        if not energy <= storage_energy <= energy + 1:
            return self.round_fail('手册与仓库的电量读数不一致')
        if not self._save_evidence('charge_balance_storage'):
            return self.round_fail('未能留存仓库余额')
        self._balances = ChargeResourceSnapshot(energy, reserve, batteries, premium)
        self.ctx.controller.btn_tap('east')
        return self.round_success(wait=0.8)

    @node_from(from_name='核对仓库余额')
    @operation_node(name='核账返回大世界', node_max_retry_times=10)
    def return_to_world(self) -> OperationRoundResult:
        if self._balances is None:
            return self.round_fail('核账余额未确认')
        if self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            return self.round_success('体力资源余额已核对', data={
                'balances': asdict(self._balances),
                'evidence_paths': self.evidence_paths,
            })
        if not self._guide_back_sent and self._guide.is_guide_visible(self.last_screenshot):
            self.ctx.controller.btn_tap('east')
            self._guide_back_sent = True
        return self.round_retry('等待仓库和功能导览退出', wait=0.8)
