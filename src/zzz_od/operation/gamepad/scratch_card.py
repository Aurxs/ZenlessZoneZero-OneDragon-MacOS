"""六分街报刊亭签到的原生手柄流程。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

import cv2

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.dialog import GamepadChooseDialogOption
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.rewards import _GamepadRewardOperation
from zzz_od.operation.gamepad.shop import fail_shop_state, save_shop_frame
from zzz_od.operation.gamepad.shop_approach import GamepadApproachShop
from zzz_od.operation.gamepad.transport import GamepadTransport

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


def _kiosk_area(operation: GamepadOperation, area: str) -> bool:
    """只匹配报刊亭平台资源中已登记的区域。"""
    return operation.last_screenshot is not None and operation.round_by_find_area(
        operation.last_screenshot, '报刊亭', area,
    ).is_success


class GamepadScratchCardTransport(GamepadTransport):
    """地图保留用户选择的是否直达选项，同时确认两种已知落地。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, '六分街', '报刊亭')

    @node_from(from_name='确认手柄地图传送')
    @operation_node(name='等待手柄传送落地', node_max_retry_times=30, timeout_seconds=40)
    def wait_for_world(self) -> OperationRoundResult:
        result = super().wait_for_world()
        if result.is_success:
            return result
        if _kiosk_area(self, '手柄-报纸标题') and _kiosk_area(self, '手柄-刮刮卡提示'):
            return self.round_success('已直接进入报刊亭')
        return result


class GamepadScratchCard(_GamepadRewardOperation):
    """从原签到选项进入报刊亭，核对首次奖励或当日已领取提示。"""

    STATUS_ALREADY_REWARDED: ClassVar[str] = '今日同类奖励已领取'
    STATUS_READY_TO_SCRATCH: ClassVar[str] = '刮刮卡可刮取'
    STATUS_CARD_REVEALED: ClassVar[str] = '刮刮卡已揭开'
    SCRATCH_STEP_SECONDS: ClassVar[float] = 0.045

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄报刊亭签到', timeout_seconds=190)
        self._sent_card: bool = False
        self._confirmed: bool = False
        self._exit_sent: bool = False
        self._opened_wake_options: bool = False
        self._chose_wake_option: bool = False
        self._confirmed_wake_result: bool = False
        self._scratch_rounds: int = 0
        self._last_scratch_frame: float = -1
        self._reward_confirmed: bool = False
        self._resumed_revealed_card: bool = False
        self._card_return_sent: bool = False

    def handle_init(self) -> None:
        self.evidence_paths.clear()
        self._sent_card = self._confirmed = self._exit_sent = False
        self._opened_wake_options = self._chose_wake_option = self._confirmed_wake_result = False
        self._scratch_rounds = 0
        self._last_scratch_frame = -1
        self._reward_confirmed = False
        self._resumed_revealed_card = self._card_return_sent = False

    def _world_visible(self) -> bool:
        return self.last_screenshot is not None and self.round_by_find_area(
            self.last_screenshot, '大世界-普通', '快捷手册',
        ).is_success

    def _already_today(self) -> bool:
        """先认同类签到专用文案，再确认背景仍是报刊亭。"""
        return _kiosk_area(self, '手柄-同类奖励已领') and _kiosk_area(self, '手柄-报纸标题')

    def _words(self, area_name: str) -> list[str]:
        area = self._read_area('报刊亭', area_name)
        return list(area[1]) if area else []

    @staticmethod
    def _dialogue_text(text: str) -> str:
        spoken_text = ''.join(text.split()).translate(str.maketrans('', '', '，,。…．.？！?!(（)）'))
        return spoken_text.replace('嗷鸣', '嗷呜')

    def _howl_sleeping(self) -> bool:
        return any(
            self._dialogue_text(text) == '嗷呜睡得很香甜要不要惊动它呢'
            for text in self._words('手柄-嗷呜对话正文')
        )

    def _howl_awakened(self) -> bool:
        return any(
            self._dialogue_text(text) == '嗷呜被你叫醒了'
            for text in self._words('手柄-嗷呜醒来文案')
        )

    def _scratch_ready(self) -> bool:
        if not (_kiosk_area(self, '手柄-刮刮卡标题') and _kiosk_area(self, '手柄-每日可刮取一次')):
            return False
        prompts = {''.join(text.split()) for text in self._words('手柄-右杆刮卡提示')}
        return {'R', '刮一刮'} <= prompts or 'R刮一刮' in prompts

    def _card_revealed(self) -> bool:
        """揭开的卡片底栏只剩靠中央的 B 返回。"""
        if not (_kiosk_area(self, '手柄-刮刮卡标题') and _kiosk_area(self, '手柄-每日可刮取一次')):
            return False
        prompts = {''.join(text.split()) for text in self._words('手柄-右杆刮卡提示')}
        return (
            ('B返回' in prompts or {'B', '返回'} <= prompts)
            and not any('刮一刮' in text or text == 'R' for text in prompts)
            and self._key_visible('手柄-揭开卡片返回按键', (0, 90, 110), (17, 255, 255))
        )

    def can_resume_open_card(self) -> bool:
        """已确认的半刮卡和揭开结果均可继续原应用。"""
        return self._scratch_ready() or self._card_revealed()

    def _advance_wake_dialogue(self) -> OperationRoundResult | None:
        """只选择叫醒嗷呜，并确认两段专属 A 提示。"""
        green = ((56, 88, 95), (82, 230, 235))
        if self._howl_sleeping():
            options = self._words('手柄-嗷呜对话选项')
            wake_options = [text for text in options if self._dialogue_text(text) == '叫醒他']
            if len(wake_options) == 1 and not self._chose_wake_option:
                result = GamepadChooseDialogOption(
                    self.ctx, wake_options[0], single_confirmation=True,
                ).execute()
                if not result.success:
                    return self.round_by_op_result(result)
                self._chose_wake_option = True
            elif (
                not options and not self._opened_wake_options and not self._chose_wake_option
                and self._key_visible('手柄-嗷呜对话按键', *green)
            ):
                self._opened_wake_options = True
                self.ctx.controller.btn_tap('south')
            return self.round_retry('等待嗷呜的叫醒选项或提示', wait=0.7)
        if self._howl_awakened():
            if not self._confirmed_wake_result and self._key_visible('手柄-嗷呜醒来按键', *green):
                self._confirmed_wake_result = True
                self.ctx.controller.btn_tap('south')
            return self.round_retry('等待报纸页打开', wait=0.7)
        return None

    def _key_visible(self, area_name: str, min_hsv: tuple[int, int, int], max_hsv: tuple[int, int, int]) -> bool:
        area = self.ctx.screen_loader.get_area('报刊亭', area_name)
        if area is None or self.last_screenshot is None:
            return False
        rect = area.rect
        crop = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
        return bool(crop.size and cv2.countNonZero(
            cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV), min_hsv, max_hsv),
        ) >= 65)

    @operation_node(name='进入报刊亭或恢复已知页面', is_start_node=True, timeout_seconds=110)
    def transport(self) -> OperationRoundResult:
        if _kiosk_area(self, '手柄-报纸标题') or self.can_resume_open_card() or self._howl_sleeping() or self._howl_awakened():
            return self.round_success('已在报刊亭')
        if not self._world_visible():
            return fail_shop_state(self, '报刊亭', '报刊亭签到起点未识别')
        return self.round_by_op_result(GamepadScratchCardTransport(self.ctx).execute())

    @node_from(from_name='进入报刊亭或恢复已知页面')
    @operation_node(name='接近嗷呜', node_max_retry_times=3, timeout_seconds=32)
    def approach(self) -> OperationRoundResult:
        if _kiosk_area(self, '手柄-报纸标题') or self.can_resume_open_card() or self._howl_sleeping() or self._howl_awakened():
            return self.round_success()
        if not self._world_visible():
            return self.round_retry('等待报刊亭或大世界', wait=0.6)
        # 当前移动端 OCR 在到站与已选中画面都将名字读为“嗷鸣”。
        return self.round_by_op_result(GamepadApproachShop(
            self.ctx, '嗷呜', aliases=('嗷鸣',), fallback_label_area='手柄-嗷呜名牌',
        ).execute())

    @node_from(from_name='接近嗷呜')
    @operation_node(name='打开刮刮卡并核对状态', node_max_retry_times=12, timeout_seconds=38)
    def open_card(self) -> OperationRoundResult:
        if self._already_today():
            return self.round_success(self.STATUS_ALREADY_REWARDED)
        if self._scratch_ready():
            return self.round_success(self.STATUS_READY_TO_SCRATCH)
        if self._card_revealed():
            self._confirmed = self._resumed_revealed_card = True
            return self.round_success(self.STATUS_CARD_REVEALED)
        if self._sent_card:
            if self.node_retry_times >= 9:
                return fail_shop_state(self, '报刊亭', '未识别刮刮卡页的右摇杆提示或已领取提示')
            return self.round_retry('等待刮刮卡页面或已领取提示', wait=0.7)
        dialogue = self._advance_wake_dialogue()
        if dialogue is not None:
            return dialogue
        if (
            _kiosk_area(self, '手柄-报纸标题')
            and _kiosk_area(self, '手柄-刮刮卡提示')
            and self._key_visible('手柄-刮刮卡按键', (85, 70, 95), (125, 255, 255))
        ):
            if save_shop_frame(self, '报刊亭', '打开刮刮卡前') is None:
                return self.round_fail('未能保存刮刮卡入口画面')
            self._sent_card = True
            self.ctx.controller.btn_tap('west')
        return self.round_retry('等待报刊亭 X 刮刮卡提示或打开结果', wait=0.7)

    @node_from(from_name='打开刮刮卡并核对状态', status=STATUS_READY_TO_SCRATCH)
    @operation_node(name='用右摇杆刮取奖励', node_max_retry_times=13, timeout_seconds=45)
    def scratch(self) -> OperationRoundResult:
        """仅对当前已确认的卡分段操作，领奖前要看到获得回执。"""
        if self._scratch_rounds and self._known_reward_visible():
            return self.round_success('已确认报刊亭获得回执')
        if not self.ctx.run_context.is_context_running:
            return self.round_retry('等待签到任务恢复', wait=0.8)
        if self._scratch_ready() and self.last_screenshot_time > self._last_scratch_frame:
            if self._scratch_rounds >= 2:
                return fail_shop_state(self, '报刊亭', '刮取次数到上限，未收到获得回执')
            if self._scratch_rounds == 0 and save_shop_frame(self, '报刊亭', '首次刮取前') is None:
                return self.round_fail('未能保存首次刮取前画面')
            self._last_scratch_frame = self.last_screenshot_time
            self._scratch_rounds += 1
            self._sweep_card()
        if self.node_retry_times >= 7:
            return fail_shop_state(self, '报刊亭', '未确认报刊亭本次刮取的结果')
        return self.round_retry('等待完整刮取结果或获得回执', wait=0.8)

    def _sweep_card(self) -> None:
        """连续设置右摇杆幅度，让刮卡指针沿网格逐行移动。"""
        controller = self.ctx.controller
        levels = tuple(step / 10 for step in range(-10, 11))
        rows = tuple(step / 4 for step in range(-4, 5))
        columns_first = self._scratch_rounds > 1
        try:
            for index, row in enumerate(rows):
                across = levels if index % 2 == 0 else reversed(levels)
                for column in across:
                    if not self.ctx.run_context.is_context_running:
                        return
                    x, y = (row, column) if columns_first else (column, row)
                    controller.set_gamepad_stick('right', x, y)
                    time.sleep(self.SCRATCH_STEP_SECONDS)
        finally:
            controller.set_gamepad_stick('right', 0, 0)

    @node_from(from_name='打开刮刮卡并核对状态', status=STATUS_ALREADY_REWARDED)
    @node_from(from_name='打开刮刮卡并核对状态', status=STATUS_CARD_REVEALED)
    @node_from(from_name='用右摇杆刮取奖励')
    @operation_node(name='确认签到结果并返回大世界', node_max_retry_times=10, timeout_seconds=25)
    def finish(self) -> OperationRoundResult:
        if self._scratch_rounds and self._known_reward_visible():
            if not self._confirmed:
                if save_shop_frame(self, '报刊亭', '首次刮取奖励') is None:
                    return self.round_fail('未能保存报刊亭获得回执')
                self._confirmed = self._reward_confirmed = True
                self.ctx.controller.btn_tap('south')
            return self.round_retry('等待报刊亭获得弹窗关闭', wait=0.7)
        if self._already_today():
            if self._confirmed:
                return self.round_retry('等待已领取提示关闭', wait=0.6)
            if not (
                self.round_by_find_area(self.last_screenshot, '画面-通用', '手柄-单按钮确认').is_success
                and _kiosk_area(self, '手柄-确认选择')
                and self._key_visible('手柄-确认按键', (56, 88, 95), (82, 230, 235))
            ):
                return self.round_retry('等待同类奖励的手柄确认提示', wait=0.6)
            if save_shop_frame(self, '报刊亭', '同类奖励已领') is None:
                return self.round_fail('未能保存报刊亭已领取画面')
            self._confirmed = True
            self.ctx.controller.btn_tap('south')
            return self.round_retry('已确认同类奖励提示', wait=0.7)
        if self._confirmed and self._world_visible():
            save_shop_frame(self, '报刊亭', '签到后返回大世界')
            if self._reward_confirmed:
                return self.round_success('刮刮卡奖励已领取')
            if self._resumed_revealed_card:
                return self.round_success('刮刮卡已完成')
            return self.round_success('今日已领取')
        if self._confirmed and self._card_revealed():
            if not self._card_return_sent:
                self._card_return_sent = True
                self.ctx.controller.btn_tap('east')
            return self.round_retry('等待揭开的卡片关闭', wait=0.7)
        if (
            self._confirmed and not self._exit_sent
            and _kiosk_area(self, '手柄-报纸标题')
            and _kiosk_area(self, '手柄-返回提示')
            and self._key_visible('手柄-返回按键', (0, 90, 110), (17, 255, 255))
        ):
            self._exit_sent = True
            self.ctx.controller.btn_tap('east')
        if self.node_retry_times >= 7:
            return fail_shop_state(self, '报刊亭', '未能确认从报刊亭返回大世界')
        return self.round_retry('等待报刊亭或大世界返回状态', wait=0.7)
