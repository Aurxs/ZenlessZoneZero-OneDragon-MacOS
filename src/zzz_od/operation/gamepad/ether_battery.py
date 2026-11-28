"""从原手册资源栏进入以太电池合成，保留原计划计数语义。"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import cv2

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import debug_utils
from zzz_od.operation.gamepad.focus import has_filled_highlight, highlight_mask
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.rewards import GamepadOpenDailyCompendium
from zzz_od.operation.gamepad.transport import find_yellow_focus_frames
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.application.charge_plan.charge_plan_config import (
        ChargePlanConfig,
        ChargePlanItem,
    )
    from zzz_od.context.zzz_context import ZContext


def ether_synthesis_materials(
    words: dict[str, MatchResultList], require_sufficient: bool = True,
) -> dict[str, int] | None:
    """单枚电池必须有唯一标题、数量和当前配方的三项足额读数。"""
    inventory: list[int] = []
    quantities: list[int] = []
    materials: dict[int, list[int]] = {1: [], 60: [], 100: []}
    for text, matches in words.items():
        compact = ''.join(text.split())
        title = re.fullmatch(r'以太电池[×xX](\d+)', compact)
        quantity = re.fullmatch(r'合成数量[×xX](\d+)', compact)
        ratio = re.fullmatch(r'(\d{1,9})/(1|60|100)', compact)
        for match in matches:
            if match.center.x <= 1200:
                continue
            if title and 150 < match.center.y < 350:
                inventory.append(int(title[1]))
            if quantity and 400 < match.center.y < 600:
                quantities.append(int(quantity[1]))
            if ratio and 600 < match.center.y < 960:
                materials[int(ratio[2])].append(int(ratio[1]))
    if len(inventory) != 1 or quantities != [1] or any(len(values) != 1 for values in materials.values()):
        return None
    if require_sufficient and any(materials[required][0] < required for required in materials):
        return None
    return {'inventory': inventory[0], 'ticket': materials[1][0],
            'charge': materials[60][0], 'denny': materials[100][0]}


def _green_a_for_label(screen: MatLike, words: dict[str, MatchResultList], label: str) -> bool:
    """只认按钮左侧同排的绿色A，不借用页脚或其他按钮的提示。"""
    labels = [match for text, matches in words.items() if ''.join(text.split()) == label
              for match in matches]
    prompts = [match for text, matches in words.items() if text == 'A' for match in matches]
    pairs = [(button, prompt) for button in labels for prompt in prompts
             if abs(button.center.y - prompt.center.y) < 30
             and 0 < button.x - prompt.right_bottom.x < 200]
    if len(pairs) != 1:
        return False
    rect = pairs[0][1].rect
    crop = screen[rect.y1:rect.y2, rect.x1:rect.x2]
    return bool(crop.size and cv2.countNonZero(cv2.inRange(
        cv2.cvtColor(crop, cv2.COLOR_RGB2HSV), (54, 75, 95), (88, 255, 255),
    )) >= 45)


def _single_battery_confirmation_state(words: dict[str, MatchResultList]) -> str:
    """过渡缺字只等新帧；三项完整且明确改变配方才拒绝。"""
    expected = {'电量': '60', '储值电卡': '1', '丁尼': '100'}
    actual: dict[str, str] = {}
    for name in expected:
        labels = [match for text, matches in words.items() if text == name for match in matches]
        if len(labels) != 1:
            return 'wait'
        label = labels[0]
        numbers = [text for text, matches in words.items() if text.isdigit() for match in matches
                   if abs(match.center.x - label.center.x) < 40 and 0 < label.center.y - match.center.y < 55]
        if len(numbers) != 1:
            return 'wait'
        actual[name] = numbers[0]
    return 'ok' if actual == expected else 'invalid'


class GamepadOpenChargeResources(GamepadOperation):
    """只打开资源详情，不使用储蓄、电池或菲林。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄打开电量资源栏', timeout_seconds=40)
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '日常', '')

    @operation_node(name='打开资源手册', is_start_node=True, screenshot_before_round=False)
    def open_handbook(self) -> OperationRoundResult:
        """复用原日常手册入口，避免从未知画面发送资源快捷键。"""
        return self.round_by_op_result(GamepadOpenDailyCompendium(self.ctx).execute())

    @node_from(from_name='打开资源手册')
    @operation_node(name='进入资源栏', node_max_retry_times=4)
    def open_resources(self) -> OperationRoundResult:
        """手册标签已确认后按View进入资源选择。"""
        self.navigation.last_screenshot = self.last_screenshot
        if self.navigation._selected('手柄-顶部标签', self.navigation.TABS, 'yellow') is None:
            return self.round_retry('等待已确认的快捷手册', wait=0.8)
        self.ctx.controller.btn_tap('back')
        return self.round_success(wait=1)

    @node_from(from_name='进入资源栏')
    @operation_node(name='读取资源入口')
    def inspect(self) -> OperationRoundResult:
        """记录进入后的实际画面；本步骤不确认任何资源使用。"""
        if self.last_screenshot is None or not any(
            text.replace(' ', '') == '查看素材' for text in self.ctx.ocr.run_ocr(self.last_screenshot)
        ):
            return self.round_fail('资源选择弹窗未出现，未使用资源')
        name = self.save_screenshot(prefix='macos_charge_resources')
        return self.round_success('资源选择已打开，未使用资源', data={
            'screenshot': debug_utils.get_debug_image_path(name),
        })



class GamepadOpenEtherBatteryDetail(GamepadOperation):
    """在资源选择中核对电池名称和焦点，再进入合成来源详情。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄打开以太电池详情', timeout_seconds=55)
        self._selected: bool = False

    def handle_init(self) -> None:
        self._selected = False

    @operation_node(name='检查电池入口', is_start_node=True)
    def check_entry(self) -> OperationRoundResult:
        """已在资源选择时不重复打开手册。"""
        if self.last_screenshot is not None and '查看素材' in self.ctx.ocr.run_ocr(self.last_screenshot):
            return self.round_success('已有资源选择')
        return self.round_success()

    @node_from(from_name='检查电池入口')
    @operation_node(name='打开电池资源选择', screenshot_before_round=False)
    def open_resources(self) -> OperationRoundResult:
        return self.round_by_op_result(GamepadOpenChargeResources(self.ctx).execute())

    @node_from(from_name='检查电池入口', status='已有资源选择')
    @node_from(from_name='打开电池资源选择')
    @operation_node(name='选择以太电池详情', node_max_retry_times=10)
    def choose(self) -> OperationRoundResult:
        """只选择以太电池；电量和储蓄电量均不确认。"""
        if self.last_screenshot is None:
            return self.round_retry('等待资源截图', wait=0.7)
        words = self.ctx.ocr.run_ocr(self.last_screenshot)
        if any('合成' in text and '获取' in text for text in words) and any('以太电池' in text for text in words):
            name = self.save_screenshot(prefix='macos_ether_battery_detail')
            return self.round_success('以太电池合成来源已显示', data={'screenshot': debug_utils.get_debug_image_path(name)})
        if '查看素材' not in words or self._selected:
            return self.round_retry('等待以太电池详情', wait=0.7)
        frames = find_yellow_focus_frames(self.last_screenshot, 100, 160, 100, 160)
        names = ('电量', '以太电池', '储蓄电量')
        positions = {text.replace(' ', ''): matches.max.center.x for text, matches in words.items()
                     if text.replace(' ', '') in names and matches.max is not None}
        if len(frames) != 1 or set(positions) != set(names):
            return self.round_retry('等待明确资源焦点', wait=0.7)
        focus = min(names, key=lambda name: abs(positions[name] - frames[0].center.x))
        if abs(positions[focus] - frames[0].center.x) > 30:
            return self.round_retry('资源焦点与名称不一致', wait=0.7)
        if focus == '以太电池':
            self.ctx.controller.btn_tap('south')
            self._selected = True
        else:
            self.ctx.controller.btn_tap('dpad_right' if names.index(focus) < 1 else 'dpad_left')
        return self.round_retry('等待电池选择反馈', wait=0.8)


class GamepadPrepareEtherBattery(GamepadOperation):
    """进入以太电池合成页，准备阶段绝不确认合成。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄准备合成以太电池', timeout_seconds=80)
        self._entered: bool = False

    def handle_init(self) -> None:
        self._entered = False

    @operation_node(name='检查合成详情', is_start_node=True)
    def check_entry(self) -> OperationRoundResult:
        if self.last_screenshot is not None:
            words = self.ctx.ocr.run_ocr(self.last_screenshot)
            if any('道具处理' in text for text in words) and any('以太电池' in text for text in words):
                return self.round_success('已有合成页')
            if any('获取' in text and '合成' in text for text in words) and any('以太电池' in text for text in words):
                return self.round_success('已有电池详情')
        return self.round_success()

    @node_from(from_name='检查合成详情')
    @operation_node(name='打开以太电池详情', screenshot_before_round=False)
    def open_detail(self) -> OperationRoundResult:
        return self.round_by_op_result(GamepadOpenEtherBatteryDetail(self.ctx).execute())

    @node_from(from_name='检查合成详情', status='已有合成页')
    @node_from(from_name='检查合成详情', status='已有电池详情')
    @node_from(from_name='打开以太电池详情')
    @operation_node(name='进入以太电池合成页', node_max_retry_times=12)
    def enter_synthesis(self) -> OperationRoundResult:
        """合成来源文字必须在唯一黄色边框内，进入后只读，不合成。"""
        if self.last_screenshot is None:
            return self.round_retry('等待截图', wait=0.7)
        words = self.ctx.ocr.run_ocr(self.last_screenshot)
        if any('道具处理' in text for text in words) and any('以太电池' in text for text in words):
            name = self.save_screenshot(prefix='macos_ether_battery_prepare')
            normalized = [text.replace(' ', '') for text in words]
            quantities = [int(match[1]) for text in normalized if (match := re.fullmatch(r'合成数量[×xX]([0-9]+)', text))]
            if len(quantities) != 1:
                return self.round_retry('等待明确合成数量', wait=0.7)
            insufficient = any('合成素材不足' in text for text in normalized)
            return self.round_success('合成素材不足' if insufficient else '以太电池合成页已打开，尚未合成', data={
                'screenshot': debug_utils.get_debug_image_path(name),
                'quantity': quantities[0], 'material_insufficient': insufficient,
            })
        if not self._entered:
            contours, _ = cv2.findContours(highlight_mask(self.last_screenshot, 'yellow'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frames = []
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if 600 <= width <= 750 and 50 <= height <= 90 and cv2.contourArea(contour) / (width * height) > 0.8:
                    frames.append((x, y, x + width, y + height))
            targets = [match.rect for text, matches in words.items() if '获取' in text and '合成' in text for match in matches]
            if len(frames) == 1 and any(frames[0][0] < rect.center.x < frames[0][2] and frames[0][1] < rect.center.y < frames[0][3] for rect in targets):
                self.ctx.controller.btn_tap('south')
                self._entered = True
        return self.round_retry('等待以太电池合成页', wait=1)



class GamepadCloseEtherBattery(GamepadOperation):
    """退出电池合成及来源弹窗，不领取、使用或销毁物品。"""

    def __init__(self, ctx: ZContext, allow_obtained_receipt: bool = False) -> None:
        super().__init__(ctx, op_name='手柄退出电池合成', timeout_seconds=30)
        self.allow_obtained_receipt: bool = allow_obtained_receipt
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '日常', '')
        self._backs: int = 0
        self._receipt_dismissed: bool = False

    def handle_init(self) -> None:
        self._backs = 0
        self._receipt_dismissed = False

    @operation_node(name='返回电池合成入口', is_start_node=True, node_max_retry_times=12)
    def close(self) -> OperationRoundResult:
        """逐层识别来源页面后返回，未知页面不猜按键。"""
        if self.last_screenshot is None:
            return self.round_retry('等待返回截图', wait=0.8)
        words = self.ctx.ocr.run_ocr(self.last_screenshot)
        if {'获得', '以太电池', '1'} <= words.keys():
            if self.allow_obtained_receipt and not self._receipt_dismissed:
                confirmations = [match for text, matches in words.items() if text == '确认'
                                 for match in matches if has_filled_highlight(self.last_screenshot, match.rect, 'yellow')]
                if len(confirmations) == 1:
                    self.save_screenshot(prefix='macos_ether_battery_delayed_receipt')
                    self.ctx.controller.btn_tap('south')
                    self._receipt_dismissed = True
            return self.round_retry('等待已核账电池的延迟获得页关闭', wait=1)
        if self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            return self.round_success('已退出电池合成并返回大世界')
        self.navigation.last_screenshot = self.last_screenshot
        item_page = any('道具处理' in text for text in words) and any('以太电池' in text for text in words)
        detail = any('以太电池' in text for text in words) and any('获取' in text and '合成' in text for text in words)
        resource_detail = any(name in words for name in ('电量', '储蓄电量', '以太电池')) and any('当前拥有' in word for word in words)
        resources = '查看素材' in words
        handbook = self.navigation._selected('手柄-顶部标签', self.navigation.TABS, 'yellow') is not None
        if (item_page or detail or resource_detail or resources or handbook) and self._backs < 5:
            self.ctx.controller.btn_tap('east')
            self._backs += 1
        return self.round_retry('等待已识别页面退出', wait=1)


class GamepadSynthesizeEtherBattery(GamepadOperation):
    """一次只合成一枚，获得证据确认后才允许原计划累计。"""

    def __init__(self, ctx: ZContext, plan: ChargePlanItem, config: ChargePlanConfig) -> None:
        super().__init__(ctx, op_name='手柄合成单枚以太电池', timeout_seconds=150)
        self.plan: ChargePlanItem = plan
        self.config: ChargePlanConfig = config
        self._dismissed: bool = False
        self._requested: bool = False
        self._confirmed: bool = False
        self._obtained: bool = False
        self._shortage: bool = False
        self._item_selected: bool = False
        self._materials_before: dict[str, int] | None = None
        self._materials_after: dict[str, int] | None = None
        self._requested_at: float = 0

    def handle_init(self) -> None:
        self._dismissed = False
        self._requested = False
        self._confirmed = False
        self._obtained = False
        self._shortage = False
        self._item_selected = False
        self._materials_before = None
        self._materials_after = None
        self._requested_at = 0

    def _record_obtained(self) -> None:
        """弹窗或四项余额任一确认后只保存一次原计划计数。"""
        if self._obtained:
            return
        self.save_screenshot(prefix='macos_ether_battery_obtained')
        self.config.add_plan_run_times(self.plan)
        self._obtained = True

    def _balance_matches_synthesis(self, after: dict[str, int]) -> bool:
        """电池、票券和丁尼精确增减；六分钟内电量只容一次自然回复。"""
        before = self._materials_before
        elapsed = time.monotonic() - self._requested_at
        return bool(before is not None and 0 <= elapsed < 360
                    and after['inventory'] == before['inventory'] + 1
                    and after['ticket'] == before['ticket'] - 1
                    and after['denny'] == before['denny'] - 100
                    and before['charge'] - after['charge'] in (59, 60))

    @operation_node(name='准备单枚电池合成', is_start_node=True, screenshot_before_round=False)
    def prepare(self) -> OperationRoundResult:
        result = GamepadPrepareEtherBattery(self.ctx).execute()
        if not result.success:
            return self.round_by_op_result(result)
        if result.data['material_insufficient']:
            self._shortage = True
            return self.round_success('合成素材不足')
        if result.data['quantity'] != 1:
            return self.round_fail('合成数量不是1，未确认合成')
        return self.round_success()

    @node_from(from_name='准备单枚电池合成')
    @operation_node(name='合成并核对获得', node_max_retry_times=15)
    def synthesize(self) -> OperationRoundResult:
        """只接受单枚电池确认与获得弹窗，不确认其他道具或批量操作。"""
        if self.last_screenshot is None:
            return self.round_retry('等待合成截图', wait=0.8)
        words = self.ctx.ocr.run_ocr(self.last_screenshot)
        text = ''.join(words).replace(' ', '')
        if self._confirmed and '获得' in words and '以太电池' in words and '1' in words:
            self._record_obtained()
            if not self._dismissed and self._press_highlighted(words, '确认'):
                self._dismissed = True
            return self.round_retry('等待单枚电池获得页关闭', wait=1)
        if self._requested and any(re.fullmatch(
            r'确认使用下列素材合成1个[\[「【]以太电池[\]」】]吗[？?]?', ''.join(word.split()),
        ) for word in words):
            material_state = _single_battery_confirmation_state(words)
            if material_state == 'invalid':
                return self.round_fail('单枚电池确认页的材料数量与计划不符，未确认合成')
            if material_state != 'ok':
                return self.round_retry('等待单枚电池确认页的三项完整材料', wait=1)
            if not self._confirmed and self._press_highlighted(words, '确认'):
                self._confirmed = True
            return self.round_retry('等待单枚电池合成结果', wait=1)
        if '道具处理' in words and '以太电池' in text:
            if self._obtained:
                return self.round_success('已确认获得单枚以太电池')
            if self._confirmed:
                after = ether_synthesis_materials(words, require_sufficient=False)
                if after is not None and self._balance_matches_synthesis(after):
                    self._materials_after = after
                    self._record_obtained()
                    return self.round_success('已核对单枚电池和三项素材的实际变化')
                return self.round_retry('等待单枚电池四项余额与本笔合成一致', wait=1)
            if '合成素材不足' in text and not self._confirmed:
                self._shortage = True
                return self.round_success('合成素材不足')
            if not self._requested:
                materials = ether_synthesis_materials(words)
                if materials is None:
                    return self.round_retry('等待单枚电池名称、数量及三项足额素材', wait=1)
                panels = find_yellow_focus_frames(self.last_screenshot, 550, 750, 700, 900, min_edge_presence=0.8)
                if len(panels) == 1 and panels[0].x1 > 1200 and _green_a_for_label(self.last_screenshot, words, '合成'):
                    self.save_screenshot(prefix='macos_ether_battery_request')
                    self._materials_before = materials
                    self._requested_at = time.monotonic()
                    self.ctx.controller.btn_tap('south')
                    self._requested = True
                elif not panels and not self._item_selected:
                    cards = find_yellow_focus_frames(self.last_screenshot, 100, 160, 100, 160, min_edge_presence=0.8)
                    if len(cards) == 1 and cards[0].x2 < 1200 and _green_a_for_label(self.last_screenshot, words, '选择'):
                        self.ctx.controller.btn_tap('south')
                        self._item_selected = True
                        return self.round_retry('已选择电池，等待右侧合成面板焦点', wait=1)
            return self.round_retry('等待合成确认或获得证据', wait=1)
        return self.round_retry('等待明确的单枚电池合成页面', wait=1)

    def _press_highlighted(self, words: dict[str, MatchResultList], label: str) -> bool:
        """仅当唯一文字按钮位于黄色焦点内才按A。"""
        matches = [match for word, values in words.items() if word.replace(' ', '') == label
                   for match in values if has_filled_highlight(self.last_screenshot, match.rect, 'yellow')]
        if len(matches) != 1:
            return False
        self.save_screenshot(prefix='macos_ether_battery_confirm')
        self.ctx.controller.btn_tap('south')
        return True

    @node_from(from_name='准备单枚电池合成', status='合成素材不足')
    @node_from(from_name='合成并核对获得')
    @node_from(from_name='合成并核对获得', status='合成素材不足')
    @operation_node(name='合成后返回大世界', screenshot_before_round=False, timeout_seconds=40)
    def close(self) -> OperationRoundResult:
        result = GamepadCloseEtherBattery(self.ctx, allow_obtained_receipt=self._obtained).execute()
        if not result.success:
            return self.round_by_op_result(result)
        return self.round_success('合成素材不足' if self._shortage else '单枚电池合成完成',
                                  data={'obtained': int(self._obtained), 'material_insufficient': self._shortage,
                                        'balance_before': self._materials_before, 'balance_after': self._materials_after})
