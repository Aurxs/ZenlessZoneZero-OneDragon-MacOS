"""零号业绩速刷的手柄选关与出战准备。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

import cv2

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.application.hollow_zero.hollow_eval_speed.hollow_eval_speed_config import (
    HollowEvalSpeedConfig,
)
from zzz_od.operation.gamepad.focus import (
    find_yellow_frames,
    has_filled_highlight,
    highlight_mask,
    text_inside_frame,
)
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.lost_void import GamepadLostVoidTransport
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:

    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadHollowEvalPrepare(GamepadOperation):
    """确认枯萎之都和施工废墟指定难度，停在原阵容准备界面。"""

    SCREEN: ClassVar[str] = '零号空洞-入口'
    STATUS_RESUME: ClassVar[str] = '已核对施工废墟内部续接'
    ZONES: ClassVar[dict[str, str]] = {
        '资质考核': 'dpad_right',
        '旧都列车': 'dpad_up',
        '施工废墟': 'south',
    }
    DIFFICULTIES: ClassVar[tuple[str, ...]] = (
        '施工废墟-前线', HollowEvalSpeedConfig.MISSION_NAME, '施工废墟-腹地', '施工废墟-核心',
    )

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄准备零号业绩速刷', timeout_seconds=150)
        self.transport: GamepadLostVoidTransport = GamepadLostVoidTransport(ctx, stop_at_hollow=True)
        self.navigation: GamepadTransportByCompendium = self.transport.navigation
        self._sent: set[str] = set()

    def handle_init(self) -> None:
        """新一次选关从画面重新确认。"""
        self._sent.clear()

    def _has_labels(self, area_name: str, *texts: str) -> bool:
        area = self.navigation._read_area(self.SCREEN, area_name)
        return area is not None and all(text in area[1] for text in texts)

    def _zone_screen(self) -> bool:
        area = self.navigation._read_area(self.SCREEN, '手柄-枯萎区域列表')
        if area is None or not self._has_labels('手柄-枯萎区域底栏', '奖励进度', '探索收集'):
            return False
        words = area[1]
        if all(zone in words for zone in self.ZONES):
            return True
        # 有保留行动时可见其他区域的锁，三个焦点的屏幕位置保持一致。
        return ('施工废墟' in words and '枯败花圃' in words
                and len(words.get('LOCKED', ())) >= 2 and len(words.get('未解锁', ())) >= 2)

    def _focused_zone(self) -> str | None:
        """浮在区域图标上方的黄箭头是焦点，区域进度灯不参与。"""
        if self.last_screenshot is None:
            return None
        focused: list[str] = []
        for zone in self.ZONES:
            area = self.ctx.screen_loader.get_area(self.SCREEN, f'手柄-{zone}焦点')
            if area is None:
                return None
            rect = area.rect
            pixels = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
            contours, _ = cv2.findContours(highlight_mask(pixels, 'yellow'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                _, _, width, height = cv2.boundingRect(contour)
                if 25 <= width <= 50 and 15 <= height <= 34 and cv2.contourArea(contour) >= 300:
                    focused.append(zone)
        return focused[0] if len(focused) == 1 else None

    def _selected_mission(self) -> str | None:
        """四个施工废墟难度标题全在时，才接受唯一整行黄框。"""
        area = self.navigation._read_area(self.SCREEN, '手柄-施工难度列表')
        if area is None or not self._has_labels('手柄-施工难度底栏', '关键敌情'):
            return None
        next_area = self.navigation._read_area(self.SCREEN, '手柄-施工下一步')
        if next_area is None or self.ctx.ocr.run_ocr_single_line(next_area[0]) != '下一步':
            return None
        image, words = area
        normalized = {text.replace('·', '-').replace(' ', ''): matches for text, matches in words.items()}
        if any(name not in normalized for name in self.DIFFICULTIES):
            return None
        frames = find_yellow_frames(image)
        if len(frames) != 1:
            return None
        selected = [
            name for name in self.DIFFICULTIES
            if normalized[name].max is not None and text_inside_frame(normalized[name].max.rect, frames)
        ]
        return selected[0] if len(selected) == 1 else None

    def _resume_prompt(self) -> tuple[bool, bool] | None:
        """返回保留任务是否匹配、确认键是否可用。"""
        if self.last_screenshot is None:
            return None
        prompt = self.navigation._read_area(self.SCREEN, '手柄-速刷续接弹窗')
        if prompt is None:
            return None
        image, words = prompt
        lines = [
            (item.center.x, word) for word, matches in words.items() for item in matches
            if item.center.y < 95
        ]
        title = ''.join(''.join(word.split()) for _, word in sorted(lines))
        if '当前正在处理' not in title or '是否继续' not in title:
            return None
        same_mission = re.fullmatch(r'当前正在处理[：:]施工废墟[·•.-]内部是否继续[？?]', title) is not None
        confirm = words.get('确认')
        can_submit = (
            confirm is not None and confirm.max is not None
            and '取消' in words and has_filled_highlight(image, confirm.max.rect, 'yellow')
            and self.round_by_find_area(self.last_screenshot, self.SCREEN, '手柄-速刷续接按键').is_success
        )
        return same_mission, can_submit

    def _is_ready(self) -> bool:
        return self._has_labels('手柄-六人编队标识', '领队', '队员') and self._has_labels(
            '手柄-六人编队底栏', '战斗设置', '出战',
        )

    def _tap_once(self, key: str, state: str) -> OperationRoundResult:
        if state not in self._sent:
            self.ctx.controller.btn_tap(key)
            self._sent.add(state)
        return self.round_retry(f'等待{state}后画面', wait=0.8)

    @operation_node(name='抵达零号空洞', is_start_node=True)
    def reach(self) -> OperationRoundResult:
        """当前可核实的入口可接续，已在准备页则重新选关确认目标。"""
        self.navigation.last_screenshot = self.last_screenshot
        self.transport.last_screenshot = self.last_screenshot
        prompt = self._resume_prompt()
        if prompt is not None:
            return self.round_success() if prompt[0] else self.round_fail('保留中的空洞任务不是施工废墟内部')
        if self.transport._at_hollow_hub() or self._zone_screen() or self._selected_mission() is not None:
            return self.round_success()
        if self._is_ready():
            return self.round_fail('当前编队准备页的副本未确认，请返回施工废墟难度列表')
        return self.round_by_op_result(self.transport.execute())

    @node_from(from_name='抵达零号空洞')
    @operation_node(name='准备施工废墟内部', node_max_retry_times=18, timeout_seconds=40)
    def prepare(self) -> OperationRoundResult:
        """逐页核实焦点，只使用当前路线需要的方向键、A 和菜单键。"""
        self.navigation.last_screenshot = self.last_screenshot
        self.transport.last_screenshot = self.last_screenshot
        prompt = self._resume_prompt()
        if prompt is not None:
            return self.round_success(self.STATUS_RESUME) if prompt[0] else self.round_fail(
                '保留中的空洞任务不是施工废墟内部',
            )
        if '进入出战准备' in self._sent and self._is_ready():
            path = self.save_screenshot(prefix='macos_hollow_eval_ready')
            return self.round_success('施工废墟内部已准备', data={'screenshot': path})
        mission = self._selected_mission()
        if mission is not None:
            target = HollowEvalSpeedConfig.MISSION_NAME
            if mission == target:
                return self._tap_once('start', '进入出战准备')
            key = 'dpad_up' if self.DIFFICULTIES.index(mission) > self.DIFFICULTIES.index(target) else 'dpad_down'
            return self._tap_once(key, f'从{mission}移焦')
        if self._zone_screen():
            zone = self._focused_zone()
            if zone is not None:
                return self._tap_once(self.ZONES[zone], f'处理{zone}焦点')
            return self.round_retry('等待已建档的区域焦点', wait=0.8)
        if self.transport._at_hollow_hub():
            if self._has_labels('手柄-枯萎之都入口', 'A'):
                return self._tap_once('south', '进入枯萎之都')
            if self._has_labels('手柄-迷失之地入口', 'A'):
                return self._tap_once('dpad_right', '移动到枯萎之都')
        return self.round_retry('等待施工废墟的已知状态', wait=0.8)


class GamepadHollowEvalDeploy(GamepadHollowEvalPrepare):
    """知悉本次选关来源后出战，等到零号空洞局内标记。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx)
        self.op_name = '手柄进入施工废墟内部'
        self.timeout_seconds = 220
        self._deployed: bool = False

    def handle_init(self) -> None:
        super().handle_init()
        self._deployed = False

    @node_from(from_name='准备施工废墟内部', status=GamepadHollowEvalPrepare.STATUS_RESUME)
    @node_from(from_name='准备施工废墟内部', status='施工废墟内部已准备')
    @operation_node(name='施工废墟出战', node_max_retry_times=45, timeout_seconds=65)
    def deploy(self) -> OperationRoundResult:
        self.navigation.last_screenshot = self.last_screenshot
        if not self._deployed:
            prompt = self._resume_prompt()
            if prompt is not None:
                if not prompt[0]:
                    return self.round_fail('保留中的空洞任务不是施工废墟内部')
                if not prompt[1]:
                    return self.round_retry('等待施工废墟内部的续接确认焦点', wait=0.8)
                key = 'south'
            elif self._is_ready():
                key = 'start'
            else:
                return self.round_retry('等待重新核对六人准备页', wait=0.8)
            self.ctx.controller.btn_tap(key)
            self._deployed = True
            return self.round_retry('等待施工废墟出战加载', wait=1)

        nav = self.navigation
        area = '零号空洞-事件'
        floor = nav._read_area(area, '手柄-走格层数')
        bag = nav._read_area(area, '手柄-走格背包')
        cards = nav._read_area(area, '手柄-速刷鸣徽')
        if (floor is not None and nav._text_rect(floor[1], '当前层数') is not None
                and bag is not None and '背包' in bag[1]) or (
            cards is not None and 'A' in cards[1] and '选择' in cards[1]
            and any('请' in text and '鸣徽' in text for text in cards[1])
        ):
            path = self.save_screenshot(prefix='macos_hollow_eval_deployed')
            return self.round_success('已进入施工废墟内部', data={'screenshot': path})
        if self.node_retry_times >= 30 and self.last_screenshot is not None:
            self.save_screenshot(prefix='macos_hollow_eval_deploy_wait')
            return self.round_fail('出战后未核实到施工废墟局内，保留当前画面')
        return self.round_retry('等待施工废墟局内标记', wait=1)
