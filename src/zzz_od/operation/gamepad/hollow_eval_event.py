"""手柄零号业绩速刷的鸣徽与右侧事件交互。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

import cv2

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.game_data.agent import AgentEnum
from zzz_od.hollow_zero.event.call_for_support import CallForSupport
from zzz_od.hollow_zero.event.normal_event_handler import is_hp_loss_option
from zzz_od.hollow_zero.event.resonium_utils import choose_resonium_by_priority
from zzz_od.operation.gamepad.focus import has_filled_highlight, highlight_mask
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.geometry.point import Point
    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext
    from zzz_od.game_data.agent import Agent
    from zzz_od.hollow_zero.game_data.hollow_zero_event import (
        HallowZeroEvent,
        HallowZeroNormalEventOption,
    )


class GamepadHollowEvalEvent(GamepadOperation):
    """沿原速刷优先级选鸣徽和可离开的事件选项。"""

    SCREEN: ClassVar[str] = '零号空洞-事件'
    SUPPORT: ClassVar[str] = '呼叫增援'
    STORE: ClassVar[str] = '邦布商人'
    CAMP_FIRE: ClassVar[str] = '篝火如家般温暖'
    ACADEMIC_AID: ClassVar[str] = '怀斯塔学会的援助'
    DETECTIVE: ClassVar[str] = '旧都失物奇怪的侦探'
    EMERGENCY_REPORT: ClassVar[str] = '特殊区域播报'
    SPECIAL_ZONE: ClassVar[str] = '特殊区域'
    TIMED_TASK_EXIT: ClassVar[str] = '放弃限时任务'
    RESOURCE_COST_WORDS: ClassVar[tuple[str, ...]] = (
        '支付', '花费', '消耗', '购买', '出售', '给与', '给予', '交出', '扣除', '扣减',
        '承受', '损失', '失去', '资源可能受损', '摧毁', '借款', '借用', '还款',
    )

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄处理零号空洞事件', timeout_seconds=80)
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', '周期征讨')
        self._last_action: tuple[str, ...] | None = None
        self._last_frame_time: float = -1
        self._handled: bool = False
        self._event_frame: MatLike | None = None
        self._event_area: tuple[MatLike, dict[str, MatchResultList]] | None = None
        self._support_agent: Agent | None = None
        self._support_position: int | None = None
        self._support_slot_sent: bool = False
        self._support_resolved: bool = False

    def handle_init(self) -> None:
        self._last_action = None
        self._last_frame_time = -1
        self._handled = False
        self._event_frame = None
        self._event_area = None
        self._support_agent = None
        self._support_position = None
        self._support_slot_sent = False
        self._support_resolved = False

    def _read(self, name: str) -> tuple[MatLike, dict[str, MatchResultList]] | None:
        self.navigation.last_screenshot = self.last_screenshot
        return self.navigation._read_area(self.SCREEN, name)

    def _read_event_panel(self) -> tuple[MatLike, dict[str, MatchResultList]] | None:
        """步进取证与事件操作共用同一帧的右侧文字。"""
        if self._event_frame is not self.last_screenshot:
            self._event_area = self._read('手柄-速刷右侧事件')
            self._event_frame = self.last_screenshot
        return self._event_area

    def identify_entry(self) -> str | None:
        """只回报能从当前画面确认的原棋盘格类型。"""
        area = self._read_event_panel()
        if area is None:
            return None
        words = area[1]
        if self._is_store(words):
            return self.STORE
        if self._has_support_heading(words) and any(
            '响应了求援信息' in word for word in words
        ):
            return self.SUPPORT
        if self._camp_stage(words) is not None:
            return '邂逅'
        if self.ACADEMIC_AID in words and self._academic_aid_stage(words) is not None:
            return '邂逅'
        if self._detective_title(words) and self._detective_stage(words) is not None:
            event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.DETECTIVE)
            return event.entry_name if event is not None else None
        if (self._has_narrative_title(words, self.EMERGENCY_REPORT) and self._emergency_body(words)
                and not _yellow_rectangles(area[0], (440, 740), (56, 150))):
            event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.EMERGENCY_REPORT)
            return event.entry_name if event is not None else None
        if self._is_special_zone_exit(words):
            event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.SPECIAL_ZONE)
            return event.entry_name if event is not None else None
        event = self._configured_list_event(words)
        if event is not None and event.event_name != '投机客' and self._configured_rows(event, words):
            return event.entry_name
        return None

    @staticmethod
    def _has_support_heading(words: dict[str, MatchResultList]) -> bool:
        return any(word.rstrip('!！') == '呼叫增援' for word in words)

    @staticmethod
    def _is_store(words: dict[str, MatchResultList]) -> bool:
        return {'鸣徽交易', '鸣徽催化'} <= words.keys() and any(
            '欢迎光临' in word and '本店只收齿轮硬币' in word for word in words
        )

    @classmethod
    def _camp_stage(cls, words: dict[str, MatchResultList]) -> str | None:
        if cls.CAMP_FIRE not in words:
            return None
        if any('小队在基地中找到了一处可以休整一下的区域' in word for word in words):
            return '营地休整'
        if cls._team_recovered(words):
            return '全队恢复'
        return None

    @staticmethod
    def _team_recovered(words: dict[str, MatchResultList]) -> bool:
        return (any('重伤的队员得到了治疗' in word for word in words)
                and any('队回复20%生命值' in word for word in words))

    @classmethod
    def _academic_aid_stage(cls, words: dict[str, MatchResultList]) -> str | None:
        if cls._team_recovered(words):
            return '学会营地恢复'
        for stage, phrase in (
            ('抵达安全点', '小队抵达了一处可当作临时营地的安全点'),
            ('学会人员离开', '营地里还有些遗留的物资'),
            ('学会离别后休整', '小队在临时营地里稍作休整'),
        ):
            if any(phrase in word for word in words):
                return stage
        return None

    @classmethod
    def _detective_title(cls, words: dict[str, MatchResultList]) -> bool:
        return cls._has_narrative_title(words, cls.DETECTIVE)

    @staticmethod
    def _normalized_event_text(text: str) -> str:
        return re.sub(r'[「」『』《》〈〉\s·：:]', '', text)

    @classmethod
    def _has_narrative_title(cls, words: dict[str, MatchResultList], title: str) -> bool:
        """按右侧上方标题位置读取，忽略书名号和间隔。"""
        return any(
            matches.max is not None and 215 <= matches.max.center.y <= 267
            and cls._normalized_event_text(word) == title
            for word, matches in words.items()
        )

    @staticmethod
    def _detective_stage(words: dict[str, MatchResultList]) -> str | None:
        for stage, phrase in (
            ('见到侦探', '一位身着侦探服饰身形高大的人'),
            ('收到数据卡', '递来一张通用音频数据卡'),
            ('侦探道别', '这与我想寻找的线索相去甚远'),
        ):
            if any(phrase in word for word in words):
                return stage
        return None

    @classmethod
    def _emergency_body(cls, words: dict[str, MatchResultList]) -> str:
        """以纯播报正文去重，有编号字段时核对位数。"""
        body = cls._narrative_text(words)
        if '紧急委托' in body and re.search(r'UC[0-9]{4}(?![0-9])', body) is None:
            return ''
        if '区域代号' in body and re.search(r'区域代号[0-9]{4}(?![0-9])', body) is None:
            return ''
        return body

    @classmethod
    def _is_special_zone_exit(cls, words: dict[str, MatchResultList]) -> bool:
        heading = words.get(cls.SPECIAL_ZONE)
        return (
            heading is not None and heading.max is not None and 105 <= heading.max.center.y <= 155
            and any('即将离开特殊区域' in word and '确认后将无法返回该区域' in word
                    and '是否继续' in word for word in words)
        )

    def _tap(self, key: str, signature: tuple[str, ...]) -> OperationRoundResult:
        if signature != self._last_action and self.last_screenshot_time > self._last_frame_time:
            self.ctx.controller.btn_tap(key)
            self._last_action = signature
            self._last_frame_time = self.last_screenshot_time
            self._handled = True
        return self.round_retry('等待事件输入后的新状态', wait=0.7)

    def _pending(self, reason: str) -> OperationRoundResult:
        if self.node_retry_times >= 3:
            self.save_screenshot(prefix='macos_hollow_eval_event_unconfirmed')
            return self.round_fail(reason)
        return self.round_retry(reason, wait=0.8)

    def _resonium(self) -> OperationRoundResult | None:
        area = self._read('手柄-速刷鸣徽')
        if area is None:
            return None
        image, words = area
        if '获得战利品' in words:
            return self._resonium_received()
        title = next((word for word, matches in words.items() if matches.max is not None
                      and matches.max.center.y < 95 and '鸣徽' in word and '请' in word), None)
        names = sorted(
            [(word, matches.max.rect) for word, matches in words.items()
             if matches.max is not None and 360 < matches.max.center.y < 430
             and re.fullmatch(r'\[[^\]]+\].+', word)],
            key=lambda item: item[1].center.x,
        )
        button = self.navigation._text_rect(words, '选择')
        if title is None:
            if not names:
                return None
            refresh = self._read('手柄-速刷鸣徽刷新')
            if refresh is None or not {'RS', '刷新'} <= refresh[1].keys():
                return None
            title = '战后鸣徽'
        if len(names) not in (1, 2, 3) or button is None:
            return self._pending('未确认完整鸣徽选项与焦点')
        prompt_center = self._card_prompt_center(image, words, button)
        if prompt_center is None:
            return self._pending('等待鸣徽卡片下方的 A 提示')
        frames = [
            frame for frame in _yellow_rectangles(image, (280, 390), (425, 565))
            if frame.x1 < prompt_center.x < frame.x2 and -20 < prompt_center.y - frame.y2 < 100
        ]
        if len(frames) != 1 or abs(prompt_center.y - button.center.y) > 20:
            return self._pending('鸣徽 A 提示与焦点不一致')
        frame = frames[0]
        selected = [index for index, (_, rect) in enumerate(names) if frame.x1 < rect.center.x < frame.x2]
        if len(selected) != 1:
            return self._pending('鸣徽选中卡片不明确')

        candidates: list[str] = [name for name, _ in names]
        priorities = self.ctx.withered_domain.challenge_config.resonium_priority
        parsed = []
        for name in candidates:
            category, label = name[1:].split(']', 1)
            resonium = self.ctx.withered_domain.data_service.match_resonium_by_ocr(category, label.strip())
            if resonium is None:
                if priorities:
                    return self._pending('无法按原优先级匹配当前鸣徽')
                parsed.clear()
                break
            parsed.append(resonium)

        order = choose_resonium_by_priority(parsed, priorities) if parsed else []
        target_idx = order[0] if order else selected[0]
        current_idx = selected[0]
        key = 'south' if target_idx == current_idx else ('dpad_right' if target_idx > current_idx else 'dpad_left')
        return self._tap(key, (title, candidates[current_idx], *candidates))

    def _resonium_received(self) -> OperationRoundResult:
        """单张鸣徽已加入背包时核对原鸣徽表与同钮 A「确定」。"""
        name = self._read('手柄-速刷战利品鸣徽')
        footer = self._read('手柄-速刷战利品确定')
        if name is None or footer is None or len(name[1]) != 1:
            return self._pending('等待单张战利品鸣徽和确定按钮')
        text = next(iter(name[1]))
        label = re.fullmatch(r'\[([^\]]+)\](.+)', text)
        known = self.ctx.withered_domain.data_service.cate_2_resonium
        if label is None or not any(r.name == label[2] for r in known.get(label[1], ())):
            return self._pending('未核对到原鸣徽表中的战利品')
        button = self.navigation._text_rect(footer[1], '确定')
        if button is not None and has_filled_highlight(footer[0], button, 'yellow'):
            prompt = self._card_prompt_center(footer[0], footer[1], button)
            if prompt is not None and 0 < button.center.x - prompt.x < 125 and abs(button.center.y - prompt.y) < 20:
                return self._tap('south', ('获得战利品', text))
        return self._pending('等待单张鸣徽的同钮 A 确定提示')

    def _card_prompt_center(
        self, image: MatLike, words: dict[str, MatchResultList], button: Rect,
    ) -> Point | None:
        """A 漏读时仅在「选择」左侧匹配原绿色按钮模板。"""
        prompt = words.get('A')
        if prompt is not None and prompt.max is not None:
            return prompt.max.center
        top = max(0, button.y1 - 15)
        left = max(0, button.x1 - 90)
        crop = image[top:button.y2 + 15, left:button.x1]
        match = self.ctx.tm.match_template(crop, 'ios_gamepad', 'dialog_a', threshold=0.7).max
        if match is None:
            return None
        center = match.center
        center.x += left
        center.y += top
        return center

    def _right_event(self) -> OperationRoundResult | None:
        area = self._read_event_panel()
        if area is None:
            return None
        image, words = area
        if self._detective_title(words):
            event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.DETECTIVE)
            stage = self._detective_stage(words)
            if (event is not None and stage is not None and not _yellow_rectangles(image, (440, 740), (56, 150))
                    and self.round_by_find_area(
                        self.last_screenshot, self.SCREEN, '手柄-速刷继续提示',
                    ).is_success):
                return self._tap('south', (self.DETECTIVE, stage))
            return self._pending('等待旧都失物的已知正文和内联 A')
        if self._is_store(words):
            footer = self._read('手柄-速刷商店返回')
            if footer is None or not {'B', '返回'} <= footer[1].keys():
                return self._pending('等待邦布商人返回提示')
            return self._tap('east', (self.STORE, '返回'))
        if self._has_support_heading(words):
            return self._support_event(image, words)
        if self.CAMP_FIRE in words:
            stage = self._camp_stage(words)
            if stage is not None and self.round_by_find_area(
                self.last_screenshot, self.SCREEN, '手柄-速刷继续提示',
            ).is_success:
                return self._tap('south', (self.CAMP_FIRE, stage))
            return self._pending('等待篝火休整正文与继续 A 提示')
        if self.ACADEMIC_AID in words:
            return self._academic_aid(image, words)
        if self._has_narrative_title(words, self.EMERGENCY_REPORT):
            return self._emergency_report(image, words)
        if self._is_special_zone_exit(words):
            return self._special_zone_exit(image, words)
        frames = _yellow_rectangles(image, (440, 740), (56, 150))
        event = self._configured_list_event(words)
        if event is not None:
            if frames:
                return self._choose_right_option(event.event_name, image, words, frames)
            narrative = self._configured_narrative(event, words)
            if narrative and self.round_by_find_area(
                self.last_screenshot, self.SCREEN, '手柄-速刷继续提示',
            ).is_success:
                return self._tap('south', (event.event_name, narrative))
        headings = [
            word[1:-1] for word, matches in words.items() if re.fullmatch(r'「[^」]{1,12}」', word)
            and matches.max is not None and matches.max.center.y < 330
        ]
        if len(headings) != 1:
            return None
        title = headings[0]
        if frames:
            return self._choose_right_option(title, image, words, frames)
        if self.round_by_find_area(self.last_screenshot, self.SCREEN, '手柄-速刷继续提示').is_success:
            return self._tap('south', (title, *sorted(words)))
        return self._pending('未确认右侧对话的继续提示或选项')

    def _configured_list_event(self, words: dict[str, MatchResultList]) -> HallowZeroEvent | None:
        """列表标题必须在右栏标题位置与原事件完全一致。"""
        titles = {
            self._normalized_event_text(word) for word, matches in words.items()
            if matches.max is not None and 145 <= matches.max.x <= 229
            and 47 <= matches.max.center.y <= 267 and matches.max.h <= 39
        }
        if not titles:
            return None
        matches = [
            event for event in self.ctx.withered_domain.data_service.normal_events
            if event.options and self._normalized_event_text(event.event_name) in titles
        ]
        return matches[0] if len(matches) == 1 else None

    def _configured_narrative(self, event: HallowZeroEvent, words: dict[str, MatchResultList]) -> str:
        """用已定位的显示页标题和逐行正文区分同一事件的各页。"""
        if not self._has_narrative_title(words, self._normalized_event_text(event.event_name)):
            return ''
        if self._configured_rows(event, words):
            return ''
        return self._narrative_text(words)

    @staticmethod
    def _narrative_text(words: dict[str, MatchResultList]) -> str:
        """通过正文区的左边缘识别段落，保留同段被 OCR 拆开的词。"""
        body = sorted(
            (matches.max.center.y, matches.max.x, word) for word, matches in words.items()
            if matches.max is not None and matches.max.x >= 195
            and 280 <= matches.max.center.y <= 411
        )
        if not any(x <= 246 for _, x, _ in body):
            return ''
        text = re.sub(r'[\W_]+', '', ''.join(word for _, _, word in body))
        return text if re.search(r'[\u4e00-\u9fff]{6,}', text) else ''

    def _configured_rows(
        self, event: HallowZeroEvent, words: dict[str, MatchResultList],
    ) -> list[tuple[HallowZeroNormalEventOption, str, tuple[str, ...]]]:
        """选项全名及原配置效果须同时落入各自行，保留原配置顺序。"""
        rows: list[tuple[HallowZeroNormalEventOption, str, tuple[str, ...]]] = []
        for option in event.options:
            aliases = {self._normalized_event_text(option.ocr_word)}
            expected_description = option.desc
            if event.event_name == '事件坍塌的房屋' and option.option_name == '钻入狭缝通道':
                aliases.add('钻入夹缝通道')
            if event.event_name == '邦布的秘宝' and option.option_name == '人为财死邦布也是':
                aliases.update(('人为财死，邦布也是', '人为财死,邦布也是'))
                expected_description = self.TIMED_TASK_EXIT
            labels = [
                text for text, matches in words.items() if self._normalized_event_text(text) in aliases
                and matches.max is not None and 278 <= matches.max.x <= 467
                and 265 < matches.max.center.y < 695
            ]
            if len(labels) != 1:
                continue
            label = labels[0]
            name = words[label].max
            if name is None:
                continue
            descriptions = tuple(
                self._normalized_event_text(text) for text, matches in words.items()
                if matches.max is not None and abs(matches.max.x - name.x) <= 35
                and 10 <= matches.max.center.y - name.center.y <= 40
            )
            if expected_description is None or self._normalized_event_text(expected_description) in descriptions:
                rows.append((option, label, descriptions))
        return rows

    def _choose_configured_list_row(
        self, event: HallowZeroEvent, words: dict[str, MatchResultList], frame: Rect,
    ) -> OperationRoundResult:
        """无成本离开优先，再依原配置挑选当前可见的完整条目。"""
        rows = self._configured_rows(event, words)
        focused = [
            label for _, label, _ in rows
            if (match := words[label].max) is not None
            and frame.x1 < match.center.x < frame.x2 and frame.y1 < match.center.y < frame.y2
        ]
        safe = [
            (option, label, details) for option, label, details in rows
            if not is_hp_loss_option(option) and not any(
                cost in text for cost in self.RESOURCE_COST_WORDS
                for text in (option.option_name, option.desc or '', *details)
            )
        ]
        leave = [
            (option, label, details) for option, label, details in safe
            if self.TIMED_TASK_EXIT in details or any(word in option.option_name for word in ('离开', '不需要', '拒绝'))
        ]
        choices = leave or safe
        if not choices:
            return self._pending('没有核实到原速刷配置中的事件选项')
        label = choices[0][1]
        target = words[label].max
        if target is None or not frame.x1 < target.center.x < frame.x2:
            return self._pending('等待目标选项与焦点同列')
        if len(focused) == 1:
            return self._choose_right_row(event.event_name, label, frame, words, focused[0])
        if not focused and leave and not frame.y1 <= target.center.y <= frame.y2:
            # 原选项说明可能随局变化。此时只移焦，随后仍须完整核对离开项才确认。
            row = f'焦点行{round(frame.center.y / 45)}'
            return self._choose_right_row(event.event_name, label, frame, words, row)
        return self._pending('等待唯一黄焦与已核对的事件条目')

    def _emergency_report(self, image: MatLike, words: dict[str, MatchResultList]) -> OperationRoundResult:
        """纯播报按正文和内联 A 逐页推进，遇到选项时留待确认。"""
        event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.EMERGENCY_REPORT)
        body = self._emergency_body(words)
        if (event is not None and body and not _yellow_rectangles(image, (440, 740), (56, 150))
                and not self._configured_rows(event, words)
                and self.round_by_find_area(
                    self.last_screenshot, self.SCREEN, '手柄-速刷继续提示',
                ).is_success):
            return self._tap('south', (self.EMERGENCY_REPORT, body))
        return self._pending('等待特殊区域播报正文与内联 A')

    def _special_zone_exit(self, image: MatLike, words: dict[str, MatchResultList]) -> OperationRoundResult:
        """只在离开特殊区域的两个明确选项中沿原速刷路径继续。"""
        event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.SPECIAL_ZONE)
        frame = self._focused_right_row(image, words, _yellow_rectangles(image, (440, 740), (56, 150)))
        if (event is None or frame is None or len(event.options) != 1
                or event.options[0].ocr_word != '确认继续' or not {'确认继续', '稍等一下'} <= words.keys()):
            return self._pending('等待离开特殊区域的两个选项与同排 A 提示')
        return self._choose_right_row(self.SPECIAL_ZONE, event.options[0].ocr_word, frame, words)

    def _academic_aid(self, image: MatLike, words: dict[str, MatchResultList]) -> OperationRoundResult:
        """安全点三段对话与一项赠礼分别核对相应的 A 提示。"""
        event = self.ctx.withered_domain.data_service.get_normal_event_by_name(self.ACADEMIC_AID)
        if event is None:
            return self._pending('缺少学会援助的原事件配置')
        frames = _yellow_rectangles(image, (440, 740), (56, 150))
        stage = self._academic_aid_stage(words)
        if stage is not None and not frames and self.round_by_find_area(
            self.last_screenshot, self.SCREEN, '手柄-速刷继续提示',
        ).is_success:
            return self._tap('south', (self.ACADEMIC_AID, stage))
        if ('接受学会的好意' in words and len(event.options) == 1
                and event.options[0].ocr_word == '接受学会的好意'
                and any('遗留的物资十分丰厚' in word for word in words)):
            frame = self._focused_right_row(image, words, frames)
            if frame is not None:
                return self._choose_right_row(self.ACADEMIC_AID, event.options[0].ocr_word, frame, words)
        return self._pending('等待学会援助的正文或赠礼焦点')

    def _lost_property_card(self) -> OperationRoundResult | None:
        """通用音频数据卡的独立预览页走 B 关闭。"""
        title = self._read('手柄-速刷音频卡标题')
        if title is None or '通用音频数据卡' not in title[1]:
            return None
        footer = self._read('手柄-速刷音频卡返回')
        if footer is not None and {'B', '返回'} <= footer[1].keys():
            return self._tap('east', ('通用音频数据卡', '返回'))
        return self._pending('等待旧都失物数据卡的 B 返回提示')

    def _focused_right_row(
        self, image: MatLike, words: dict[str, MatchResultList], frames: list[Rect],
    ) -> Rect | None:
        """唯一黄色整行和同排 A 提示共同确认当前焦点。"""
        if len(frames) != 1:
            return None
        frame = frames[0]
        prompt = words.get('A')
        if prompt is not None and prompt.max is not None and (
            frame.x1 - 77 < prompt.max.center.x < frame.x1
            and abs(prompt.max.center.y - frame.center.y) <= 26
        ):
            return frame
        region = image[frame.y1:frame.y2, max(0, frame.x1 - 77):frame.x1]
        if self.ctx.tm.match_template(region, 'ios_gamepad', 'dialog_a', threshold=0.7).max is not None:
            return frame
        return None

    def _choose_right_row(
        self, title: str, label: str, frame: Rect, words: dict[str, MatchResultList],
        focused: str | None = None,
    ) -> OperationRoundResult:
        item = words.get(label)
        if item is None or item.max is None:
            return self._pending('等待已知事件目标选项')
        target = item.max.rect
        key = 'south'
        if not frame.y1 < target.center.y < frame.y2:
            key = 'dpad_down' if target.center.y > frame.center.y else 'dpad_up'
        return self._tap(key, (title, label, focused or str(frame.y1)))

    def _support_event(self, image: MatLike, words: dict[str, MatchResultList]) -> OperationRoundResult:
        """读取局内队伍和回应代理人，用原规则选择接应与位置。"""
        frame = self._focused_right_row(image, words, _yellow_rectangles(image, (530, 680), (69, 150)))
        if frame is None:
            return self._pending('等待增援选项与同排 A 提示')
        if '选择所需的物资' in words:
            return self._choose_support_reward(frame, words)
        if any('代理人位置' in word for word in words):
            if self._support_agent is None or self._support_position is None:
                return self._pending('等待本次接应的代理人和目标位置')
            slots = tuple(
                f'{self._support_position}{suffix}' for suffix in ('号位', '号队员的位置')
            )
            options = [word for word in words if word in (slots[0], f'接替{slots[1]}')]
            if len(options) != 1:
                return self._pending('等待明确的增援目标位置')
            label = options[0]
            last_action = self._last_action
            result = self._choose_right_row(f'{self.SUPPORT}-选位', label, frame, words)
            if (self._last_action != last_action and words[label].max is not None
                    and frame.y1 < words[label].max.center.y < frame.y2):
                self._support_slot_sent = True
            return result

        accept = next((name for name in ('接应支援代理人', '接替小队成员') if name in words), None)
        if accept is None:
            return self._pending('等待明确的增援响应选项')
        source = self._read('呼叫增援-角色行')
        agents = [
            agent.value for agent in AgentEnum if source is not None and any(
                re.fullmatch(rf'[「『]?{re.escape(agent.value.agent_name)}[」』]?响应了求援信息[。！!]?', word)
                for word in source[1]
            )
        ]
        if len(agents) != 1:
            return self._pending('未核对到本次回应的代理人')
        domain = self.ctx.withered_domain
        team = domain.check_agent_list(self.last_screenshot)
        if team is None or len(team) != 3 or team[0] is None:
            return self._pending('未核对到当前队伍位置')
        target = CallForSupport(self.ctx)._should_call_backup(team, agents[0])
        if target is None:
            rejects = [word for word in words if (
                (event := domain.data_service.get_normal_event_by_name(word)) is not None
                and event.entry_name == self.SUPPORT
            )]
            if len(rejects) != 1:
                return self._pending('等待本次增援的拒绝选项')
            return self._choose_right_row(self.SUPPORT, rejects[0], frame, words)

        last_action = self._last_action
        result = self._choose_right_row(self.SUPPORT, accept, frame, words)
        if (self._last_action != last_action and words[accept].max is not None
                and frame.y1 < words[accept].max.center.y < frame.y2):
            self._support_agent, self._support_position = agents[0], target
        return result

    def _choose_support_reward(
        self, frame: Rect, words: dict[str, MatchResultList],
    ) -> OperationRoundResult:
        """增援离开后筛出完整的免费鸣徽配方，依配置的类别次序选择。"""
        domain = self.ctx.withered_domain
        offers: dict[str, tuple[str, float]] = {}
        for text, found in words.items():
            match = re.fullmatch(r'随机获得一枚S级或A级\[([^\]]+)\]类型鸣徽', text)
            if match is not None and found.max is not None and match[1] in domain.data_service.resonium_cate_list:
                offers[match[1]] = text, found.max.center.y
        if len(offers) != 3 or not any('支援物资' in word for word in words):
            return self._pending('等待三项明确的免费增援鸣徽')
        category = next(
            (kind for priority in domain.challenge_config.resonium_priority
             if (kind := priority.partition(' ')[0]) in offers),
            None,
        )
        label = min(offers.values(), key=lambda offer: offer[1])[0] if category is None else offers[category][0]
        return self._choose_right_row('增援物资', label, frame, words)

    def _single_yellow_confirm(self, button_area: str, prompt_area: str, label: str) -> bool:
        """已明确正文时核对黄色单确认与底栏 A「选择」。"""
        button = self._read(button_area)
        prompt = self._read(prompt_area)
        match = None if button is None else button[1].get(label)
        return (
            button is not None and match is not None and match.max is not None
            and has_filled_highlight(button[0], match.max.rect, 'yellow')
            and prompt is not None and {'A', '选择'} <= prompt[1].keys()
        )

    def _notice(self) -> OperationRoundResult | None:
        """接受支援或压力过载后，只关闭正文明确的单按钮提示。"""
        notice = self._read('手柄-速刷结果')
        if notice is None:
            return None
        words = notice[1]
        support = any('加入队伍' in word for word in words)
        corrosion = next((word for word in words if word.startswith('受到侵蚀')), '')
        if not support and not corrosion:
            return None
        agent, position = self._support_agent, self._support_position
        if support and (
            not self._support_slot_sent or agent is None or position is None
            or not any(re.fullmatch(rf'[「『]{re.escape(agent.agent_name)}[」』]加入队伍[。！!]?', word)
                       for word in words)
        ):
            return self._pending('增援结果与本次选位不一致')
        if corrosion:
            title = words[corrosion].max
            if (re.fullmatch(r'受到侵蚀[「『][\u4e00-\u9fff]{1,12}[」』]', corrosion) is None
                    or title is None or not any(
                        matches.max is not None and 25 <= matches.max.center.y - title.center.y <= 65
                        and re.search(r'[\u4e00-\u9fff]{4,}', word)
                        for word, matches in words.items()
                    )):
                return self._pending('等待完整的侵蚀症状和说明')
        if not self._single_yellow_confirm('手柄-速刷结果按钮', '手柄-速刷结果提示', '确认'):
            return self._pending('等待零号空洞结果的单确认按钮')
        if support and not self._support_resolved and agent is not None and position is not None:
            self.ctx.withered_domain.update_agent_list_after_support(agent, position)
            self._support_resolved = True
        return self._tap('south', ('侵蚀提示', corrosion) if corrosion else ('增援结果', *sorted(words)))

    def _plugin_reward(self) -> OperationRoundResult | None:
        """关闭自动获得的邦布插件提示，保留原插件及数量。"""
        area = self._read('手柄-速刷插件奖励')
        if area is None or '获得邦布插件' not in area[1]:
            return None
        if (len(area[1]) < 2 or not self._single_yellow_confirm(
            '手柄-速刷插件奖励按钮', '手柄-速刷插件奖励提示', '确定',
        )):
            return self._pending('等待邦布插件奖励的确定提示')
        return self._tap('south', ('获得邦布插件', *sorted(area[1])))

    def _choose_right_option(
        self, title: str, image: MatLike, words: dict[str, MatchResultList], frames: list[Rect],
    ) -> OperationRoundResult:
        """可见离开选项优先，按方向键移动后复核选中行和同排 A。"""
        event = self.ctx.withered_domain.data_service.get_normal_event_by_name(title)
        frame = self._focused_right_row(image, words, frames)
        if event is None or frame is None:
            return self._pending('右侧事件或同排 A 焦点未确认')
        if title != '投机客':
            return self._choose_configured_list_row(event, words, frame)
        options = [opt for opt in event.options if opt.ocr_word in words]
        leave = [opt for opt in options if any(word in opt.option_name for word in ('离开', '不需要', '拒绝'))]
        safe = leave or self._free_speculator_aid(options, words)
        if not safe:
            return self._pending('没有核实到速刷允许的事件选项')
        return self._choose_right_row(title, safe[0].ocr_word, frame, words)

    def _free_speculator_aid(
        self, options: list[HallowZeroNormalEventOption], words: dict[str, MatchResultList],
    ) -> list[HallowZeroNormalEventOption]:
        """仅接受两个效果都可核对的援助分支，按游戏原事件配置顺序选择。"""
        expected = {'清除压力', '恢复身体'}
        if '请选择一项援助' not in words or {opt.option_name for opt in options} != expected:
            return []
        for opt in options:
            name = words[opt.ocr_word].max
            description = None if opt.desc is None else words.get(opt.desc)
            effect = None if description is None else description.max
            if (name is None or effect is None or not 10 <= effect.center.y - name.center.y <= 40
                    or abs(effect.x - name.x) > 26):
                return []
        return options

    @operation_node(name='识别零号空洞交互', is_start_node=True, node_max_retry_times=24)
    def respond(self) -> OperationRoundResult:
        result = self.try_respond()
        if result is not None:
            return result
        if self._handled:
            return self.round_success('零号空洞交互已关闭')
        return self._pending('未确认可处理的零号空洞交互')

    def try_respond(self) -> OperationRoundResult | None:
        """外层走格可在当前帧优先处理确切事件。"""
        result = self._resonium()
        if result is not None:
            return result
        result = self._right_event()
        if result is not None:
            return result
        result = self._notice() or self._plugin_reward() or self._lost_property_card()
        if result is not None:
            return result
        if self._support_slot_sent and not self._support_resolved:
            return self._pending('等待本次增援加入队伍的回执')
        return None


def _yellow_rectangles(image: MatLike, width: tuple[int, int], height: tuple[int, int]) -> list[Rect]:
    """从观测到的尺寸范围提取黄色卡片边框或当前选项填充。"""
    contours, _ = cv2.findContours(highlight_mask(image, 'yellow'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[Rect] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if width[0] <= w <= width[1] and height[0] <= h <= height[1]:
            result.append(Rect(x, y, x + w, y + h))
    return result
