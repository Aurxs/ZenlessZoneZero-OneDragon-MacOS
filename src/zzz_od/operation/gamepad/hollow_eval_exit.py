"""手柄保留当前施工废墟进度并返回街区。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

import cv2

from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.focus import has_filled_highlight
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.hollow_eval_receipt import confirms_weekly_merit_cap
from zzz_od.operation.gamepad.hollow_eval_speed import GamepadHollowEvalPrepare
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadHollowEvalSuspend(GamepadOperation):
    """在施工废墟使用 X 暂离，核对暂停提示并保留本局。"""

    SCREEN: ClassVar[str] = '零号空洞-事件'
    ACTION: ClassVar[str] = '暂离'
    PROMPT: ClassVar[str] = '是否确认暂停空洞探索，返回街区'
    KEY: ClassVar[str] = 'west'

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄暂离施工废墟', timeout_seconds=30)
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', '周期征讨')
        self._menu_requested: bool = False
        self._leave_requested: bool = False
        self._confirmed: bool = False

    def handle_init(self) -> None:
        self._menu_requested = False
        self._leave_requested = False
        self._confirmed = False

    def _matches_action(self, image: MatLike, words: dict[str, MatchResultList]) -> bool:
        return any(text.upper() == 'X' for text in words)

    def _can_confirm(self) -> bool:
        return True

    def _returned(self) -> OperationRoundResult:
        path = self.save_screenshot(prefix='macos_hollow_eval_suspended')
        return self.round_success('本局已暂离并返回街区', data={'screenshot': path})

    @operation_node(name='确认暂离并返回街区', is_start_node=True, node_max_retry_times=15)
    def leave(self) -> OperationRoundResult:
        if self.last_screenshot is None:
            return self.round_retry('等待当前画面', wait=0.8)
        if self._confirmed and self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            return self._returned()
        nav = self.navigation
        nav.last_screenshot = self.last_screenshot
        modal = nav._read_area(self.SCREEN, '手柄-速刷退出弹窗')
        if modal is not None and nav._text_rect(modal[1], self.PROMPT) is not None:
            if not self._leave_requested:
                return self.round_fail(f'当前已有{self.ACTION}确认，未核实本次请求')
            confirm_word = modal[1].get('确认')
            confirm = confirm_word.max.rect if confirm_word is not None and confirm_word.max is not None else None
            if (confirm is not None and has_filled_highlight(modal[0], confirm, 'yellow')
                    and not self._confirmed and self._can_confirm()):
                self.ctx.controller.btn_tap('south')
                self._confirmed = True
            return self.round_retry(f'等待{self.ACTION}确认和街区画面', wait=0.8)
        if self._leave_requested:
            return self.round_retry(f'等待{self.ACTION}弹窗或返回加载', wait=0.8)

        tab = nav._read_area(self.SCREEN, '手柄-速刷战况页签')
        mission = nav._read_area(self.SCREEN, '手柄-速刷战况副本')
        footer = nav._read_area(self.SCREEN, '手柄-速刷战况底栏')
        if (tab is not None and '战况' in tab[1] and mission is not None
                and nav._text_rect(mission[1], '施工废墟·内部') is not None and footer is not None
                and all(text in footer[1] for text in ('暂离', '放弃', '返回'))
                and self._matches_action(*footer)):
            self.ctx.controller.btn_tap(self.KEY)
            self._leave_requested = True
            return self.round_retry(f'等待{self.ACTION}确认弹窗', wait=0.8)

        floor = nav._read_area(self.SCREEN, '手柄-走格层数')
        bag = nav._read_area(self.SCREEN, '手柄-走格背包')
        if (not self._menu_requested and floor is not None and any('当前层数' in text for text in floor[1])
                and bag is not None and '背包' in bag[1]):
            self.ctx.controller.btn_tap('start')
            self._menu_requested = True
        return self.round_retry('等待施工废墟战况菜单', wait=0.8)


class GamepadHollowEvalFinish(GamepadHollowEvalSuspend):
    """在已核实的停止点结束本局，街区回执只证明退出完成。"""

    ACTION: ClassVar[str] = '放弃'
    PROMPT: ClassVar[str] = '是否确认放弃空洞探索，返回街区'
    KEY: ClassVar[str] = 'north'
    STATUS_RETURNED: ClassVar[str] = '施工废墟本局已经结束并返回街区'

    def __init__(self, ctx: ZContext, stop_evidence: dict[str, object], resume_settlement: bool = False) -> None:
        super().__init__(ctx)
        self.op_name = '手柄结束施工废墟本局'
        self.timeout_seconds = 45
        self.stop_evidence: dict[str, object] = stop_evidence
        self.resume_settlement: bool = resume_settlement
        self._settlement: str | None = None
        self._settlement_merit: int | None = None
        self._settlement_cap: str | None = None
        self._entrance: GamepadHollowEvalPrepare = GamepadHollowEvalPrepare(ctx)
        self._returned_pages: set[str] = set()

    def handle_init(self) -> None:
        super().handle_init()
        self._settlement = None
        self._settlement_merit = None
        self._settlement_cap = None
        self._returned_pages.clear()

    def _matches_action(self, image: MatLike, words: dict[str, MatchResultList]) -> bool:
        button = words['放弃'].max
        if button is None:
            return False
        hint = image[max(0, button.y - 8):button.y + button.h + 8,
                     max(0, button.x - 52):max(0, button.x - 1)]
        return bool(hint.size and self.ctx.ocr.run_ocr_single_line(cv2.resize(hint, None, fx=4, fy=4)).upper() == 'Y')

    def _can_confirm(self) -> bool:
        return self.round_by_find_area(
            self.last_screenshot, '零号空洞-入口', '手柄-速刷续接按键',
        ).is_success

    def _returned(self) -> OperationRoundResult:
        if self._settlement is None:
            return self.round_fail('已看到街区，但本次正式结算未核实')
        path = self.save_screenshot(prefix='macos_hollow_eval_finished')
        cap = self._settlement_cap or self.stop_evidence.get('weekly_cap_screenshot')
        receipts = self.stop_evidence.get('merit_receipts')
        verified: dict[str, int] = {}
        if isinstance(receipts, list):
            for proof in receipts:
                if not isinstance(proof, dict):
                    continue
                amount, image = proof.get('amount'), proof.get('screenshot')
                if type(amount) is int and amount > 0 and isinstance(image, str) and image:
                    verified.setdefault(image, amount)
        return self.round_success(self.STATUS_RETURNED, data={
            'stop_screenshot': self.stop_evidence['screenshot'],
            'settlement_screenshot': self._settlement,
            'screenshot': path,
            'merit_increase': sum(verified.values()) if verified else self._settlement_merit,
            'merit_receipts': [{'amount': amount, 'screenshot': image} for image, amount in verified.items()],
            'weekly_cap_confirmed': isinstance(cap, str) and bool(cap),
            'weekly_cap_screenshot': cap,
        })

    def _read_settlement_merit(self) -> tuple[bool, int | None]:
        """右侧同一奖励槽同时命中原业绩图标和正数数量。"""
        area = self.navigation._read_area('零号空洞-入口', '手柄-速刷结算奖励')
        if area is None:
            return False, None
        image, words = area
        if not any(text == 'EMPTY' or re.fullmatch(r'[0-9]+', text) for text in words):
            return False, None
        icon = self.ctx.tm.match_template(image, 'hollow', 'challenge_reward_eval', threshold=0.72).max
        if icon is None:
            return True, None
        if icon.confidence < 0.80:
            return False, None
        amounts = {
            int(text) for text, matches in words.items() if re.fullmatch(r'[1-9][0-9]{0,3}', text)
            and any(icon.x < word.center.x < icon.x + icon.w
                    and 1 <= word.center.y - icon.y - icon.h <= 45 for word in matches)
        }
        return (True, next(iter(amounts))) if len(amounts) == 1 else (False, None)

    def _finish_settlement(self) -> OperationRoundResult | None:
        nav = self.navigation
        nav.last_screenshot = self.last_screenshot
        screen = '零号空洞-入口'
        title = nav._read_area(screen, '手柄-速刷结算标题')
        heading = ''.join(''.join(word.split()) for word in title[1]) if title is not None else ''
        if heading != '施工废墟·内部MISSIONFAILED':
            return None
        progress = nav._read_area(screen, '手柄-速刷结算进度')
        footer = nav._read_area(screen, '手柄-速刷完成按键')
        if self._settlement is None and progress is not None and footer is not None and (
            all(text in progress[1] for text in ('执照等级', '空洞层数'))
            and ('B完成' in footer[1] or ('B' in footer[1] and '完成' in footer[1]))
        ):
            readable, self._settlement_merit = self._read_settlement_merit()
            if not readable:
                path = self.save_screenshot(prefix='macos_hollow_eval_settlement_unconfirmed')
                return self.round_fail('终止结算的零号业绩回执尚未核实，已保留当前画面', data={'screenshot': path})
            self._settlement = self.save_screenshot(prefix='macos_hollow_eval_settlement')
            if not self._settlement:
                return self.round_fail('未能留存结算回执，保持当前结算')
            if confirms_weekly_merit_cap(self.ctx.ocr.run_ocr(self.last_screenshot)):
                self._settlement_cap = self._settlement
            self.ctx.controller.btn_tap('east')
        return self.round_retry('等待结束结算并返回街区', wait=0.8)

    def _leave_entrances(self) -> OperationRoundResult | None:
        """正式结算后只从已确认的区域选择和公共入口各返回一次。"""
        entrance = self._entrance
        entrance.navigation.last_screenshot = entrance.last_screenshot = self.last_screenshot
        entrance.transport.last_screenshot = self.last_screenshot
        if (entrance._has_labels('手柄-枯萎区域列表', '资质考核', '旧都列车', '施工废墟')
                and entrance._has_labels('手柄-枯萎区域底栏', '奖励进度', '探索收集')):
            page = '枯萎区域选择'
        elif entrance.transport._at_hollow_hub():
            page = '零号空洞公共入口'
        else:
            return None
        if page not in self._returned_pages:
            self.ctx.controller.btn_tap('east')
            self._returned_pages.add(page)
        return self.round_retry(f'等待从{page}返回', wait=0.8)

    @operation_node(name='确认暂离并返回街区', is_start_node=True, node_max_retry_times=24)
    def leave(self) -> OperationRoundResult:
        floor = self.stop_evidence.get('floor')
        empty_tile = self.stop_evidence.get('empty_merit_tile')
        if (type(floor) is not int or not 1 <= floor <= 3 or type(empty_tile) is not bool
                or (floor != 3 and not empty_tile) or not self.stop_evidence.get('screenshot')):
            return self.round_fail('缺少本次已核实的业绩停止结果，保持当前挑战')
        if self.resume_settlement and not self._confirmed:
            settlement = self._finish_settlement()
            if self._settlement:
                self._confirmed = True
            return settlement if settlement is not None else self.round_retry('等待已终止施工废墟的正式结算', wait=0.8)
        if self._confirmed:
            settlement = self._finish_settlement()
            if settlement is not None:
                return settlement
            if self._settlement is not None:
                exit_entrance = self._leave_entrances()
                if exit_entrance is not None:
                    return exit_entrance
        return super().leave()
