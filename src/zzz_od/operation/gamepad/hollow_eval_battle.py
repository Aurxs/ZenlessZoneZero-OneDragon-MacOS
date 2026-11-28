"""复用零号空洞原战斗策略，以手柄 HUD 限制战斗输入。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.controller.macos_battle_input_guard import MacBattleHudGuard
from zzz_od.hollow_zero.hollow_battle import HollowBattle
from zzz_od.operation.gamepad.battle_assistant import is_native_battle
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.lost_void_result import GamepadLostVoidBattleResult
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_base import OperationResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadHollowEvalBattle(GamepadOperation):
    """从已确认的实战 HUD 跑原自动战斗，结算交回局内流程。"""

    realtime_nodes: ClassVar[frozenset[str]] = frozenset({'执行零号空洞原战斗'})
    HALT_RESULT: ClassVar[tuple[tuple[str, str], ...]] = (
        ('零号空洞-战斗', '挑战结果'),
        ('零号空洞-战斗', '结算周期上限-确认'),
        ('战斗画面', '手柄-挑战结果'),
    )

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄零号空洞战斗', timeout_seconds=630)
        self.original: HollowBattle = HollowBattle(ctx)
        self.navigation: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', '周期征讨')
        self._scope: bool = False
        self._active: bool = False
        self._approached: bool = False
        self._ending: bool = False
        self._lost_hud_at: float | None = None

    def handle_init(self) -> None:
        self.original.handle_init()
        self._active = False
        self._approached = False
        self._ending = False
        self._lost_hud_at = None

    @operation_node(name='确认零号空洞手柄战斗', is_start_node=True, node_max_retry_times=10)
    def prepare(self) -> OperationRoundResult:
        if not is_native_battle(self.ctx, self.last_screenshot):
            return self.round_retry('等待原生手柄战斗 HUD', wait=0.5)
        self.original.load_auto_op()
        battle = self.ctx.auto_battle_context
        if battle.auto_op is None:
            return self.round_fail('原速刷自动战斗策略未加载')
        self.ctx.controller.begin_battle_input_scope(MacBattleHudGuard(battle))
        self._scope = True
        return self.round_success()

    @node_from(from_name='确认零号空洞手柄战斗')
    @operation_node(name='执行零号空洞原战斗', mute=True, timeout_seconds=600)
    def fight(self) -> OperationRoundResult:
        battle = self.ctx.auto_battle_context
        if self._ending or self.last_screenshot is None or not MacBattleHudGuard(battle)(self.last_screenshot):
            if self._active:
                battle.stop_auto_battle()
                self._active = False
            if self._in_return_scene():
                self.save_screenshot(prefix='macos_hollow_eval_combat_return')
                return self.round_success('已识别零号空洞战斗后画面')
            now = time.monotonic()
            if self._lost_hud_at is None:
                self._lost_hud_at = now
            if now - self._lost_hud_at >= 15:
                self.save_screenshot(prefix='macos_hollow_eval_combat_unconfirmed')
                return self.round_fail('战斗 HUD 消失，后续画面未确认')
            return self.round_wait(wait=0.25)
        if self.ctx.controller.battle_input_blocked and not self.ctx.controller.refresh_battle_input_scope():
            return self.round_wait(wait=0.25)
        self._lost_hud_at = None
        original = self.original
        original.last_screenshot = self.last_screenshot
        original.last_screenshot_time = self.last_screenshot_time
        if self._active:
            result = original.auto_battle()
            if not result.is_success:
                return result
            self._active = False
            if result.status != HollowBattle.STATUS_NEED_SPECIAL_MOVE:
                self._ending = True
                return self.round_wait('等待实际战斗返回', wait=0.25)
        if not self._approached:
            result = original.check_special_move()
            if not result.is_success:
                return result
            self._approached = True
            if result.status == HollowBattle.STATUS_NO_NEED_SPECIAL_MOVE:
                self._active = True
                return self.round_wait()
            approach = original.special_move()
            return self.round_wait() if approach.is_success else approach
        result = original.move_to_battle()
        if result.is_success:
            self._active = True
            return self.round_wait()
        return result

    def _in_return_scene(self) -> bool:
        """只用原结算标志或手柄棋盘锚点判断画面已移交。"""
        if self.last_screenshot is None:
            return False
        if any(self.round_by_find_area(self.last_screenshot, screen, area).is_success
               for screen, area in self.HALT_RESULT):
            return True
        nav = self.navigation
        nav.last_screenshot = self.last_screenshot
        bag = nav._read_area('零号空洞-事件', '手柄-走格背包')
        floor = nav._read_area('零号空洞-事件', '手柄-走格层数')
        return bool(bag and '背包' in bag[1] and floor and nav._text_rect(floor[1], '当前层数'))

    def handle_pause(self) -> None:
        self.ctx.auto_battle_context.stop_auto_battle()
        self._active = False

    def execute(self) -> OperationResult:
        try:
            return super().execute()
        finally:
            try:
                self.ctx.auto_battle_context.stop_auto_battle()
            finally:
                if self._scope:
                    self.ctx.controller.end_battle_input_scope()
                    self._scope = False


class GamepadHollowEvalBattleResult(GamepadLostVoidBattleResult):
    """两种空洞玩法共用道中结果画面，仅确认通关一关和 A。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, mid_result_only=True)
        self.op_name = '手柄确认零号空洞道中战斗'
