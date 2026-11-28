"""串联原零号业绩速刷应用的 Mac 手柄局内流程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.hollow_eval_board import (
    GamepadHollowEvalExplore,
    GamepadHollowEvalStep,
)
from zzz_od.operation.gamepad.hollow_eval_exit import GamepadHollowEvalFinish
from zzz_od.operation.gamepad.hollow_eval_speed import GamepadHollowEvalDeploy

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadHollowEvalRun(GamepadOperation):
    """以原速刷配置执行固定路线，记录以实际最终回执为准。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄零号业绩速刷', timeout_seconds=1200)

    @operation_node(name='进入施工废墟内部', is_start_node=True)
    def enter(self) -> OperationRoundResult:
        return self.round_by_op_result(GamepadHollowEvalDeploy(self.ctx).execute())

    @node_from(from_name='进入施工废墟内部')
    @operation_node(name='连续探索到业绩退出条件')
    def explore(self) -> OperationRoundResult:
        result = GamepadHollowEvalExplore(self.ctx, continue_batches=True).execute()
        if not result.success:
            return self.round_by_op_result(result)
        if result.status != GamepadHollowEvalStep.STATUS_READY_TO_LEAVE or not isinstance(result.data, dict):
            return self.round_fail('本局业绩停止条件尚未核实', data=result.data)
        return self.round_by_op_result(GamepadHollowEvalFinish(self.ctx, result.data).execute())
