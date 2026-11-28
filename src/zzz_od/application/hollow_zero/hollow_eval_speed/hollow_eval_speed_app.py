from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar, cast

from one_dragon.base.operation.application_run_record import AppRunRecord
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.application.hollow_zero.hollow_eval_speed import hollow_eval_speed_const
from zzz_od.application.hollow_zero.hollow_eval_speed.hollow_eval_speed_config import (
    HollowEvalSpeedConfig,
)
from zzz_od.application.hollow_zero.withered_domain.withered_domain_app import (
    WitheredDomainApp,
)
from zzz_od.hollow_zero.hollow_quick_runner import HollowQuickRunner

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_base import OperationResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext
    from zzz_od.hollow_zero.hollow_runner import HollowRunner
    from zzz_od.hollow_zero.hollow_zero_challenge_config import (
        HollowZeroChallengeConfig,
    )


class HollowEvalSpeedApp(WitheredDomainApp):

    """零号业绩速刷:固定进入枯萎之都-施工废墟-内部,只速刷零号业绩。注意:用这个方法刷业绩会使周期丁尼的总量减少。"""

    SPEED_RUN: ClassVar[bool] = True

    def __init__(self, ctx: ZContext) -> None:
        WitheredDomainApp.__init__(
            self,
            ctx=ctx,
            app_id=hollow_eval_speed_const.APP_ID,
            op_name=hollow_eval_speed_const.APP_NAME,
        )

    def _get_challenge_config(self) -> HollowZeroChallengeConfig | None:
        return cast(HollowEvalSpeedConfig, self.config).build_challenge_config()

    def _create_runner(self) -> HollowRunner:
        return HollowQuickRunner(self.ctx)

    @operation_node(name='初始画面识别', is_start_node=True)
    def check_first_screen(self) -> OperationRoundResult:
        """Mac 使用手柄固定路线，其他平台沿用原空洞应用。"""
        if sys.platform != 'darwin':
            return super().check_first_screen()
        if self.run_record.run_status_under_now == AppRunRecord.STATUS_SUCCESS:
            status = '本周业绩已刷满' if self.run_record.is_finished_by_week() else '达到今日进入次数上限'
            return self.round_success(status)

        from zzz_od.operation.gamepad.hollow_eval_exit import GamepadHollowEvalFinish
        from zzz_od.operation.gamepad.hollow_eval_run import GamepadHollowEvalRun

        while self.ctx.run_context.is_context_running:
            result = GamepadHollowEvalRun(self.ctx).execute()
            if not result.success:
                return self.round_by_op_result(result)
            if result.status != GamepadHollowEvalFinish.STATUS_RETURNED or not isinstance(result.data, dict):
                return self.round_fail('本次结束结算或返回街区未确认')
            record = self.run_record
            receipt = result.data
            record.check_and_update_status()
            record.update('daily_run_times', record.daily_run_times + 1, save=False)
            if receipt.get('weekly_cap_confirmed') is True and receipt.get('weekly_cap_screenshot'):
                record.update('no_eval_point', True, save=False)
                record.update_status(AppRunRecord.STATUS_SUCCESS)
                return self.round_success('已核实本周零号业绩已满', data=receipt)
            increase = receipt.get('merit_increase')
            if type(increase) is not int or increase <= 0:
                record.update_status(AppRunRecord.STATUS_FAIL)
                return self.round_fail('本局已结束，零号业绩增量未获核实', data=receipt)
            if record.is_finished_by_day():
                record.update_status(AppRunRecord.STATUS_SUCCESS)
                return self.round_success('本局业绩已核实，达到今日进入次数上限', data=receipt)
            record.update_status(AppRunRecord.STATUS_RUNNING)
        return self.round_fail('零号业绩速刷已停止')

    def _update_record_after_stop(self, result: OperationResult) -> None:
        """Mac 只在实际退出回执到达时更新计数和运行状态。"""
        if sys.platform != 'darwin':
            super()._update_record_after_stop(result)
