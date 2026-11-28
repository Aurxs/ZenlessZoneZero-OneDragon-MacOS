"""GUI 与 CLI 共用的 Mac 运行请求、预览和结果；不连接游戏。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from one_dragon.base.operation.application.application_run_context import (
        ApplicationRunResult,
    )
    from zzz_od.context.zzz_context import ZContext


@dataclass(frozen=True)
class MacRunOptions:
    """只对本次单项任务生效，不写入账户或任务配置。"""

    redeem_code: str | None = None
    claim_all_mail: bool = False
    resume_lost_void: bool = False
    resume_store_refreshed: bool = False
    resume_signin_reward: bool = False
    tinman_only: bool = False

    def validate(self, app_id: str, game_region: str, require_explicit: bool = True) -> None:
        """限制选项所属任务，邮件和兑换码要求明确指定。"""
        from zzz_od.macos.redemption_catalog import normalize_code

        for selected, target, label in (
            (self.redeem_code is not None, 'redemption_code', '兑换码'),
            (self.claim_all_mail, 'email', '全部领取邮件'),
            (self.resume_lost_void, 'lost_void', '续接迷失之地'),
            (self.resume_store_refreshed, 'lost_void', '商店已刷新一次'),
            (self.resume_signin_reward, 'daily_signin', '续接占卜奖励'),
            (self.tinman_only, 'coffee', '仅汀曼特调'),
        ):
            if selected and app_id != target:
                raise ValueError(f'「{label}」仅适用于单项任务 {target}')
        if self.resume_store_refreshed and not self.resume_lost_void:
            raise ValueError('商店已刷新一次必须同时选择续接迷失之地')
        if self.redeem_code is not None:
            normalize_code(self.redeem_code)
            if game_region != 'cn':
                raise ValueError('Mac 兑换码仅支持已配置的国服账户')
        if require_explicit and app_id == 'email' and not self.claim_all_mail:
            raise ValueError('请明确勾选「全部领取邮件附件」后运行单项邮件任务')
        if require_explicit and app_id == 'redemption_code' and self.redeem_code is None:
            raise ValueError('请明确输入一枚国服兑换码后运行单项兑换任务')

    def describe(self) -> list[str]:
        """显示本次模式；不在预览或日志中回显兑换码。"""
        return [label for selected, label in (
            (self.redeem_code is not None, '手动兑换一枚国服兑换码'),
            (self.claim_all_mail, '全部领取邮件附件'),
            (self.resume_lost_void, '续接已确认的战线肃清挑战'),
            (self.resume_store_refreshed, '当前商店已刷新一次，不再刷新'),
            (self.resume_signin_reward, '续接已确认的当日占卜奖励或结果页'),
            (self.tinman_only, '本次仅选择汀曼特调'),
        ) if selected]

    @contextmanager
    def scoped(self, ctx: ZContext, app_id: str) -> Iterator[None]:
        """覆盖本次全部开关，异常和停止后恢复调用前状态。"""
        from zzz_od.macos.redemption_catalog import normalize_code

        self.validate(app_id, ctx.game_account_config.game_region)
        values = {
            'macos_redeem_code': normalize_code(self.redeem_code) if self.redeem_code else None,
            'macos_claim_all_mail': self.claim_all_mail,
            'macos_resume_lost_void': self.resume_lost_void,
            'macos_resume_store_refreshed': self.resume_store_refreshed,
            'macos_resume_signin_reward': self.resume_signin_reward,
            'macos_tinman_only': self.tinman_only,
        }
        missing = object()
        previous = {key: getattr(ctx, key, missing) for key in values}
        try:
            for key, value in values.items():
                setattr(ctx, key, value)
            yield
        finally:
            for key, value in previous.items():
                if value is missing:
                    delattr(ctx, key)
                else:
                    setattr(ctx, key, value)


def validate_request(ctx: ZContext, app_id: str) -> None:
    """在连接采集和手柄前拒绝未知、内部或尚未适配的任务。"""
    from one_dragon.base.config.game_account_config import GamePlatformEnum
    from zzz_od.macos.application_capabilities import (
        get_macos_one_dragon_unavailable_reason,
    )

    if ctx.game_account_config.platform != GamePlatformEnum.MACOS_PLAYCOVER.value.value:
        raise ValueError('请先在 GUI 账户管理中将当前账户设为 macOS PlayCover')
    if not ctx.run_context.is_app_registered(app_id) or (
        app_id != 'one_dragon' and not ctx.run_context.is_app_standalone_available(app_id)
    ):
        raise ValueError(f'未知或不可独立调用的任务：{app_id}')
    reason = ctx.get_application_unavailable_reason(app_id)
    if reason is None and app_id == 'one_dragon':
        reason = get_macos_one_dragon_unavailable_reason(ctx)
    if reason is not None:
        raise ValueError(reason)


def describe_request(ctx: ZContext, app_id: str) -> dict[str, object]:
    """预览已保存配置，不初始化模型、采集或输入，也不重置运行记录。"""
    from zzz_od.macos.application_capabilities import (
        get_macos_app_limits,
        get_macos_one_dragon_unavailable_reason,
    )

    app_ids = [app_id]
    if app_id == 'one_dragon':
        group = ctx.app_group_manager.get_one_dragon_group_config(ctx.current_instance_idx)
        app_ids = [item.app_id for item in group.app_list if item.enabled]
    return {
        'instance_idx': ctx.current_instance_idx,
        'app_id': app_id,
        'unavailable_reason': (
            get_macos_one_dragon_unavailable_reason(ctx) if app_id == 'one_dragon'
            else ctx.get_application_unavailable_reason(app_id)
        ),
        'tasks': [{
            'app_id': item,
            'name': ctx.run_context.get_application_name(item),
            'limits': get_macos_app_limits(item),
            'unavailable_reason': ctx.get_application_unavailable_reason(item),
        } for item in app_ids],
    }


def summarize_result(ctx: ZContext, result: ApplicationRunResult) -> tuple[dict[str, object], int]:
    """应用组完成后核对启用任务记录，不能把子任务失败当成命令成功。"""
    from one_dragon.base.operation.application.application_run_context import (
        RunFinishReason,
    )
    from one_dragon.base.operation.application_run_record import AppRunRecord

    failed_apps: list[str] = []
    tasks: list[dict[str, object]] = []
    gaps: list[str] = []
    app_ids = [result.app_id]
    if result.app_id == 'one_dragon':
        group = ctx.app_group_manager.get_one_dragon_group_config(ctx.current_instance_idx)
        app_ids = [item.app_id for item in group.app_list if item.enabled]
        for item in group.app_list:
            if not item.enabled:
                continue
            record = ctx.run_context.get_run_record(item.app_id, ctx.current_instance_idx)
            state = getattr(record, 'run_status_under_now', getattr(record, 'run_status', None))
            if state == AppRunRecord.STATUS_FAIL:
                failed_apps.append(item.app_id)
            state_text = {
                AppRunRecord.STATUS_SUCCESS: '记录完成',
                AppRunRecord.STATUS_FAIL: '失败',
                AppRunRecord.STATUS_WAIT: '未完成',
                AppRunRecord.STATUS_RUNNING: '运行中断',
            }.get(state, '结果未确认')
            tasks.append({
                'app_id': item.app_id, 'outcome': state_text,
                'name': ctx.run_context.get_application_name(item.app_id),
            })
            if state != AppRunRecord.STATUS_SUCCESS:
                gaps.append(f'{ctx.run_context.get_application_name(item.app_id) or item.app_id}：{state_text}')
    operation_result = ctx.run_context.last_application_result
    if result.app_id != 'one_dragon' and operation_result is None:
        gaps.append('任务结果未确认')
    charge_plans: list[dict[str, object]] = []
    if 'charge_plan' in app_ids:
        charge_plans, charge_gaps = describe_charge_plan_gaps(ctx, result)
        gaps.extend(charge_gaps)
    # 原 CLI 的正常不足或跳过仍以 0 退出，完成程度单独报告。
    success = result.finish_reason == RunFinishReason.COMPLETED and not failed_apps
    return {
        'success': success,
        'all_tasks_finished': success and not gaps,
        'app_id': result.app_id,
        'instance_idx': result.instance_idx,
        'finish_reason': str(result.finish_reason),
        'status': operation_result.status if operation_result is not None else None,
        'failed_apps': failed_apps,
        'tasks': tasks,
        'gaps': gaps,
        'charge_plans': charge_plans,
    }, 0 if success else 1


def describe_charge_plan_gaps(
    ctx: ZContext, result: ApplicationRunResult,
) -> tuple[list[dict[str, object]], list[str]]:
    """只核对已有计划的真实次数；跳过和电量不足不能充作全部完成。"""
    from one_dragon.base.operation.application import application_const

    plans: list[dict[str, object]] = []
    gaps: list[str] = []
    getter = getattr(ctx.run_context, 'get_config', None)
    if callable(getter):
        try:
            config = getter(
                app_id='charge_plan', instance_idx=result.instance_idx,
                group_id=result.group_id or application_const.DEFAULT_GROUP_ID,
            )
            for index, plan in enumerate(config.plan_list, start=1):
                name = plan.mission_name or plan.mission_type_name or plan.category_name
                remaining = max(0, plan.plan_times - plan.run_times)
                plans.append({
                    'name': name, 'run_times': plan.run_times, 'plan_times': plan.plan_times,
                    'remaining': remaining, 'skipped': plan.skipped,
                })
                if remaining:
                    suffix = ''
                    if plan.skipped:
                        cause = getattr(plan, 'skip_reason', '')
                        suffix = f'（本轮已跳过，{cause}）' if cause else '（本轮已跳过）'
                    gaps.append(f'体力计划第{index}项 {name}：{plan.run_times}/{plan.plan_times}，剩余{remaining}次{suffix}')
        except Exception:
            gaps.append('体力计划进度未确认，请查看日志和原计划')
    elif result.app_id == 'charge_plan':
        data = getattr(ctx.run_context.last_application_result, 'data', None)
        if isinstance(data, dict) and data.get('plans_finished') is False:
            gaps.append('体力计划仍有未完成次数（电量不足或本轮已跳过）')
    return plans, gaps


