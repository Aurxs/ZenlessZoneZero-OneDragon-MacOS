"""原运行页共用的 Mac 单次选项、只读预览和任务结果。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    LineEdit,
    MessageBoxBase,
    PlainTextEdit,
    PushButton,
)

from one_dragon.base.operation.application.application_run_context import (
    ApplicationRunResult,
    RunFinishReason,
)
from one_dragon.utils.log_utils import log
from one_dragon_qt.view.app_run_interface import AppRunner
from zzz_od.macos.run_request import (
    MacRunOptions,
    describe_request,
    summarize_result,
    validate_request,
)

if TYPE_CHECKING:
    from zzz_od.context.zzz_context import ZContext


MACOS_OPTION_APP_IDS: frozenset[str] = frozenset({
    'email', 'redemption_code', 'coffee', 'lost_void', 'daily_signin',
})


class MacAppRunner(AppRunner):
    """原应用线程只在本次运行期间传递选项，始终清理总线与临时参数。"""

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx)
        self.options: MacRunOptions = MacRunOptions()
        self.summary: dict[str, object] | None = None

    def run(self) -> None:
        self.run_result = None
        self.summary = None
        dispatched = False
        try:
            validate_request(self.ctx, self.app_id)
            with self.options.scoped(self.ctx, self.app_id):
                dispatched = True
                super().run()
                if self.run_result is not None:
                    self.summary, _ = summarize_result(self.ctx, self.run_result)
        except Exception as error:
            log.error('Mac 任务运行失败：%s', error, exc_info=True)
            self.run_result = ApplicationRunResult(
                RunFinishReason.FAILED if dispatched else RunFinishReason.NOT_STARTED,
                self.app_id, self.ctx.current_instance_idx, None,
            )
            self.summary = {
                'success': False, 'app_id': self.app_id, 'status': str(error),
                'finish_reason': str(self.run_result.finish_reason), 'gaps': [str(error)],
            }
        finally:
            self.options = MacRunOptions()
            self.ctx.run_context.event_bus.unlisten_all_event(self)


class MacCatalogRunner(QThread):
    """目录查询不初始化上下文、采集或输入，不在主线程等待网络。"""

    catalog_ready = Signal(object)
    catalog_failed = Signal(str)

    def run(self) -> None:
        from zzz_od.macos.redemption_catalog import fetch_cn_catalog

        try:
            self.catalog_ready.emit(fetch_cn_catalog())
        except Exception as error:
            self.catalog_failed.emit(f'国服目录读取失败：{error}')


class MacRunOptionsPanel(QWidget):
    """选项只存在于当前页面；换任务或启动后重置，不保存到 YAML。"""

    def __init__(self, ctx: ZContext, app_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx: ZContext = ctx
        self.app_id: str = ''
        self.catalog_runner: MacCatalogRunner | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.hint: BodyLabel = BodyLabel('本次运行选项；换任务或开始后清空。整组使用已保存配置。')
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.claim_all: CheckBox = CheckBox('全部领取邮件附件（仅本次单项任务）')
        self.code: LineEdit = LineEdit()
        self.code.setPlaceholderText('明确输入一枚国服兑换码')
        self.catalog_button: PushButton = PushButton('查看国服兑换码目录（只读）')
        self.catalog_button.clicked.connect(self._load_catalog)
        self.tinman: CheckBox = CheckBox('本次仅选择汀曼特调')
        self.resume_lost_void: CheckBox = CheckBox('续接已确认的战线肃清中断挑战')
        self.store_refreshed: CheckBox = CheckBox('当前商店已核对刷新一次，本次不再刷新')
        self.resume_signin: CheckBox = CheckBox('续接已确认的当日占卜奖励或结果页')
        self.resume_lost_void.toggled.connect(self._update_store_option)
        for widget in (
            self.claim_all, self.code, self.catalog_button, self.tinman,
            self.resume_lost_void, self.store_refreshed, self.resume_signin,
        ):
            layout.addWidget(widget)
        self.preview_button: PushButton = PushButton('预览本次任务与适配范围')
        self.preview_button.clicked.connect(self.show_preview)
        layout.addWidget(self.preview_button)
        self.output: PlainTextEdit = PlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(110)
        self.output.setMaximumHeight(220)
        self.output.setPlaceholderText('预览、目录及运行结果会显示在这里')
        layout.addWidget(self.output)
        self.set_app(app_id)

    def set_app(self, app_id: str) -> None:
        """切换时清空授权与输入；刷新同一任务保留尚未启动的选择。"""
        if self.app_id != app_id:
            self.reset_options()
            self.output.clear()
        self.app_id = app_id
        for widget, target in (
            (self.claim_all, 'email'), (self.code, 'redemption_code'),
            (self.catalog_button, 'redemption_code'), (self.tinman, 'coffee'),
            (self.resume_lost_void, 'lost_void'), (self.store_refreshed, 'lost_void'),
            (self.resume_signin, 'daily_signin'),
        ):
            widget.setVisible(app_id == target)
        self.preview_button.setText('预览已启用任务与适配范围' if app_id == 'one_dragon' else '预览本次任务与适配范围')

    def reset_options(self) -> None:
        """开始或换任务后不保留上一项的特殊授权。"""
        self.code.clear()
        for widget in (self.claim_all, self.tinman, self.resume_lost_void, self.store_refreshed, self.resume_signin):
            widget.setChecked(False)
        self._update_store_option(False)

    def _update_store_option(self, selected: bool) -> None:
        self.store_refreshed.setEnabled(selected)
        if not selected:
            self.store_refreshed.setChecked(False)

    def options(self) -> MacRunOptions:
        """只读取当前任务可用的控件，隐藏选项不能影响其它任务。"""
        return MacRunOptions(
            redeem_code=(self.code.text().strip() or None) if self.app_id == 'redemption_code' else None,
            claim_all_mail=self.app_id == 'email' and self.claim_all.isChecked(),
            tinman_only=self.app_id == 'coffee' and self.tinman.isChecked(),
            resume_lost_void=self.app_id == 'lost_void' and self.resume_lost_void.isChecked(),
            resume_store_refreshed=self.app_id == 'lost_void' and self.store_refreshed.isChecked(),
            resume_signin_reward=self.app_id == 'daily_signin' and self.resume_signin.isChecked(),
        )

    def validate_options(self) -> MacRunOptions:
        """邮件和兑换码在显式选择后才允许启动。"""
        options = self.options()
        options.validate(self.app_id, self.ctx.game_account_config.game_region)
        return options

    def show_preview(self) -> None:
        """只读取当前任务及组配置，不初始化模型或刷新运行记录。"""
        if not self.app_id:
            self.output.setPlainText('请先选择一个应用')
            return
        try:
            preview = describe_request(self.ctx, self.app_id)
            options = self.options()
            lines = [f'账户实例：{preview["instance_idx"]}']
            lines.extend(options.describe())
            try:
                self.validate_options()
            except ValueError as error:
                lines.append(f'运行前待设置：{error}')
            if preview['unavailable_reason']:
                lines.append(f'运行限制：{preview["unavailable_reason"]}')
            for task in preview['tasks']:
                lines.append(f'{task["name"] or task["app_id"]}：{task["unavailable_reason"] or task["limits"]}')
            if not preview['tasks']:
                lines.append('当前组没有启用的任务')
            self.output.setPlainText('\n'.join(lines))
        except Exception as error:
            self.output.setPlainText(f'预览失败：{error}')

    def show_result(self, summary: dict[str, object] | None) -> None:
        """结束不等于完成，单独列出失败或未确认的任务。"""
        if summary is None:
            self.output.setPlainText('运行结果未确认，请查看日志')
            return
        if not summary.get('success'):
            title = '运行未成功'
        elif not summary.get('all_tasks_finished', True):
            title = '运行结束，仍有待完成项'
        else:
            title = '任务完成'
        lines = [title]
        if summary.get('app_id'):
            name = self.ctx.run_context.get_application_name(str(summary['app_id']))
            lines.append(f'运行任务：{name or summary["app_id"]}')
        finish_text = {
            'STOPPED': '运行已停止', 'FAILED': '运行失败', 'NOT_STARTED': '任务未开始',
        }.get(summary.get('finish_reason'))
        if finish_text:
            lines.append(finish_text)
        if summary.get('status'):
            lines.append(str(summary['status']))
        for task in summary.get('tasks', []):
            lines.append(f'{task.get("name") or task["app_id"]}：{task["outcome"]}')
        for gap in summary.get('gaps', []):
            lines.append(f'待完成：{gap}')
        self.output.setPlainText('\n'.join(lines))

    def _load_catalog(self) -> None:
        """仅用户点击时查询公开国服目录，结果不会自动填入兑换框。"""
        if self.catalog_runner is not None and self.catalog_runner.isRunning():
            return
        self.catalog_button.setEnabled(False)
        self.output.setPlainText('正在读取国服兑换码目录；不会启动游戏或兑换')
        runner = MacCatalogRunner(self)
        self.catalog_runner = runner
        runner.catalog_ready.connect(self._show_catalog)
        runner.catalog_failed.connect(self._show_catalog_error)
        runner.finished.connect(self._catalog_finished)
        runner.start()

    def _show_catalog(self, codes: list[dict[str, str]]) -> None:
        """目录始终只读；每次兑换仍须用户明确输入一枚。"""
        if self.app_id != 'redemption_code' or not self.isEnabled():
            return
        lines = ['国服目录（只读，日期按北京时间）：']
        lines.extend(f'{item["code"]} · 有效期至 {item["expiry_date_beijing"]}' for item in codes)
        if not codes:
            lines.append('当前目录没有有效兑换码')
        self.output.setPlainText('\n'.join(lines))

    def _show_catalog_error(self, message: str) -> None:
        if self.app_id == 'redemption_code' and self.isEnabled():
            self.output.setPlainText(message)

    def _catalog_finished(self) -> None:
        self.catalog_button.setEnabled(True)


class MacTaskRunDialog(MessageBoxBase):
    """一条龙卡片单独运行时使用同一个临时选项面板。"""

    def __init__(self, ctx: ZContext, app_id: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.panel: MacRunOptionsPanel = MacRunOptionsPanel(ctx, app_id, self)
        self.viewLayout.addWidget(self.panel)
        self.yesButton.setText('运行本项')
        self.cancelButton.setText('取消')
        self.widget.setMinimumWidth(520)
        self.panel.show_preview()

    def validate(self) -> bool:
        try:
            self.panel.validate_options()
            return True
        except ValueError as error:
            self.panel.output.setPlainText(str(error))
            return False
