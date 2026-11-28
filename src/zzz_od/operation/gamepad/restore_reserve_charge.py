"""原体力计划的最小储蓄补充；确认前落盘，结果不明不重领。"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import os_utils
from zzz_od.application.charge_plan.charge_plan_config import (
    MACOS_RESERVE_DAILY_LIMIT,
    RestoreChargeEnum,
)
from zzz_od.operation.gamepad.area_patrol import GamepadChargeBackToWorld
from zzz_od.operation.gamepad.charge_balance import GamepadReadChargeBalances
from zzz_od.operation.gamepad.focus import has_filled_highlight
from zzz_od.operation.gamepad.reserve_charge import (
    ChargeResourceSnapshot,
    ReserveChargeBudget,
    remaining_plan_energy,
)
from zzz_od.operation.gamepad.rewards import _GamepadRewardOperation
from zzz_od.operation.gamepad.transport import find_yellow_focus_frames

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.application.charge_plan.charge_plan_config import ChargePlanConfig
    from zzz_od.context.zzz_context import ZContext
    from zzz_od.operation.gamepad.combat_simulation import GamepadCombatSimulation


def reserve_budget(config: ChargePlanConfig) -> ReserveChargeBudget:
    """显式日历日和正额度缺一不可，默认仍不支持自动补电。"""
    allowed = date.fromisoformat(config.get('macos_reserve_date', ''))
    maximum = config.get('macos_reserve_limit', 0)
    if type(maximum) is not int or not 0 < maximum <= MACOS_RESERVE_DAILY_LIMIT:
        raise ValueError(f'Mac储蓄恢复需要1至{MACOS_RESERVE_DAILY_LIMIT}的单日明确额度')
    setting = config.restore_charge
    if setting == RestoreChargeEnum.BOTH.value.value:
        setting = RestoreChargeEnum.BACKUP_ONLY.value.value
    budget = ReserveChargeBudget(allowed, maximum, setting)
    budget._check_date(datetime.now(ZoneInfo('Asia/Shanghai')))
    if config.loop or config.double_reward:
        raise ValueError('储蓄恢复不支持循环或双倍计划')
    return budget


def charge_restore_budget(config: ChargePlanConfig) -> ReserveChargeBudget:
    """电池单独授权沿用原恢复开关，仍核对当日共享回执。"""
    if config.restore_charge != RestoreChargeEnum.ETHER_ONLY.value.value:
        return reserve_budget(config)
    if config.loop or config.double_reward:
        raise ValueError('Mac电量恢复暂不支持循环或双倍计划')
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date()
    return ReserveChargeBudget(day, 0, RestoreChargeEnum.BACKUP_ONLY.value.value)


class GamepadRestoreReserveCharge(_GamepadRewardOperation):
    """补足本场预设的实际差额，再通过手册和仓库核账。"""

    STATUS_SHORTAGE: ClassVar[str] = '可用恢复来源不足'
    RESERVE: ClassVar[str] = '储蓄电量'
    ETHER: ClassVar[str] = '以太电池'
    SOURCES: ClassVar[tuple[str, ...]] = (RESERVE, ETHER, '菲林')
    SOURCE_DETAILS: ClassVar[dict[str, str]] = {
        RESERVE: '取出等量储备电量', ETHER: '使用后恢复60电量',
    }

    def __init__(
        self, ctx: ZContext, config: ChargePlanConfig, challenge: GamepadCombatSimulation,
    ) -> None:
        super().__init__(ctx, '手柄最小电量恢复', timeout_seconds=250)
        self.config: ChargePlanConfig = config
        self.challenge: GamepadCombatSimulation = challenge
        self.budget: ReserveChargeBudget = charge_restore_budget(config)
        self._mode: str = config.restore_charge
        self._plan_at_start: dict[str, str | int | None] = challenge.plan.to_dict()
        self._required: int | None = challenge.expected_charge
        self._instance_idx: int = ctx.current_instance_idx
        self.journal: Path = (
            Path(os_utils.get_work_dir())
            / '.debug'
            / 'macos'
            / f'reserve_{ctx.current_instance_idx}_{self.budget.allowed_date.isoformat()}.json'
        )
        self._opened: bool = False
        self._source_sent: bool = False
        self._confirm_sent: bool = False
        self._reward_sent: bool = False
        self._initial_energy: int | None = None
        self._after_energy: int | None = None
        self._regen_deadline: float = 0
        self._before: ChargeResourceSnapshot | None = None
        self._after: ChargeResourceSnapshot | None = None
        self._amount: int = 0
        self._source: str = self.RESERVE
        self._source_action: tuple[str, str, bool] | None = None

    def _area(self, name: str) -> tuple[MatLike, dict[str, MatchResultList]] | None:
        return self._read_area('恢复电量', name)

    def _texts(self, name: str) -> set[str]:
        area = self._area(name)
        return (
            {unicodedata.normalize('NFKC', text).replace(' ', '') for text in area[1]}
            if area
            else set()
        )

    def _range_value(self, name: str) -> int | None:
        """滑块端点为单行数字，检测漏框时使用原单行识别复读。"""
        words = self._texts(name)
        if not words and self.last_screenshot is not None:
            area = self.ctx.screen_loader.get_area('恢复电量', name)
            if area is not None:
                rect = area.rect
                image = self.last_screenshot[rect.y1:rect.y2, rect.x1:rect.x2]
                if image.size:
                    text = self.ctx.ocr.run_ocr_single_line(image, threshold=0.85)
                    words = {unicodedata.normalize('NFKC', text).strip()}
        if len(words) == 1:
            value = next(iter(words))
            if value.isdecimal():
                return int(value)
        return None

    def _main(self) -> bool:
        return '恢复电量' in self._texts('手柄-恢复标题')

    def _source_data(self, energy: int) -> tuple[ChargeResourceSnapshot, str] | None:
        """核对三个来源的名称、独立余额及唯一手柄焦点。"""
        area = self._area('手柄-恢复来源')
        if area is None:
            return None
        image, words = area
        names = self.SOURCES
        positions: dict[str, float] = {}
        numbers: list[tuple[float, int]] = []
        for text, matches in words.items():
            text = unicodedata.normalize('NFKC', text).replace(' ', '')
            if matches.max is None:
                continue
            if text in names:
                positions[text] = matches.max.center.x
            elif text.isdecimal():
                numbers.append((matches.max.center.x, int(text)))
        if set(positions) != set(names) or len(numbers) != 3:
            return None
        frames = find_yellow_focus_frames(image, 90, 160, 90, 160)
        if len(frames) != 1 or not positions[names[0]] < positions[names[1]] < positions[names[2]]:
            return None
        focused = [name for name in names if abs(frames[0].center.x - positions[name]) <= 35]
        if len(focused) != 1:
            return None
        values = []
        for name in names:
            nearby = [value for x, value in numbers if abs(x - positions[name]) < 35]
            if len(nearby) != 1:
                return None
            values.append(nearby[0])
        return ChargeResourceSnapshot(energy, *values), focused[0]

    def _source_confirm_focused(self) -> bool | None:
        area = self._area('手柄-恢复页确认')
        if area is None:
            return None
        image, words = area
        matches = [match for text, results in words.items() if text.strip() == '确认' for match in results]
        return has_filled_highlight(image, matches[0].rect, 'yellow') if len(matches) == 1 else None

    def _select_source(self, balance: ChargeResourceSnapshot, required: int) -> tuple[str, int] | None:
        """储蓄足以覆盖差额时优先使用，电池按所需整枚取用。"""
        gap = required - balance.energy
        if gap <= 0:
            return None
        if (self._mode != RestoreChargeEnum.ETHER_ONLY.value.value
                and min(balance.reserve, self.budget.maximum_reserve - self.budget.restored) >= gap):
            return self.RESERVE, gap
        units = (gap + 59) // 60
        if self._mode != RestoreChargeEnum.BACKUP_ONLY.value.value and balance.ether_batteries >= units:
            return self.ETHER, units
        return None

    def _focus_source(self, wanted: str, focus: str, selected: str, confirming: bool) -> bool:
        """将焦点移到指定来源，确认键只用于已识别的目标卡片。"""
        if focus == selected == wanted and confirming:
            return True
        signature = (focus, selected, confirming)
        if signature == self._source_action:
            return False
        if confirming:
            key = 'dpad_up'
        elif focus != wanted:
            key = 'dpad_right' if self.SOURCES.index(focus) < self.SOURCES.index(wanted) else 'dpad_left'
        else:
            key = 'dpad_down' if selected == wanted else 'south'
        self.ctx.controller.btn_tap(key)
        self._source_action = signature
        return False

    @staticmethod
    def _load_journal(path: Path) -> dict[str, Any]:
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError(f'无法核对储蓄回执：{path.name}') from error
        if not isinstance(saved, dict):
            raise ValueError(f'储蓄回执格式无效：{path.name}')
        return saved

    def _verified_journal_total(self) -> int:
        """跨进程读取累计量，跨日仍阻止重领尚未核实的旧操作。"""
        paths = set(self.journal.parent.glob(f'reserve_{self._instance_idx}_*.json'))
        if self.journal.exists():
            paths.add(self.journal)
        total = 0
        for path in paths:
            saved = self._load_journal(path)
            if saved.get('state') != 'verified':
                raise ValueError(f'已有储蓄领取待核账，禁止重领：{path.name}')
            if path == self.journal:
                total = saved.get('restored_total')
                if type(total) is not int or total < 0:
                    raise ValueError(f'无法核对本日累计储蓄：{path.name}')
        return total

    def _save_journal(self, state: str) -> None:
        """独占校验及写回回执，确认前持久化领取意图。"""
        import fcntl

        data = {
            'state': state,
            'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'before': asdict(self._before) if self._before else None,
            'after': asdict(self._after) if self._after else None,
            'source': self._source,
            'amount': self._amount,
            'restored_total': self.budget.restored,
            'evidence_paths': self.evidence_paths,
        }
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        lock = self.journal.with_name(f'reserve_{self._instance_idx}.lock')
        with lock.open('a', encoding='utf-8') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if state == 'confirmation_issued':
                if self._verified_journal_total() != self.budget.restored:
                    raise ValueError('本日累计储蓄已变化，未提交新的补电')
            else:
                saved = self._load_journal(self.journal)
                if (saved.get('state') != 'confirmation_issued' or saved.get('before') != data['before']
                        or saved.get('amount') != self._amount or saved.get('source') != self._source):
                    raise ValueError('待核账回执与本次领取不一致')
            temporary = self.journal.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8') as receipt:
                json.dump(data, receipt, ensure_ascii=False, indent=2)
                receipt.write('\n')
                receipt.flush()
                os.fsync(receipt.fileno())
            temporary.replace(self.journal)

    def _plan_matches(self) -> bool:
        plan = self.challenge.plan
        authorization = charge_restore_budget(self.config)
        return (any(plan is item for item in self.config.plan_list) and not plan.skipped
                and plan.to_dict() == self._plan_at_start and self._required is not None
                and self.config.restore_charge == self._mode
                and authorization.allowed_date == self.budget.allowed_date
                and authorization.maximum_reserve == self.budget.maximum_reserve
                and remaining_plan_energy((plan,), observed_cost=self._required) >= self._required)

    def _challenge_charge(self) -> tuple[int, int] | None:
        """复用同一副本选择和真实电量读数，避免已换目标的页面被确认。"""
        self.challenge.last_screenshot = self.last_screenshot
        if (not self.challenge._found('手柄-挑战等级') or not self.challenge._found('手柄-下一步')
                or not self.challenge._preset_selected()):
            return None
        charge = self.challenge.read_charge()
        return charge if charge is not None and charge[1] == self._required else None

    @operation_node(name='打开不足电量来源', is_start_node=True, node_max_retry_times=8)
    def open_source(self) -> OperationRoundResult:
        """只从本次仍不足的已确认副本进入，并检查已有回执。"""
        if self._opened:
            if self._main():
                return self.round_success()
            return self.round_retry('等待储蓄来源页', wait=0.8)
        try:
            self.budget.restored = self._verified_journal_total()
            if not self._plan_matches():
                return self.round_fail('本场计划或实际消耗未获确认，未打开补电')
        except ValueError as error:
            return self.round_fail(str(error))
        charge = self._challenge_charge()
        if charge is None:
            return self.round_retry('等待本场副本与完整电量读数，未打开补电', wait=0.8)
        if charge[0] >= charge[1]:
            return self.round_fail('当前电量已足够，未打开补电')
        self._initial_energy = charge[0]
        if not self._save_evidence('reserve_preset_before'):
            return self.round_fail('未能留存副本补电前状态')
        self.ctx.controller.btn_tap('start')
        self._opened = True
        return self.round_retry('等待储蓄来源页', wait=0.8)

    @node_from(from_name='打开不足电量来源')
    @operation_node(name='核对储蓄来源', node_max_retry_times=12)
    def choose_source(self) -> OperationRoundResult:
        if self._source_sent:
            return self.round_success()
        if not self._main() or self._initial_energy is None or self._required is None:
            return self.round_fail('储蓄来源页不明确')
        timers = [
            re.search(r'电量恢复(\d+)分(\d+)秒', text)
            for text in self._texts('手柄-恢复倒计时')
        ]
        seconds = [int(m[1]) * 60 + int(m[2]) for m in timers if m]
        if len(seconds) != 1:
            return self.round_retry('等待恢复倒计时', wait=0.8)
        deadline = time.monotonic() + seconds[0]
        self._regen_deadline = min(self._regen_deadline or deadline, deadline)
        if time.monotonic() + 20 >= self._regen_deadline:
            return self.round_fail('自然恢复时刻接近或倒计时不明，未领取储蓄')
        state = self._source_data(self._initial_energy)
        confirming = self._source_confirm_focused()
        description = self._texts('手柄-恢复说明')
        selected = [source for source, label in self.SOURCE_DETAILS.items() if label in description]
        if state is None or confirming is None or len(selected) != 1:
            return self.round_retry('等待来源说明、焦点和三项余额', wait=0.8)
        self._before, focus = state
        choice = self._select_source(self._before, self._required)
        if choice is None:
            self.ctx.controller.btn_tap('east')
            return self.round_success(self.STATUS_SHORTAGE, wait=0.8)
        source, amount = choice
        if not self._focus_source(source, focus, selected[0], confirming):
            return self.round_retry('等待所选来源及确认焦点', wait=0.8)
        if source == self.RESERVE:
            transfer = self.budget.prepare(
                self._before, self._required,
                remaining_plan_energy((self.challenge.plan,), observed_cost=self._required),
                1, 240, datetime.now(ZoneInfo('Asia/Shanghai')),
            )
            if transfer is None or transfer.amount != amount:
                return self.round_fail('储蓄额度或实际差额已变化')
        self._source = source
        self._amount = amount
        self._save_evidence('reserve_source_before')
        self.ctx.controller.btn_tap('south')
        self._source_sent = True
        return self.round_success(wait=0.8)

    @node_from(from_name='核对储蓄来源', status=STATUS_SHORTAGE)
    @operation_node(name='储蓄不足后核对原副本', node_max_retry_times=6)
    def return_without_restore(self) -> OperationRoundResult:
        charge = self._challenge_charge()
        if charge is None:
            return self.round_retry('等待储蓄来源关闭后核对原副本', wait=0.8)
        return self.round_success(self.STATUS_SHORTAGE, data={'current_charge': charge[0]})

    @node_from(from_name='核对储蓄来源')
    @operation_node(name='确认最小兑换量', node_max_retry_times=8)
    def confirm_quantity(self) -> OperationRoundResult:
        if self._confirm_sent:
            return self.round_success()
        quantities = self._texts('手柄-兑换数值')
        if self._before is None:
            return self.round_fail('未保存恢复前余额')
        previous_amount = self._before.reserve if self._source == self.RESERVE else self._before.ether_batteries
        if (
            '快捷使用' not in self._texts('手柄-快捷标题')
            or not any(
                re.fullmatch(rf'{self._source}[×xX]{previous_amount}', text)
                for text in self._texts('手柄-快捷来源')
            )
            or quantities != {str(self._amount)}
        ):
            return self.round_retry('等待准确恢复来源和最小数量，未确认', wait=0.8)
        if time.monotonic() + 10 >= self._regen_deadline:
            return self.round_fail('自然恢复即将改变差额，未确认补电')
        minimum = self._range_value('手柄-兑换最小')
        maximum = self._range_value('手柄-兑换最大')
        known_limits = (240,) if self._source == self.RESERVE else (10, min(previous_amount, 10))
        prompt = self._texts('手柄-兑换确认')
        if (minimum != 1 or maximum is None or maximum not in known_limits or self._amount > maximum
                or not ({'A', '确认'} <= prompt or 'A确认' in prompt)):
            # 快捷页初现时小字可能尚未被完整识别，等待稳定画面后再确认。
            return self.round_retry('等待明确的兑换范围和 A 确认提示', wait=0.8)
        if not self._save_evidence('reserve_quantity_before_confirm'):
            return self.round_fail('补电确认前截图保存失败，未使用资源')
        try:
            if not self._plan_matches():
                return self.round_fail('确认前计划已变化，未补电')
            if self._source == self.RESERVE:
                self.budget.check_confirmation(
                    self.RESERVE, self._amount, self._before, datetime.now(ZoneInfo('Asia/Shanghai')),
                )
            self._save_journal('confirmation_issued')
        except (OSError, ValueError) as error:
            return self.round_fail(str(error))
        self._confirm_sent = True
        self.ctx.controller.btn_tap('south')
        return self.round_success(wait=1)

    def _gained_energy(self) -> int:
        return self._amount * 60 if self._source == self.ETHER else self._amount

    @node_from(from_name='确认最小兑换量')
    @operation_node(name='确认储蓄获得', node_max_retry_times=10)
    def confirm_obtained(self) -> OperationRoundResult:
        if self._known_reward_visible():
            self._save_evidence('reserve_obtained')
            if not self._reward_sent:
                self.ctx.controller.btn_tap('south')
                self._reward_sent = True
            return self.round_retry('等待电量获得后返回原副本', wait=0.8)
        charge = self._challenge_charge()
        if self._reward_sent and charge is not None:
            if self._before is None or charge[0] != self._before.energy + self._gained_energy():
                return self.round_fail('补电增量不明确，保留待核账，不再次领取')
            self._after_energy = charge[0]
            self._save_evidence('reserve_preset_after')
            return self.round_success()
        return self.round_retry('等待本次储蓄获得及返回证据，未重发确认', wait=0.8)

    @node_from(from_name='确认储蓄获得')
    @operation_node(name='核对补电扣减', screenshot_before_round=False, timeout_seconds=190)
    def verify_resources(self) -> OperationRoundResult:
        """独立读取手册和仓库，保留精确获得值并容许核账期间一格自然恢复。"""
        result = GamepadChargeBackToWorld(self.ctx, self.challenge.plan.category_name).execute()
        if not result.success:
            return self.round_by_op_result(result)
        result = GamepadReadChargeBalances(self.ctx).execute()
        if not result.success:
            return self.round_by_op_result(result)
        balance = ChargeResourceSnapshot(**result.data['balances'])
        if self._after_energy is None or not self._after_energy <= balance.energy <= self._after_energy + 1:
            return self.round_fail('手册电量与获得后的副本读数不一致，保留待核账')
        self.evidence_paths.update(result.data['evidence_paths'])
        self._after = replace(balance, energy=self._after_energy)
        try:
            if self._source == self.RESERVE:
                self.budget.record_result(self._after)
            elif (self._before is None or self._after.energy != self._before.energy + self._gained_energy()
                    or self._after.ether_batteries != self._before.ether_batteries - self._amount
                    or self._after.reserve != self._before.reserve
                    or self._after.premium_currency != self._before.premium_currency):
                return self.round_fail('以太电池使用结果未能对账，禁止再次补电')
            self._save_journal('verified')
        except (OSError, ValueError) as error:
            return self.round_fail(str(error))
        return self.round_success(
            f'{self._source}恢复已对账',
            data={
                'source': self._source,
                'amount': self._amount,
                'gained_energy': self._gained_energy(),
                'before': asdict(self._before),
                'after': asdict(self._after),
                'journal': str(self.journal),
            },
        )
