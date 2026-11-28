"""区域巡防单场挑战，复用模拟室的编队、战斗保护与结算计数。"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, ClassVar

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.charge_transport import charge_mission_names
from zzz_od.operation.gamepad.combat_simulation import GamepadCombatSimulation
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.simulation_preset import (
    read_focused_choice,
    read_preset_area,
)

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.application.charge_plan.charge_plan_config import (
        ChargePlanConfig,
        ChargePlanItem,
    )
    from zzz_od.context.zzz_context import ZContext


class GamepadChargeBackToWorld(GamepadOperation):
    """从已识别的副本准备或出战页返回，不在未知页面连续按 B。"""

    def __init__(self, ctx: ZContext, category_name: str) -> None:
        super().__init__(ctx, op_name=f'手柄退出{category_name}', timeout_seconds=75)
        self.category_name: str = category_name
        self._deployment_back_sent: bool = False
        self._detail_back_sent: bool = False
        self._list_back_sent: bool = False

    def handle_init(self) -> None:
        self._deployment_back_sent = False
        self._detail_back_sent = False
        self._list_back_sent = False

    @operation_node(name='返回副本大世界', is_start_node=True, node_max_retry_times=30)
    def back(self) -> OperationRoundResult:
        """每个已确认页面只发送一次返回，并等待下一页。"""
        if self.category_name not in ('实战模拟室', '区域巡防', '专业挑战室'):
            return self.round_fail('未适配的副本返回分类')
        if self.last_screenshot is None:
            return self.round_retry('等待截图', wait=1)
        if self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            return self.round_success('已返回大世界')
        if all(self.round_by_find_area(self.last_screenshot, '实战模拟室', name).is_success for name in ('手柄-出战', '手柄-预备编队')):
            if not self._deployment_back_sent:
                self.ctx.controller.btn_tap('east')
                self._deployment_back_sent = True
            return self.round_retry('等待副本详情', wait=1)
        if all(self.round_by_find_area(self.last_screenshot, self.category_name, name).is_success for name in ('手柄-挑战等级', '手柄-下一步')):
            if not self._detail_back_sent:
                self.ctx.controller.btn_tap('east')
                self._detail_back_sent = True
            return self.round_retry('等待大世界', wait=1)
        if self.category_name == '实战模拟室' and all(
            self.round_by_find_area(self.last_screenshot, '实战模拟室', name).is_success
            for name in ('手柄-基础材料分类', '手柄-代理人晋升分类')
        ):
            if not self._list_back_sent:
                self.ctx.controller.btn_tap('east')
                self._list_back_sent = True
            return self.round_retry('等待模拟室分类页退出', wait=1)
        if self.category_name == '专业挑战室':
            # 活动目标在游戏列表中带引号，仍要求完整名字处于选中黄框内。
            names = tuple(variant for name in charge_mission_names('专业挑战室')
                          for variant in (name, f'「{name}」', f'「{name}', f'{name}」'))
            selected, _ = read_focused_choice(self.ctx, self.last_screenshot, '手柄-副本列表', names, '专业挑战室')
            if selected is not None:
                if not self._list_back_sent:
                    self.ctx.controller.btn_tap('east')
                    self._list_back_sent = True
                return self.round_retry('等待专业挑战室列表退出', wait=1)
        if self.node_retry_times >= 15:
            self.save_screenshot(prefix='macos_charge_exit_pending')
        return self.round_retry('等待已建档的副本返回画面', wait=1)


class GamepadAreaPatrol(GamepadCombatSimulation):
    """每场核对指定驱动盘副本及六十电量，再复用既有战斗闭环。"""

    STATUS_WRONG_TARGET: ClassVar[str] = '区域巡防副本标题或选中卡片明确不符'
    STATUS_WRONG_LEVEL: ClassVar[str] = '区域巡防挑战等级明确不是计划的Lv.60'

    def __init__(self, ctx: ZContext, plan: ChargePlanItem, prepare_only: bool = False, config: ChargePlanConfig | None = None) -> None:
        super().__init__(ctx, plan, prepare_only=prepare_only, config=config)
        self.op_name = '手柄区域巡防单次挑战'
        self.timeout_seconds = 600
        self._level_confirmed: bool = False
        self._guard_plan = self._plan_identity()
        self._detail_frame_time: float = 0
        self._next_frame_time: float = 0
        self._ready_frame_time: float = 0
        self._last_frame_time: float = 0
        self._next_sent_at: float = 0
        self._proof_started: float = 0

    def _plan_identity(self) -> tuple[str | int | None, ...]:
        """绑定本场计划和已完成次数，防止计划变更后复用旧证明。"""
        return (self.plan.plan_id, self.plan.tab_name, self.plan.category_name,
                self.plan.mission_type_name, self.plan.level, self.plan.run_times,
                self.plan.plan_times, self.plan.predefined_team_idx)

    def _plan_changed(self) -> bool:
        """仅显式等级保护绑定计划；旧默认等级保持原有行为。"""
        return ((self._guard_plan[4] != '默认等级' or self.plan.level != '默认等级')
                and self._guard_plan != self._plan_identity())

    def _clear_level_proof(self) -> None:
        """清除页面间的单场等级证明。"""
        self._level_confirmed = False
        self._detail_frame_time = 0
        self._next_frame_time = 0
        self._ready_frame_time = 0
        self._last_frame_time = 0
        self._next_sent_at = 0
        self._proof_started = 0

    def _fresh_frame(self) -> bool:
        """仅接受最近五秒内实际采集的帧，不接受未来或无时间戳的帧。"""
        age = time.time() - self.last_screenshot_time
        return self.last_screenshot is not None and 0 <= age <= 5

    def _new_frame(self) -> bool:
        """控制器返回同一缓存帧时保留原采集时间，不能当成新观察。"""
        if not self._fresh_frame() or self.last_screenshot_time <= self._last_frame_time:
            return False
        self._last_frame_time = self.last_screenshot_time
        return True

    def _proof_valid(self) -> bool:
        """编队页面不含等级，因此仅接续两分钟内本场详情证明。"""
        age = time.monotonic() - self._proof_started
        return (self._level_confirmed and self._next_sent and self._proof_started > 0
                and 0 <= age <= 120 and self._guard_plan == self._plan_identity())

    def handle_init(self) -> None:
        """每次执行重新核等级，不复用上一场或上次中断的确认。"""
        super().handle_init()
        self._guard_plan = self._plan_identity()
        self._clear_level_proof()

    def handle_resume(self) -> None:
        """暂停期间游戏可能变化，恢复后不能沿用旧页面证明。"""
        self._clear_level_proof()
        super().handle_resume()

    def _level_state(self) -> str:
        """只用唯一的完整 Lv.60 作证明；短暂多框/杂字等待新帧。"""
        if self.plan.level != '等级Lv.60':
            return 'invalid'
        area = read_preset_area(self.ctx, self.last_screenshot, '手柄-挑战等级数值', '区域巡防')
        if area is None or not area[1]:
            return 'wait'
        readable = False
        for text, matches in area[1].items():
            value = text.replace(' ', '').strip()
            match = re.fullmatch(r'(?:等级)?[Ll][Vv]\.(\d{1,3})', value)
            if match is not None:
                if match[1] != '60':
                    return 'invalid'
                readable = len(matches) == 1
        return 'ok' if readable and len(area[1]) == 1 else 'wait'

    def _level_matches(self) -> bool:
        """仅用于只读识别；默认等级保持既有行为。"""
        return self.plan.level == '默认等级' or self._level_state() == 'ok'

    @property
    def expected_charge(self) -> int:
        """只用于比对，是否足够仍读取实际画面。"""
        return 60

    def _found(self, area_name: str) -> bool:
        """区域巡防有独立的等级位置，编队页继续共用已验证区域。"""
        if area_name in ('手柄-挑战等级', '手柄-下一步'):
            return self.last_screenshot is not None and self.round_by_find_area(self.last_screenshot, '区域巡防', area_name).is_success
        return super()._found(area_name)

    def _preset_selected(self) -> bool:
        """详情标题和列表黄框必须同时指向配置副本。"""
        title = read_preset_area(self.ctx, self.last_screenshot, '手柄-副本标题', '区域巡防')
        if title is None or not any(text.replace(' ', '') == self.plan.mission_type_name for text in title[1]):
            return False
        if self.plan.level != '默认等级':
            if len(title[1]) != 1 or len(next(iter(title[1].values()))) != 1:
                return False
        selected, _ = read_focused_choice(
            self.ctx, self.last_screenshot, '手柄-副本列表', charge_mission_names('区域巡防'), '区域巡防',
            border_gap=35, allow_corner_gap=True, reread_card_name=True,
        )
        return selected == self.plan.mission_type_name

    def _target_state(self) -> str:
        """目标标题和唯一焦点都须吻合；缺失等待，明确冲突拒绝。"""
        area = read_preset_area(self.ctx, self.last_screenshot, '手柄-副本标题', '区域巡防')
        if area is None or not area[1]:
            return 'wait'
        names = charge_mission_names('区域巡防')
        target = self.plan.mission_type_name
        titles = {text.replace(' ', '') for text in area[1]}
        if any(name != target and name in names for name in titles):
            return 'invalid'
        # 标题仍须有且仅有一个识别框。噪声不算通过，只复读下一张新帧。
        if len(area[1]) != 1 or target not in titles or len(next(iter(area[1].values()))) != 1:
            return 'wait'
        selected, _ = read_focused_choice(
            self.ctx, self.last_screenshot, '手柄-副本列表', names, '区域巡防',
            border_gap=35, allow_corner_gap=True,
        )
        if selected is None:
            return 'wait'
        return 'ok' if selected == target else 'invalid'

    def _detail_mismatch(self, target: str, level: str) -> OperationRoundResult | None:
        """明确读到其他目标或等级时立即停，并标明是哪一项。"""
        if 'invalid' not in (target, level):
            return None
        self._clear_level_proof()
        if target == 'invalid':
            return self.round_fail(self.STATUS_WRONG_TARGET)
        return self.round_fail(self.STATUS_WRONG_LEVEL)

    def _charge_state(self) -> tuple[str, tuple[int, int] | None]:
        """只接受唯一完整费用；可识别的冲突立即拒绝，缺读数等待。"""
        area = read_preset_area(self.ctx, self.last_screenshot, '手柄-电量')
        if area is None or not area[1]:
            return 'wait', None
        values: list[tuple[int, int]] = []
        unknown = False
        for text, matches in area[1].items():
            match = re.fullmatch(r'\s*(\d{1,4})\s*/\s*(\d{1,3})\s*', text)
            if match is not None:
                if len(matches) != 1:
                    return 'invalid', None
                values.append((int(match[1]), int(match[2])))
            elif (text == '三' and len(matches) == 1 and matches.max is not None
                  and matches.max.rect.x1 >= 105):
                continue
            else:
                unknown = True
        if len(values) > 1:
            return 'invalid', None
        if unknown or not values:
            return 'wait', None
        return 'ok', values[0]

    def _charge(self) -> tuple[int, int] | None:
        if self.plan.level == '默认等级':
            return super()._charge()
        state, charge = self._charge_state()
        return charge if state == 'ok' else None

    @operation_node(name='检查手柄计划', is_start_node=True)
    def check_plan(self) -> OperationRoundResult:
        """只接受原数据中的普通巡防副本，不将特训目标当普通关卡。"""
        if self.plan.tab_name != '训练' or self.plan.category_name != '区域巡防' or self.plan.mission_type_name not in charge_mission_names('区域巡防'):
            return self.round_fail('区域巡防目标不在已支持的原副本数据中')
        if self.plan.level not in ('默认等级', '等级Lv.60') or self.plan.predefined_team_idx not in range(-1, 20):
            return self.round_fail('区域巡防暂支持默认等级或严格核验Lv.60，与当前编队或二十个预备编队')
        if self.plan.plan_times <= 0 or self.plan.run_times < 0 or self.plan.run_times >= self.plan.plan_times:
            return self.round_fail('计划次数无效或已完成')
        return self.round_success()

    @node_from(from_name='加载既有自动战斗')
    @operation_node(name='配置挑战卡片', node_max_retry_times=8, timeout_seconds=20)
    def prepare_cards(self) -> OperationRoundResult:
        """巡防详情须确认目标与等级；缺少证据只有限等待。"""
        if self._plan_changed():
            self._clear_level_proof()
            return self.round_fail('本场计划已变化，停止区域巡防')
        if self._deployment_visible():
            self._clear_level_proof()
            if not self._return_to_preset_sent:
                self.ctx.controller.btn_tap('east')
                self._return_to_preset_sent = True
            return self.round_retry('返回详情重新确认驱动盘副本', wait=1)
        if self.plan.level == '默认等级':
            if self._found('手柄-挑战等级') and self._found('手柄-下一步'):
                if not self._preset_selected():
                    self.save_screenshot(prefix='macos_patrol_wrong_target')
                    return self.round_fail(f'当前详情并非计划副本{self.plan.mission_type_name}')
                return self.round_success('驱动盘副本已确认')
            return self.round_retry('等待区域巡防详情', wait=1)
        if not self._new_frame():
            return self.round_retry('等待新鲜巡防详情帧', wait=1)
        if not (self._found('手柄-挑战等级') and self._found('手柄-下一步')):
            return self.round_retry('等待区域巡防详情', wait=1)
        target, level = self._target_state(), self._level_state()
        mismatch = self._detail_mismatch(target, level)
        if mismatch is not None:
            return mismatch
        if target != 'ok' or level != 'ok' or not self._fresh_frame():
            return self.round_retry('等待明确的巡防目标与Lv.60', wait=1)
        self._level_confirmed = True
        self._detail_frame_time = self.last_screenshot_time
        return self.round_success('驱动盘副本已确认')

    @node_from(from_name='配置挑战卡片')
    @operation_node(name='准备当前编队', node_max_retry_times=10, timeout_seconds=180)
    def prepare_deployment(self) -> OperationRoundResult:
        """下一步前重读详情；发送后等待过渡与新编队页，不重复输入。"""
        if self._plan_changed():
            self._clear_level_proof()
            return self.round_fail('本场计划已变化，停止区域巡防')
        if self.plan.level == '默认等级':
            return super().prepare_deployment()
        if not self._new_frame():
            return self.round_retry('等待新的有效采集帧', wait=1)
        if self._next_sent:
            if not self._proof_valid() or self.last_screenshot_time <= self._next_sent_at:
                return self.round_retry('等待本场有效的编队证明', wait=1)
            if not self._deployment_visible():
                if self._found('手柄-挑战等级') and self._found('手柄-下一步'):
                    mismatch = self._detail_mismatch(self._target_state(), self._level_state())
                    if mismatch is not None:
                        return mismatch
                return self.round_retry('等待下一步后的编队页面', wait=1)
            state, charge = self._charge_state()
            if state == 'invalid':
                return self.round_fail('编队电量读数有歧义')
            if state != 'ok' or charge is None:
                return self.round_retry('等待编队电量读数', wait=1)
            if charge[1] != self.expected_charge:
                return self.round_fail(f'当前编队消耗 {charge[1]} 电量，与目标 {self.expected_charge} 不一致')
            if charge[0] < charge[1]:
                return self.round_fail(self.STATUS_CHARGE_NOT_ENOUGH)
            result = super().prepare_deployment()
            if result.is_success:
                if not self._fresh_frame() or not self._proof_valid():
                    return self.round_retry('等待新鲜的编队证明', wait=1)
                self._ready_frame_time = self.last_screenshot_time
            return result
        if not self._level_confirmed or self.last_screenshot_time <= self._detail_frame_time:
            return self.round_retry('等待本场新的Lv.60详情帧', wait=1)
        if not (self._found('手柄-挑战等级') and self._found('手柄-下一步')):
            return self.round_retry('等待区域巡防详情', wait=1)
        target, level = self._target_state(), self._level_state()
        mismatch = self._detail_mismatch(target, level)
        if mismatch is not None:
            return mismatch
        state, charge = self._charge_state()
        if state == 'invalid':
            return self.round_fail('区域巡防电量读数有歧义')
        if target != 'ok' or level != 'ok' or state != 'ok' or charge is None:
            return self.round_retry('等待巡防目标、等级和电量读数', wait=1)
        if charge[1] != self.expected_charge:
            return self.round_fail(f'当前预设消耗 {charge[1]} 电量，与目标 {self.expected_charge} 不一致')
        if charge[0] < charge[1]:
            return self.round_fail(self.STATUS_CHARGE_NOT_ENOUGH)
        if not self._fresh_frame():
            return self.round_retry('等待新鲜的巡防详情', wait=1)
        self.ctx.controller.btn_tap('start')
        self._next_sent = True
        self._next_sent_at = time.time()
        self._next_frame_time = self.last_screenshot_time
        self._proof_started = time.monotonic()
        return self.round_retry('等待出战准备页', wait=1)

    @node_from(from_name='准备当前编队', status=GamepadCombatSimulation.STATUS_DEPLOYMENT_READY)
    @operation_node(name='手柄出战', node_max_retry_times=10, timeout_seconds=15)
    def deploy(self) -> OperationRoundResult:
        """仅在本场证明及新鲜编队页均有效时出战一次。"""
        if self._plan_changed():
            self._clear_level_proof()
            return self.round_fail('本场计划已变化，停止区域巡防')
        if self.plan.level == '默认等级':
            return super().deploy()
        if self._deploy_sent:
            if self._proof_valid() and self._ready_frame_time:
                return self.round_success()
            return self.round_retry('等待本场出战证明', wait=1)
        if (not self._new_frame() or not self._proof_valid() or not self._ready_frame_time
                or self.last_screenshot_time <= max(self._ready_frame_time, self._next_sent_at)):
            return self.round_retry('等待本场新鲜编队证明', wait=1)
        if not self._deployment_visible():
            return self.round_retry('等待出战准备页', wait=1)
        state, charge = self._charge_state()
        if state == 'invalid':
            return self.round_fail('出战电量读数有歧义')
        if state != 'ok' or charge is None:
            return self.round_retry('等待出战电量读数', wait=1)
        if charge[1] != self.expected_charge:
            return self.round_fail(f'出战消耗 {charge[1]} 电量，与目标 {self.expected_charge} 不一致')
        if charge[0] < charge[1]:
            return self.round_fail(self.STATUS_CHARGE_NOT_ENOUGH)
        if not self._fresh_frame() or not self._proof_valid():
            return self.round_retry('等待新鲜的出战证明', wait=1)
        self.ctx.controller.btn_tap('start')
        self._deploy_sent = True
        return self.round_success(wait=1)

    def move_to_battle(self) -> None:
        """沿用原区域巡防的一秒前移，在已有战斗输入保护内执行。"""
        self.ctx.controller.move_w(press=True, press_time=1, release=True)

    @node_from(from_name='确认结算返回')
    @operation_node(name='手柄返回大世界', screenshot_before_round=False, timeout_seconds=90)
    def back_to_world(self) -> OperationRoundResult:
        """复用同一已建档退出操作，用于完成与资源不足。"""
        return self.round_by_op_result(GamepadChargeBackToWorld(self.ctx, '区域巡防').execute())
