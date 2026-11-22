"""区域巡防单场挑战，复用模拟室的编队、战斗保护与结算计数。"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

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

    def __init__(self, ctx: ZContext, plan: ChargePlanItem, prepare_only: bool = False, config: ChargePlanConfig | None = None) -> None:
        super().__init__(ctx, plan, prepare_only=prepare_only, config=config)
        self.op_name = '手柄区域巡防单次挑战'
        self.timeout_seconds = 600
        self._level_confirmed: bool = False
        self._guard_plan = self._plan_identity()
        self._detail_frame_time: float = 0
        self._next_frame_time: float = 0
        self._ready_frame_time: float = 0
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
        self._proof_started = 0

    def _fresh_frame(self) -> bool:
        """仅接受最近五秒内实际采集的帧，不接受未来或无时间戳的帧。"""
        age = time.time() - self.last_screenshot_time
        return self.last_screenshot is not None and 0 <= age <= 5

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

    def _level_matches(self) -> bool:
        """显式等级只接受独立数值区域的完整读数，不把六十电量当等级。"""
        if self.plan.level == '默认等级':
            return True
        if self.plan.level != '等级Lv.60':
            return False
        area = read_preset_area(self.ctx, self.last_screenshot, '手柄-挑战等级数值', '区域巡防')
        if area is None or len(area[1]) != 1:
            return False
        text, matches = next(iter(area[1].items()))
        if len(matches) != 1:
            return False
        return re.fullmatch(r'(?:等级)?[Ll][Vv]\.60', text.replace(' ', '').strip()) is not None

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
        selected, _ = read_focused_choice(self.ctx, self.last_screenshot, '手柄-副本列表', charge_mission_names('区域巡防'), '区域巡防', border_gap=35, allow_corner_gap=True)
        return selected == self.plan.mission_type_name

    def _charge(self) -> tuple[int, int] | None:
        """显式等级要求电量区只有一个完整读数，不合并多个候选。"""
        if self.plan.level == '默认等级':
            return super()._charge()
        area = read_preset_area(self.ctx, self.last_screenshot, '手柄-电量')
        if area is None:
            return None
        values: list[tuple[int, int]] = []
        for text, matches in area[1].items():
            match = re.fullmatch(r'\s*(\d{1,4})\s*/\s*(\d{1,3})\s*', text)
            if match is not None and len(matches) == 1:
                values.append((int(match[1]), int(match[2])))
            elif (text == '三' and len(matches) == 1 and matches.max is not None
                  and matches.max.rect.x1 >= 105):
                # 已观察的右侧 Start 图标被 OCR 读成“三”，不能作为电量。
                continue
            else:
                return None
        return values[0] if len(values) == 1 else None

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
        """巡防不编辑模拟室卡片，必须重新确认目标详情。"""
        if self._plan_changed():
            self._clear_level_proof()
            return self.round_fail('本场计划已变化，停止区域巡防')
        if self._deployment_visible():
            self._clear_level_proof()
            if not self._return_to_preset_sent:
                self.ctx.controller.btn_tap('east')
                self._return_to_preset_sent = True
            return self.round_retry('返回详情重新确认驱动盘副本', wait=1)
        if self._found('手柄-挑战等级') and self._found('手柄-下一步'):
            if self.plan.level != '默认等级' and not self._fresh_frame():
                self._clear_level_proof()
                return self.round_fail('区域巡防详情截图过期或无效，未进入编队')
            if not self._preset_selected():
                self._clear_level_proof()
                self.save_screenshot(prefix='macos_patrol_wrong_target')
                return self.round_fail(f'当前详情并非计划副本{self.plan.mission_type_name}')
            self._level_confirmed = self._level_matches()
            if not self._level_confirmed:
                return self.round_fail('未确认要求的区域巡防Lv.60，未进入编队或出战')
            self._detail_frame_time = self.last_screenshot_time
            return self.round_success('驱动盘副本已确认')
        return self.round_retry('等待区域巡防详情', wait=1)

    @node_from(from_name='配置挑战卡片')
    @operation_node(name='准备当前编队', node_max_retry_times=10, timeout_seconds=180)
    def prepare_deployment(self) -> OperationRoundResult:
        """显式等级须用本场新鲜详情确认，编队只接续有限时间的证明。"""
        if self._plan_changed():
            self._clear_level_proof()
            return self.round_fail('本场计划已变化，停止区域巡防')
        if self.plan.level == '默认等级':
            return super().prepare_deployment()
        if not self._fresh_frame():
            self._clear_level_proof()
            return self.round_fail('区域巡防截图过期或无效，未允许出战')
        if self._deployment_visible():
            if not self._proof_valid() or self.last_screenshot_time <= self._next_frame_time:
                self._clear_level_proof()
                return self.round_fail('本场Lv.60详情证明缺失或过期，未允许出战')
            result = super().prepare_deployment()
            if result.is_success:
                if not self._fresh_frame() or not self._proof_valid():
                    self._clear_level_proof()
                    return self.round_fail('编队核验期间截图或等级证明过期，未允许出战')
                self._ready_frame_time = self.last_screenshot_time
            return result
        if not self._level_confirmed or self.last_screenshot_time <= self._detail_frame_time:
            self._clear_level_proof()
            return self.round_fail('未取得本场新的Lv.60详情帧，未进入编队')
        if self._next_sent and not self._proof_valid():
            self._clear_level_proof()
            return self.round_fail('本场Lv.60详情证明缺失或过期，未进入编队')
        if not (self._found('手柄-挑战等级') and self._found('手柄-下一步')
                and self._preset_selected() and self._level_matches()):
            self._clear_level_proof()
            return self.round_fail('区域巡防目标或Lv.60未知、已变化，未进入编队')
        charge = self._charge()
        if charge is None:
            return self.round_retry('未能确认实际电量消耗', wait=1)
        current, required = charge
        if not self.bind_prepared_charge(required):
            self._clear_level_proof()
            return self.round_fail(f'当前预设消耗 {required} 电量，与目标 {self.expected_charge} 不一致')
        if current < required:
            return self.round_fail(self.STATUS_CHARGE_NOT_ENOUGH)
        if not self._fresh_frame() or (self._next_sent and not self._proof_valid()):
            self._clear_level_proof()
            return self.round_fail('区域巡防详情截图或等级证明过期，未进入编队')
        if not self._next_sent:
            self.ctx.controller.btn_tap('start')
            self._next_sent = True
            self._next_frame_time = self.last_screenshot_time
            self._proof_started = time.monotonic()
        return self.round_retry('等待出战准备页', wait=1)

    @node_from(from_name='准备当前编队', status=GamepadCombatSimulation.STATUS_DEPLOYMENT_READY)
    @operation_node(name='手柄出战', node_max_retry_times=10, timeout_seconds=15)
    def deploy(self) -> OperationRoundResult:
        """出战前保留本场详情证明，并重新核验编队画面和实际费用。"""
        if self._plan_changed():
            self._clear_level_proof()
            return self.round_fail('本场计划已变化，停止区域巡防')
        if self.plan.level == '默认等级':
            return super().deploy()
        if (not self._fresh_frame() or not self._proof_valid() or not self._ready_frame_time
                or self.last_screenshot_time <= self._ready_frame_time):
            self._clear_level_proof()
            return self.round_fail('本场Lv.60证明或出战帧缺失、过期，未允许出战')
        if not self._deployment_visible():
            return self.round_retry('等待出战准备页', wait=1)
        if self._deploy_sent:
            return super().deploy()
        charge = self._charge()
        if charge is None:
            return self.round_retry('未能确认出战电量', wait=1)
        current, required = charge
        if not self.bind_prepared_charge(required):
            self._clear_level_proof()
            return self.round_fail(f'出战消耗 {required} 电量，与目标 {self.expected_charge} 不一致')
        if current < required:
            return self.round_fail(self.STATUS_CHARGE_NOT_ENOUGH)
        if not self._fresh_frame() or not self._proof_valid():
            self._clear_level_proof()
            return self.round_fail('出战核验期间截图或等级证明过期，未允许出战')
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
