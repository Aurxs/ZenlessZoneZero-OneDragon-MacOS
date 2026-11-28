"""原式舆防卫战的手柄导航与剧变节点选择。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import cv2_utils
from zzz_od.application.shiyu_defense.shiyu_defense_team_utils import (
    DefensePhaseTeamInfo,
    select_teams,
)
from zzz_od.game_data.agent import DmgTypeEnum
from zzz_od.operation.gamepad.focus import (
    find_yellow_frames,
    has_filled_highlight,
    highlight_mask,
    text_inside_frame,
)
from zzz_od.operation.gamepad.gamepad_operation import GamepadOperation
from zzz_od.operation.gamepad.predefined_team import (
    GamepadFocusPredefinedTeam,
    GamepadOpenPredefinedTeam,
)
from zzz_od.operation.gamepad.transport import find_yellow_focus_frames
from zzz_od.operation.gamepad.transport_by_compendium import (
    GamepadTransportByCompendium,
)
from zzz_od.operation.predefined_team_recognizer import recognize_predefined_team

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.geometry.rectangle import Rect
    from one_dragon.base.matcher.match_result import MatchResultList
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


SCREEN = '式舆防卫战'
GAMEPAD_SCREEN = '式舆防卫战-手柄'
ORDINALS = '一二三四五'


def read_critical_progress(words: dict[str, MatchResultList]) -> tuple[int, int] | None:
    """只接受剧变标签内独立的有效完成数和总数。"""
    compact = {''.join(text.split()) for text in words}
    if not any('剧变节点' in text for text in compact):
        return None
    ratios = {
        (int(match[1]), int(match[2]))
        for text in compact if (match := re.fullmatch(r'(?:剧变节点)?([0-5])[/／]([1-5])', text)) is not None
    }
    if len(ratios) != 1:
        return None
    current, total = ratios.pop()
    return (current, total) if current <= total else None


def focused_critical_node(image: MatLike, node_areas: list[Rect]) -> int | None:
    """圆形黄边需唯一，并完整落在一个已建档节点上。"""
    contours, _ = cv2.findContours(highlight_mask(image, 'yellow'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: list[tuple[int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if 148 <= width <= 200 and 148 <= height <= 200 and abs(width - height) < 14:
            if 0.69 <= cv2.contourArea(contour) / (width * height) <= 0.85:
                centers.append((x + width // 2, y + height // 2))
    if not centers and len(node_areas) == 5:
        # 五号的银色徽章遮住部分圆环，以外环四个象限的黄色覆盖识别活动态。
        area = node_areas[4]
        part = cv2_utils.crop_image_only(image, area)
        mask = highlight_mask(part, 'yellow') != 0
        rows, cols = np.indices(mask.shape)
        dx, dy = cols - area.width // 2, rows - area.height // 2
        radius = dx ** 2 + dy ** 2
        visible = mask & (radius >= 86 ** 2) & (radius <= 108 ** 2)
        if all(np.count_nonzero(visible & (sx * dx > 0) & (sy * dy > 0)) >= 220
               for sx in (-1, 1) for sy in (-1, 1)):
            return 5
    if len(centers) != 1:
        return None
    x, y = centers[0]
    nodes = [idx for idx, area in enumerate(node_areas, start=1) if area.x1 < x < area.x2 and area.y1 < y < area.y2]
    return nodes[0] if len(nodes) == 1 else None


class _ShiyuMenu(GamepadOperation):
    """复用已有裁图、等待和确认接口读取式舆菜单。"""

    def __init__(self, ctx: ZContext, op_name: str, timeout_seconds: float) -> None:
        super().__init__(ctx, op_name=op_name, timeout_seconds=timeout_seconds)
        self._nav: GamepadTransportByCompendium = GamepadTransportByCompendium(ctx, '作战', SCREEN, '剧变节点')

    def _found(self, name: str) -> bool:
        return self.last_screenshot is not None and self.round_by_find_area(
            self.last_screenshot, GAMEPAD_SCREEN, name,
        ).is_success

    def _read(self, area_name: str) -> tuple[MatLike, dict[str, MatchResultList]] | None:
        self._nav.last_screenshot = self.last_screenshot
        return self._nav._read_area(GAMEPAD_SCREEN, area_name)

    def read_attributes(self, area_name: str, label: str) -> list[DmgTypeEnum] | None:
        """按同一队的区域读取属性或明确空位。"""
        area = self._read(area_name)
        if area is None:
            return None
        normalized = {''.join(text.split()).upper() for text in area[1]}
        if label not in normalized:
            return None
        values = [value for value in DmgTypeEnum if value != DmgTypeEnum.UNKNOWN and value.value in normalized]
        if not values and not any('EMPTY' in text for text in normalized):
            return None
        return values


class GamepadShiyuTransport(_ShiyuMenu):
    """从快捷手册的作战分类进入剧变节点。"""

    COMBAT_CATEGORIES: ClassVar[tuple[str, ...]] = (
        '周期征讨', '式舆防卫战', '危局强袭战', '临界推演', '拟真鏖战试炼', '零号空洞',
    )

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄前往式舆防卫战', timeout_seconds=90)
        self._entered_list: bool = False
        self._requested: bool = False

    def handle_init(self) -> None:
        """每次运行重新确认手册及传送状态。"""
        self._nav.handle_init()
        self._entered_list = False
        self._requested = False

    @operation_node(name='打开手册', is_start_node=True, node_max_retry_times=7)
    def open_handbook(self) -> OperationRoundResult:
        """已有式舆入口时保留当地画面。"""
        if self._found('手柄-前哨档案') or self._found('手柄-前次记录'):
            return self.round_success('已在式舆入口')
        self._nav.last_screenshot = self.last_screenshot
        return self._nav.open_compendium()

    @node_from(from_name='打开手册')
    @operation_node(name='切到作战', node_max_retry_times=8)
    def choose_combat(self) -> OperationRoundResult:
        """由当前高亮手册标签决定肩键方向。"""
        nav = self._nav
        nav.last_screenshot = self.last_screenshot
        current = nav._selected('手柄-顶部标签', nav.TABS, 'yellow')
        if current == '作战':
            return self.round_success()
        if current is not None:
            shoulder = 'right_shoulder' if nav.TABS.index(current) < 3 else 'left_shoulder'
            self.ctx.controller.btn_tap(shoulder)
        return self.round_retry('等待作战标签高亮', wait=0.8)

    @node_from(from_name='切到作战')
    @operation_node(name='切到式舆分类', node_max_retry_times=8)
    def choose_category(self) -> OperationRoundResult:
        """根据当前蓝色分类走到式舆防卫战。"""
        nav = self._nav
        nav.last_screenshot = self.last_screenshot
        current = nav._selected('手柄-分类列表', self.COMBAT_CATEGORIES, 'blue')
        if current == SCREEN:
            return self.round_success()
        if current is not None:
            key = 'dpad_down' if current == '周期征讨' else 'dpad_up'
            self.ctx.controller.btn_tap(key)
        return self.round_retry('等待式舆防卫战分类高亮', wait=0.8)

    @node_from(from_name='切到式舆分类')
    @operation_node(name='选择剧变入口', node_max_retry_times=14)
    def choose_mission(self) -> OperationRoundResult:
        """剧变标题与同卡黄色「前往」都明确后才发起传送。"""
        nav = self._nav
        nav.last_screenshot = self.last_screenshot
        if nav._teleport_dialog():
            if self._requested:
                return self.round_success('等待传送确认')
            return self.round_fail('传送弹窗来源未确认')
        if self._requested:
            if self._found('手柄-前哨档案') or self._found('手柄-前次记录'):
                return self.round_success('已在式舆入口')
            return self.round_retry('等待剧变节点传送', wait=1)
        area = nav._read_area('快捷手册', '手柄-副本列表')
        if area is None:
            return self.round_retry('等待式舆副本列表', wait=0.8)
        image, words = area
        target = nav._text_rect(words, '剧变节点')
        if target is None or nav._text_rect(words, '稳定节点') is None:
            return self.round_retry('等待完整剧变节点标题', wait=0.8)
        frames = find_yellow_frames(image)
        buttons = [
            match.rect for text, matches in words.items() if ''.join(text.split()) in ('前往', 'A前往')
            for match in matches if has_filled_highlight(image, match.rect, 'yellow')
        ]
        if (
            len(buttons) == 1
            and 60 < buttons[0].center.y - target.center.y < 154
            and target.center.x < buttons[0].center.x
        ):
            self.ctx.controller.btn_tap('south')
            self._requested = True
        elif len(frames) == 1 and not text_inside_frame(target, frames):
            if target.y2 < frames[0].y1:
                self.ctx.controller.btn_tap('dpad_up')
        elif not frames and not buttons and not self._entered_list:
            if nav._selected('手柄-分类列表', (SCREEN,), 'blue') == SCREEN:
                self.ctx.controller.btn_tap('south')
                self._entered_list = True
        return self.round_retry('等待剧变节点与前往焦点', wait=0.8)

    @node_from(from_name='选择剧变入口', status='等待传送确认')
    @operation_node(name='确认传送', node_max_retry_times=6)
    def confirm_teleport(self) -> OperationRoundResult:
        """沿用原传送弹窗的确认判据。"""
        self._nav.last_screenshot = self.last_screenshot
        return self._nav.confirm_teleport()

    @node_from(from_name='打开手册', status='已在式舆入口')
    @node_from(from_name='选择剧变入口', status='已在式舆入口')
    @node_from(from_name='确认传送')
    @operation_node(name='等待式舆入口', node_max_retry_times=22)
    def wait_for_entry(self) -> OperationRoundResult:
        """识别到式舆专属锚点才将选关交给父操作。"""
        if self._found('手柄-前哨档案') or self._found('手柄-前次记录'):
            self.save_screenshot(prefix='macos_shiyu_entry')
            return self.round_success('已进入式舆防卫战')
        if self.node_retry_times == 20:
            self.save_screenshot(prefix='macos_shiyu_entry_pending')
        return self.round_retry('等待式舆防卫战入口', wait=1)


class GamepadShiyuOpenNode(_ShiyuMenu):
    """从本期剧变进度定位下一防线，先建立选择状态再进入详情。"""

    STATUS_ALL_DONE: ClassVar[str] = '本期剧变节点已完成'
    STATUS_THREE_ROOMS: ClassVar[str] = '已进入第五防线三间'

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, op_name='手柄选择式舆剧变节点', timeout_seconds=55)
        self._target: int | None = None
        self._requested: bool = False
        self._last_focus: int | None = None

    def handle_init(self) -> None:
        """状态和焦点均由本次现场重新读取。"""
        self._target = None
        self._requested = False
        self._last_focus = None

    @operation_node(name='选择下一防线', is_start_node=True, node_max_retry_times=15)
    def open_node(self) -> OperationRoundResult:
        """进入后核对防线标题，避免把发送 A 当成选关成功。"""
        if self._requested:
            if self._target == 5:
                if self._found('手柄-本期最佳总分') and self._found('手柄-防线选择'):
                    self.save_screenshot(prefix='macos_shiyu_rooms')
                    return self.round_success(self.STATUS_THREE_ROOMS)
                return self._retry('等待第五防线房间列表')
            title = self._read('手柄-防线标题')
            if title is not None and self._target is not None:
                expected = f'剧变节点第{ORDINALS[self._target - 1]}防线'
                if any(''.join(text.split()) == expected for text in title[1]):
                    self.save_screenshot(prefix='macos_shiyu_node')
                    return self.round_success('已确认剧变防线', data={'node_idx': self._target})
            return self.round_retry('等待所选防线详情加载', wait=1)
        if not self._found('手柄-前哨档案') or not self._found('手柄-选关选择'):
            return self._retry('等待式舆选关画面')
        tab = self._read('手柄-剧变进度')
        if tab is None or (progress := read_critical_progress(tab[1])) is None:
            return self._retry('等待剧变节点当前进度')
        current, total = progress
        if total != 5 or not find_yellow_focus_frames(tab[0], 280, 305, 120, 145, min_edge_presence=0.65):
            return self._retry('等待已建档的五防线剧变标签')
        if current == total:
            return self.round_success(self.STATUS_ALL_DONE, data={'completed': current, 'total': total})
        self._target = current + 1
        nodes = self.ctx.screen_loader.get_area(GAMEPAD_SCREEN, '手柄-节点区域')
        if nodes is None:
            return self._retry('式舆节点区域尚未配置')
        areas = [self.ctx.screen_loader.get_area(GAMEPAD_SCREEN, f'手柄-防线{i}') for i in range(1, total + 1)]
        if any(area is None for area in areas):
            return self._retry('式舆节点槽位尚未配置')
        # 整张图的坐标用于将已识别的圆形焦点分配到固定节点槽位。
        if self.last_screenshot is None:
            return self._retry('等待手柄截图')
        screen = self.last_screenshot.copy()
        screen[:nodes.y1, :] = 0
        screen[nodes.y2:, :] = 0
        screen[:, :nodes.x1] = 0
        screen[:, nodes.x2:] = 0
        focus = focused_critical_node(screen, [area.rect for area in areas if area is not None])
        if focus is None:
            return self._retry('等待唯一防线焦点')
        if focus == self._target:
            self.save_screenshot(prefix='macos_shiyu_node_focused')
            self.ctx.controller.btn_tap('south')
            self._requested = True
        elif focus != self._target and focus != self._last_focus:
            self.ctx.controller.btn_tap('dpad_right' if focus < self._target else 'dpad_left')
        self._last_focus = focus
        return self._retry('等待防线焦点变化')

    def _retry(self, reason: str) -> OperationRoundResult:
        if self.node_retry_times == 12:
            self.save_screenshot(prefix='macos_shiyu_menu_pending')
        return self.round_retry(reason, wait=1)


class GamepadShiyuPrepareNode(_ShiyuMenu):
    """在普通节点核对双队弱点并应用互不冲突的原预备编队。"""

    def __init__(self, ctx: ZContext, node_idx: int) -> None:
        super().__init__(ctx, op_name='手柄配置式舆双队', timeout_seconds=410)
        self.node_idx: int = node_idx
        self._team_indices: list[int] = []
        self._visible_slot: int | None = None
        self._selected_phase: int = 0
        self._requested: bool = False
        self._confirmed: bool = False

    def handle_init(self) -> None:
        """每次重新识别属性和编队。"""
        self._team_indices = []
        self._visible_slot = None
        self._selected_phase = 0
        self._requested = False
        self._confirmed = False

    @operation_node(name='核对防线并计算配队', is_start_node=True, node_max_retry_times=8)
    def plan_teams(self) -> OperationRoundResult:
        """沿用原式舆属性解析和互斥评分，成员稍后和游戏核对。"""
        if not 1 <= self.node_idx < 5:
            return self.round_fail('普通防线仅支持第一至第四防线')
        title = self._read('手柄-防线标题')
        if title is None or self.last_screenshot is None:
            return self.round_retry('等待剧变防线详情', wait=0.8)
        expected = f'剧变节点第{ORDINALS[self.node_idx - 1]}防线'
        if not any(''.join(text.split()) == expected for text in title[1]):
            return self.round_retry('等待预期防线标题', wait=0.8)
        targets: list[DefensePhaseTeamInfo] = []
        for phase in (1, 2):
            weakness = self.read_attributes(f'手柄-第{phase}队弱点', '推荐属性')
            resistance = self.read_attributes(f'手柄-第{phase}队抗性', '强敌抗性')
            if not weakness or resistance is None:
                break
            targets.append(DefensePhaseTeamInfo(weakness, resistance))
        if len(targets) != 2:
            return self.round_retry('等待两支队伍的推荐属性', wait=0.8)
        candidates = [
            team.idx for team in self.ctx.team_config.team_list
            if len(set(team.agent_id_list)) == 3 and 'unknown' not in team.agent_id_list
        ]
        teams = select_teams(self.ctx, targets, candidates)
        if len(teams) != 2:
            return self.round_fail('需要两支代理人互不重复的三人预备编队，请同步当前账号编队')
        self._team_indices = [team.team_idx for team in teams]
        return self.round_success()

    @node_from(from_name='核对防线并计算配队')
    @operation_node(name='进入双队出战准备', node_max_retry_times=12)
    def open_prepare(self) -> OperationRoundResult:
        """只在防线详情确认后打开队伍页。"""
        if self._deployment_visible():
            return self.round_success()
        if not self._requested and self._found('手柄-防线选择'):
            self.ctx.controller.btn_tap('south')
            self._requested = True
        return self.round_retry('等待双队准备页', wait=0.8)

    def _deployment_visible(self) -> bool:
        return self.last_screenshot is not None and all(
            self.round_by_find_area(self.last_screenshot, '实战模拟室', name).is_success
            for name in ('手柄-出战', '手柄-预备编队')
        )

    @node_from(from_name='进入双队出战准备')
    @operation_node(name='打开式舆预备编队', screenshot_before_round=False)
    def open_teams(self) -> OperationRoundResult:
        """原出战准备页的 X 预备编队入口。"""
        return self.round_by_op_result(GamepadOpenPredefinedTeam(self.ctx).execute())

    @node_from(from_name='打开式舆预备编队')
    @node_from(from_name='核对并选中编队', status='继续第二队')
    @operation_node(name='定位式舆预备编队', screenshot_before_round=False)
    def locate_team(self) -> OperationRoundResult:
        """沿用已有二十编队分页和当前焦点检查。"""
        result = _ShiyuFocusPredefinedTeam(self.ctx, self._team_indices[self._selected_phase]).execute()
        if result.success:
            self._visible_slot = result.data['visible_slot']
            self._requested = False
        return self.round_by_op_result(result)

    @node_from(from_name='定位式舆预备编队')
    @operation_node(name='核对并选中编队', node_max_retry_times=12)
    def choose_team(self) -> OperationRoundResult:
        """按牌面 TEAM 序号逐队确认；选错队时保持在列表。"""
        if self.last_screenshot is None or self._visible_slot is None:
            return self.round_retry('等待式舆预备编队截图', wait=0.8)
        area = self.ctx.screen_loader.get_area('编队选择', f'编队槽位{self._visible_slot}')
        if area is None:
            return self.round_fail('编队槽位区域缺失')
        words = self.ctx.ocr.run_ocr(self.last_screenshot)
        actual = recognize_predefined_team(self.ctx, self.last_screenshot, words, area.rect)
        team = self.ctx.team_config.get_team_by_idx(self._team_indices[self._selected_phase])
        members = [match.data.agent_id for match in actual.agent_match_result_list]
        if team is None or members != team.agent_id_list:
            return self.round_fail('游戏编队成员与原配置不一致，请同步当前账号编队')
        slot_words = words
        selected = read_team_order(slot_words, area.rect)
        wanted = self._selected_phase + 1
        if selected == wanted:
            self._selected_phase += 1
            return self.round_success('继续第二队' if wanted == 1 else '双队均已选中')
        if selected is not None:
            return self.round_fail('当前选中队伍顺序与式舆计划不符')
        if not self._requested:
            focus = find_focused_team(self.last_screenshot, words)
            unselected = any(
                ''.join(text.upper().split()) in ('+SELECT', 'SELECT')
                and any(area.x2 - 135 < match.center.x < area.x2 and area.y1 < match.center.y < area.y2 for match in matches)
                for text, matches in slot_words.items()
            )
            if unselected and focus is not None and (
                area.x1 < focus.center.x < area.x2 and area.y1 < focus.center.y < area.y2
            ):
                self.ctx.controller.btn_tap('south')
                self._requested = True
        return self.round_retry('等待式舆预备编队序号', wait=0.8)

    @node_from(from_name='核对并选中编队', status='双队均已选中')
    @operation_node(name='确认双队预备出战', node_max_retry_times=12)
    def confirm_teams(self) -> OperationRoundResult:
        """预备出战只发送一次，等已选双队返回准备页。"""
        if self._confirmed and self._deployment_visible():
            if self._found('手柄-出战准备详情') and self._found('手柄-满员计数'):
                self.save_screenshot(prefix='macos_shiyu_teams_ready')
                return self.round_success('式舆双队已准备', data={'team_indices': self._team_indices})
        elif not self._confirmed and self._found('手柄-预备出战'):
            self.ctx.controller.btn_tap('start')
            self._confirmed = True
        return self.round_retry('等待核对双队出战准备', wait=1)


def read_team_order(words: dict[str, MatchResultList], card: Rect) -> int | None:
    """只读卡片右侧的 TEAM 序号，排除同屏其他队伍和代理人数字。"""
    badge_x = card.x2 - card.width // 5
    markers = [
        (''.join(text.upper().split()), match.center)
        for text, matches in words.items() for match in matches
        if badge_x < match.center.x < card.x2 and card.y1 + 55 < match.center.y < card.y2 - 23
    ]
    orders = {int(text[-1]) for text, _ in markers if text in ('TEAM01', 'TEAM02')}
    for text, center in markers:
        if text in ('01', '02') and any(
            marker == 'TEAM' and 8 < center.y - point.y < 81 and abs(center.x - point.x) < 38
            for marker, point in markers
        ):
            orders.add(int(text[-1]))
    return orders.pop() if len(orders) == 1 else None


def find_focused_team(image: MatLike, words: dict[str, MatchResultList]) -> Rect | None:
    """式舆可同时保留多个选中框，活动卡标题栏带 RS 或 Y。"""
    frames = find_yellow_focus_frames(image, 750, 850, 240, 300)
    if len(frames) == 1:
        return frames[0]
    hints = [
        point.center for text, matches in words.items() for point in matches
        if text.strip().upper() in ('RS', 'Y')
    ]
    focused = [
        frame for frame in frames
        if any(frame.x1 < hint.x < frame.x2 and frame.y1 < hint.y < frame.y1 + 66 for hint in hints)
    ]
    return focused[0] if len(focused) == 1 else None


class _ShiyuFocusPredefinedTeam(GamepadFocusPredefinedTeam):
    """复用通用分页。已选卡和焦点卡同时可见时按手柄提示区分。"""

    def find_current_focus(self, words: dict[str, MatchResultList]) -> Rect | None:
        if self.last_screenshot is None:
            return None
        return find_focused_team(self.last_screenshot, words)

    @operation_node(name='定位编队', is_start_node=True, node_max_retry_times=42)
    def focus_team(self) -> OperationRoundResult:
        """保留实际导航反馈，卡在未知焦点时保存整帧。"""
        result = super().focus_team()
        if not result.is_success and self.node_retry_times == 8:
            self.save_screenshot(prefix='macos_shiyu_team_focus_pending')
        return result


class GamepadShiyuDeploy(_ShiyuMenu):
    """已核对选队的父节点过渡到已识别的原生战斗。"""

    def __init__(self, ctx: ZContext, single_room: bool = False) -> None:
        super().__init__(ctx, op_name='手柄式舆出战', timeout_seconds=45)
        self._single_room: bool = single_room
        self._sent: bool = False

    def handle_init(self) -> None:
        """只跟踪本次出战。"""
        self._sent = False

    @operation_node(name='出战并确认进入战斗', is_start_node=True, node_max_retry_times=24)
    def deploy(self) -> OperationRoundResult:
        """只在当次已准备双队的界面发一次出战。"""
        from zzz_od.operation.gamepad.battle_assistant import is_native_battle

        if self._sent:
            if is_native_battle(self.ctx, self.last_screenshot):
                return self.round_success('已进入式舆战斗')
        elif self._found('手柄-出战准备详情') and self._found(
            '手柄-单队满员计数' if self._single_room else '手柄-满员计数'
        ):
            action = self._read('手柄-出战确认')
            if action is not None and any('出战' in text.replace(' ', '') for text in action[1]):
                self.ctx.controller.btn_tap('start')
                self._sent = True
        return self.round_retry('等待式舆出战或战斗画面', wait=1)


class GamepadShiyuExitResult(_ShiyuMenu):
    """从已核对的式舆结算返回，复核游戏内的进度变化。"""

    def __init__(self, ctx: ZContext, node_idx: int) -> None:
        super().__init__(ctx, op_name='手柄式舆结算返回', timeout_seconds=55)
        self.node_idx: int = node_idx
        self._sent: bool = False

    def handle_init(self) -> None:
        """只记录本次确认发送的退出动作。"""
        self._sent = False

    @operation_node(name='返回剧变选关并核对进度', is_start_node=True, node_max_retry_times=22)
    def close_result(self) -> OperationRoundResult:
        """比赛结束后只按结算页同一行明确标注的退出键。"""
        if self._sent and self._found('手柄-前哨档案'):
            tab = self._read('手柄-剧变进度')
            progress = read_critical_progress(tab[1]) if tab is not None else None
            if progress is not None and progress[0] >= self.node_idx:
                return self.round_success('已核对剧变节点完成记录', data={'completed': progress[0]})
            return self.round_retry('等待剧变节点新进度', wait=1)
        if not self._sent and self.last_screenshot is not None and self.round_by_find_area(
            self.last_screenshot, '战斗画面', '手柄-挑战结果',
        ).is_success:
            from zzz_od.operation.gamepad.shiyu_defense_battle import read_enemy_goal

            goals = self._read('手柄-结算目标')
            action = self._read('手柄-结果退出')
            if goals is not None and read_enemy_goal(goals[1]) is True and action is not None:
                key = _result_exit_button(action[1])
                if key is not None:
                    self.ctx.controller.resume_input()
                    self.ctx.controller.btn_tap(key)
                    self._sent = True
        if self.node_retry_times == 12:
            self.save_screenshot(prefix='macos_shiyu_result_exit_pending')
        return self.round_retry('等待式舆结算退出提示或选关画面', wait=1)


def _result_exit_button(words: dict[str, MatchResultList]) -> str | None:
    """退出文字必须由唯一手柄 A/B 图标标注，因此不会选择下一防线。"""
    keys: dict[str, str] = {'A': 'south', 'B': 'east'}
    entries = [(''.join(text.upper().split()), match.rect) for text, matches in words.items() for match in matches]
    result = {keys[word[0]] for word, _ in entries if word in ('A退出', 'B退出')}
    for label, text in entries:
        if label != '退出':
            continue
        result.update(
            keys[icon] for icon, rect in entries
            if icon in keys and abs(rect.center.y - text.center.y) < 21 and 0 < text.x1 - rect.x2 < 145
        )
    return result.pop() if len(result) == 1 else None
