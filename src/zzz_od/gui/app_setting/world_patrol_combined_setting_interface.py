from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from one_dragon_qt.widgets.segmented_setting_interface import SegmentedSettingInterface

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from zzz_od.context.zzz_context import ZContext


class WorldPatrolCombinedSettingInterface(SegmentedSettingInterface):

    def __init__(self, ctx: ZContext, parent: QWidget | None = None) -> None:
        from zzz_od.gui.view.world_patrol.world_patrol_route_list_interface import (
            WorldPatrolRouteListInterface,
        )
        from zzz_od.gui.view.world_patrol.world_patrol_setting_interface import (
            WorldPatrolSettingInterface,
        )

        pages = [WorldPatrolSettingInterface(ctx), WorldPatrolRouteListInterface(ctx)]
        # 录制页会操作镜头或直接执行路线，仅在有对应输入实现的 Windows 加载。
        if sys.platform == 'win32':
            from zzz_od.gui.view.world_patrol.world_patrol_large_map_recorder_interface import (
                LargeMapRecorderInterface,
            )
            from zzz_od.gui.view.world_patrol.world_patrol_route_recorder_interface import (
                WorldPatrolRouteRecorderInterface,
            )

            pages.extend([LargeMapRecorderInterface(ctx), WorldPatrolRouteRecorderInterface(ctx)])

        SegmentedSettingInterface.__init__(
            self,
            object_name='world_patrol_combined_setting',
            nav_text_cn='锄大地配置',
            sub_interfaces=pages,
            parent=parent,
        )
