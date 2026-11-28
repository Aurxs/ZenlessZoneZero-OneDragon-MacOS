"""在剧变节点页领取当前已解锁的挑战奖励。"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from one_dragon.base.operation.operation_node import operation_node
from zzz_od.operation.gamepad.rewards import _GamepadRewardOperation

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


class GamepadClaimShiyuRewards(_GamepadRewardOperation):
    """从选关或奖励页开始，领取后复核并返回选关。"""

    SCREEN: ClassVar[str] = '式舆防卫战-剧变奖励'

    def __init__(self, ctx: ZContext) -> None:
        super().__init__(ctx, '手柄领取剧变节点奖励', timeout_seconds=55)
        self._opened: bool = False
        self._claim_sent: bool = False
        self._confirmed: bool = False
        self._closing: bool = False

    def handle_init(self) -> None:
        self.evidence_paths.clear()
        self._opened = self._claim_sent = self._confirmed = self._closing = False

    def _node_list(self) -> bool:
        return (self._has_text(self.SCREEN, '作战记录', '战报')
                and self._has_text(self.SCREEN, '剧变页签', '剧变节点')
                and self._has_text(self.SCREEN, '节点退出', '退出'))

    @operation_node(name='核对剧变奖励', is_start_node=True, node_max_retry_times=20)
    def claim(self) -> OperationRoundResult:
        """只有奖励页的「全部领取」已启用才提交，收到回执后才关闭。"""
        if self._claim_sent and self._known_reward_visible():
            if not self._confirmed:
                if not self._save_evidence('shiyu_obtained'):
                    return self.round_fail('未保存剧变奖励回执')
                self.ctx.controller.btn_tap('south')
                self._confirmed = True
            return self.round_retry('等待获得弹窗关闭', wait=0.8)

        if self._has_text(self.SCREEN, '奖励标题', '剧变节点奖励领取') and self._has_text(
            self.SCREEN, '防线列表', '第一防线'
        ):
            if self._closing:
                return self.round_retry('等待奖励页返回选关', wait=0.8)
            if self._yellow_text(self.SCREEN, '领取按钮', '全部领取'):
                if not self._claim_sent:
                    if not self._save_evidence('shiyu_before'):
                        return self.round_fail('未保存可领取的剧变奖励')
                    self.ctx.controller.btn_tap('west')
                    self._claim_sent = True
                return self.round_retry('等待剧变奖励领取结果', wait=1)
            if self._has_text(self.SCREEN, '奖励状态', '已领取') or self._confirmed:
                self._save_evidence('shiyu_finished')
                self.ctx.controller.btn_tap('east')
                self._closing = True
                return self.round_retry('等待返回剧变选关', wait=0.8)
        elif self._node_list():
            if self._closing:
                return self.round_success('剧变奖励已领取' if self._claim_sent else '剧变奖励已核对')
            if not self._opened and not self._claim_sent:
                self.ctx.controller.btn_tap('west')
                self._opened = True
        return self.round_retry('等待可确认的剧变节点或奖励画面', wait=0.8)
