"""手柄每次只能走相邻格，短环时从已识别的有向连边改路。"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zzz_od.hollow_zero.hollow_map.hollow_zero_map import (
        HollowZeroMap,
        HollowZeroMapNode,
    )


def neighbor_direction(first: HollowZeroMapNode, second: HollowZeroMapNode) -> tuple[int, int] | None:
    """相邻格中心沿一个主要轴移动，忽略透视造成的小偏移。"""
    dx = second.pos.center.x - first.pos.center.x
    dy = second.pos.center.y - first.pos.center.y
    if abs(dx) > 2 * abs(dy):
        return (1 if dx > 0 else -1), 0
    if abs(dy) > 2 * abs(dx):
        return 0, (1 if dy > 0 else -1)
    return None


def _road_distances(
    board: HollowZeroMap, goal: int, blocked: set[int],
) -> dict[int, int]:
    """逆向遍历目标可达的格子，正向移动仍遵守轨道方向。"""
    incoming: dict[int, list[int]] = {}
    for origin, linked in board.edges.items():
        if origin in blocked:
            continue
        for target in linked:
            if target not in blocked:
                incoming.setdefault(target, []).append(origin)
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for origin in incoming.get(node, ()):
            if origin not in distances:
                distances[origin] = distances[node] + 1
                queue.append(origin)
    return distances


def route_around_loop(
    board: HollowZeroMap, goals: list[HollowZeroMapNode], tried: set[tuple[int, int]], avoid: set[str],
) -> tuple[HollowZeroMapNode, tuple[int, int], HollowZeroMapNode] | None:
    """只选实际通向目标且从当前状态没走过的相邻方向。"""
    start = board.current_idx
    if start is None:
        return None
    candidates: dict[int, tuple[int, int]] = {}
    for index in board.edges.get(start, ()):
        node = board.nodes[index]
        direction = neighbor_direction(board.nodes[start], node)
        if direction is not None and direction not in tried and node.entry.entry_name != '未知':
            candidates[index] = direction
    if not candidates:
        return None

    goal_indexes = [board.nodes.index(goal) for goal in goals if goal in board.nodes]
    hazards = {i for i, node in enumerate(board.nodes) if node.entry.entry_name in avoid}
    for goal in goal_indexes:
        if goal == start:
            continue
        for blocked in ((hazards - {goal}) | {start}, {start}):
            distances = _road_distances(board, goal, blocked)
            reachable = candidates.keys() & distances.keys()
            if reachable:
                chosen = min(reachable, key=lambda index: (distances[index], index))
                return board.nodes[chosen], candidates[chosen], board.nodes[goal]
    return None
