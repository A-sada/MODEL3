"""Tabu search baseline planner for VRPTW route evaluation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from classes import Task
from rl_route_planner import PlannerConfig


def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class RouteStats:
    total_distance: float
    total_wait: float
    total_lateness: float
    raw_reward: float
    avg_reward: float
    end_time: float


@dataclass
class TabuSearchConfig:
    max_iterations: int = 200
    tabu_tenure: int = 20
    max_no_improve: int = 60
    candidate_pool_size: int = 60
    swap_depth: int = 5
    relocation_depth: int = 5
    seed: Optional[int] = None


@dataclass
class TabuSolution:
    route: List[Task]
    stats: RouteStats


class TabuRoutePlanner:
    """Heuristic planner based on a simple tabu search neighbourhood."""

    def __init__(self, planner_config: PlannerConfig, tabu_config: Optional[TabuSearchConfig] = None):
        self.config = planner_config
        self.tabu_config = tabu_config or TabuSearchConfig()
        self._rng = random.Random(self.tabu_config.seed)

    def plan_route(
        self,
        tasks: Sequence[Task],
        capacity: float,
        dep_x: float,
        dep_y: float,
        greedy: bool = True,  # noqa: ARG002 - kept for API parity with RL planner
    ) -> Tuple[List[Task], Dict[str, float]]:
        task_list = list(tasks)
        total_tasks = len(task_list)
        if total_tasks == 0:
            info = {
                "total_reward": 0.0,
                "raw_total_reward": 0.0,
                "total_distance": 0.0,
                "total_lateness": 0.0,
                "unassigned_tasks": 0,
            }
            return [], info

        depot = (float(dep_x), float(dep_y))
        initial = self._initial_solution(task_list, capacity, depot, total_tasks)
        best = initial
        current = initial
        tabu: Dict[Tuple[str, int, int], int] = {}
        no_improve = 0

        for iteration in range(self.tabu_config.max_iterations):
            self._purge_tabu(tabu, iteration)
            neighbours = self._generate_neighbours(current, task_list, capacity, depot, total_tasks)
            if not neighbours:
                break

            candidate: Optional[TabuSolution] = None
            candidate_move: Optional[Tuple[str, int, int]] = None

            for solution, move in neighbours:
                if solution.stats is None:
                    continue
                is_tabu = move in tabu and iteration < tabu[move]
                improves_best = solution.stats.raw_reward > best.stats.raw_reward + 1e-6
                if is_tabu and not improves_best:
                    continue
                if candidate is None or self._better(solution, candidate, total_tasks):
                    candidate = solution
                    candidate_move = move

            if candidate is None or candidate_move is None:
                break

            current = candidate
            tabu[candidate_move] = iteration + self.tabu_config.tabu_tenure

            if current.stats.raw_reward > best.stats.raw_reward + 1e-6 or (
                abs(current.stats.raw_reward - best.stats.raw_reward) <= 1e-6
                and len(current.route) > len(best.route)
            ):
                best = current
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.tabu_config.max_no_improve:
                break

        best_route = list(best.route)
        assigned_ids = {task.id for task in best_route}
        leftovers = [task for task in task_list if task.id not in assigned_ids]
        info = {
            "total_reward": best.stats.avg_reward,
            "raw_total_reward": best.stats.raw_reward,
            "total_distance": best.stats.total_distance,
            "total_lateness": best.stats.total_lateness,
            "unassigned_tasks": len(leftovers),
        }
        best_route.extend(leftovers)
        return best_route, info

    def _better(self, lhs: TabuSolution, rhs: TabuSolution, total_tasks: int) -> bool:
        if lhs.stats.raw_reward > rhs.stats.raw_reward + 1e-6:
            return True
        if rhs.stats.raw_reward > lhs.stats.raw_reward + 1e-6:
            return False
        lhs_unassigned = total_tasks - len(lhs.route)
        rhs_unassigned = total_tasks - len(rhs.route)
        if lhs_unassigned < rhs_unassigned:
            return True
        if rhs_unassigned < lhs_unassigned:
            return False
        return lhs.stats.total_distance < rhs.stats.total_distance - 1e-6

    def _initial_solution(
        self,
        tasks: Sequence[Task],
        capacity: float,
        depot: Tuple[float, float],
        total_tasks: int,
    ) -> TabuSolution:
        route: List[Task] = []
        stats = RouteStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for task in sorted(tasks, key=lambda t: (t.due_date, t.ready_time)):
            best_pos, candidate_stats = self._best_insertion(route, task, capacity, depot, total_tasks)
            if best_pos is not None and candidate_stats is not None:
                route.insert(best_pos, task)
                stats = candidate_stats
        if route:
            stats = self._evaluate_route(route, capacity, depot, total_tasks) or stats
        return TabuSolution(route, stats)

    def _best_insertion(
        self,
        route: Sequence[Task],
        task: Task,
        capacity: float,
        depot: Tuple[float, float],
        total_tasks: int,
    ) -> Tuple[Optional[int], Optional[RouteStats]]:
        best_pos: Optional[int] = None
        best_stats: Optional[RouteStats] = None
        trial = list(route)
        for pos in range(len(route) + 1):
            trial.insert(pos, task)
            stats = self._evaluate_route(trial, capacity, depot, total_tasks)
            trial.pop(pos)
            if stats is None:
                continue
            if best_stats is None or stats.raw_reward > best_stats.raw_reward + 1e-6 or (
                abs(stats.raw_reward - best_stats.raw_reward) <= 1e-6 and stats.total_distance < best_stats.total_distance
            ):
                best_stats = stats
                best_pos = pos
        return best_pos, best_stats

    def _generate_neighbours(
        self,
        solution: TabuSolution,
        tasks: Sequence[Task],
        capacity: float,
        depot: Tuple[float, float],
        total_tasks: int,
    ) -> List[Tuple[TabuSolution, Tuple[str, int, int]]]:
        neighbours: List[Tuple[TabuSolution, Tuple[str, int, int]]] = []
        route = solution.route
        route_len = len(route)
        if route_len == 0:
            return neighbours

        limit = self.tabu_config.candidate_pool_size
        rng = self._rng
        indices = list(range(route_len))

        # Swap moves within limited depth from each position
        for i in indices:
            max_j = min(route_len, i + 1 + max(1, self.tabu_config.swap_depth))
            for j in range(i + 1, max_j):
                candidate_route = route[:]
                candidate_route[i], candidate_route[j] = candidate_route[j], candidate_route[i]
                stats = self._evaluate_route(candidate_route, capacity, depot, total_tasks)
                if stats is None:
                    continue
                move = ("swap", min(route[i].id, route[j].id), max(route[i].id, route[j].id))
                neighbours.append((TabuSolution(candidate_route, stats), move))
                if len(neighbours) >= limit:
                    return neighbours

        # Relocation moves
        for i in indices:
            for delta in range(-self.tabu_config.relocation_depth, self.tabu_config.relocation_depth + 1):
                if delta == 0:
                    continue
                target = i + delta
                if target < 0:
                    target = 0
                if target > route_len:
                    target = route_len
                if target == i:
                    continue
                candidate_route = route[:]
                task = candidate_route.pop(i)
                if target > i:
                    target -= 1
                candidate_route.insert(target, task)
                stats = self._evaluate_route(candidate_route, capacity, depot, total_tasks)
                if stats is None:
                    continue
                move = ("relocate", task.id, target)
                neighbours.append((TabuSolution(candidate_route, stats), move))
                if len(neighbours) >= limit:
                    return neighbours

        # Insertion of currently unassigned tasks
        assigned_ids = {task.id for task in route}
        unassigned = [task for task in tasks if task.id not in assigned_ids]
        rng.shuffle(unassigned)
        for candidate_task in unassigned:
            for pos in range(route_len + 1):
                candidate_route = route[:]
                candidate_route.insert(pos, candidate_task)
                stats = self._evaluate_route(candidate_route, capacity, depot, total_tasks)
                if stats is None:
                    continue
                move = ("insert", candidate_task.id, pos)
                neighbours.append((TabuSolution(candidate_route, stats), move))
                if len(neighbours) >= limit:
                    return neighbours

        return neighbours

    def _purge_tabu(self, tabu: Dict[Tuple[str, int, int], int], iteration: int) -> None:
        expired = [key for key, expiry in tabu.items() if expiry <= iteration]
        for key in expired:
            tabu.pop(key, None)

    def _evaluate_route(
        self,
        route: Sequence[Task],
        capacity: float,
        depot: Tuple[float, float],
        total_tasks: int,
    ) -> Optional[RouteStats]:
        if not route:
            return RouteStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_weight = sum(task.weight for task in route)
        if total_weight - capacity > 1e-6:
            return None

        current_x, current_y = depot
        current_time = 0.0
        total_distance = 0.0
        total_wait = 0.0
        total_lateness = 0.0

        for task in route:
            travel = _euclidean(current_x, current_y, task.x_coordinate, task.y_coordinate)
            arrival = current_time + travel
            wait_time = max(0.0, task.ready_time - arrival)
            start_service = arrival + wait_time
            finish = start_service + task.service_time
            if finish - task.due_date > 1e-6:
                return None
            total_distance += travel
            total_wait += wait_time
            total_lateness += max(0.0, finish - task.due_date)
            current_time = finish
            current_x = task.x_coordinate
            current_y = task.y_coordinate

        raw_reward = -total_distance
        raw_reward -= self.config.wait_penalty * total_wait
        raw_reward -= self.config.late_penalty * total_lateness
        if len(route) == total_tasks:
            raw_reward += self.config.completion_bonus
        avg_reward = raw_reward / float(max(total_tasks, 1))
        return RouteStats(total_distance, total_wait, total_lateness, raw_reward, avg_reward, current_time)
