from __future__ import annotations

from typing import Optional

from rl_route_planner import PlannerConfig
from Strategy_Vehicle_verX import Vehicle as BaseVehicle
from tabu_route_planner import TabuRoutePlanner, TabuSearchConfig


class Vehicle(BaseVehicle):
    def __init__(self, id, max_weight, dep_x, dep_y):
        super().__init__(id, max_weight, dep_x, dep_y)
        self._tabu_config = TabuSearchConfig()
        self._tabu_planner: Optional[TabuRoutePlanner] = None
        self._tabu_max_tasks = 0

    def _tabu_planner_config(self, max_tasks: int) -> PlannerConfig:
        base_config = getattr(self.route_planner, "config", None)
        if isinstance(base_config, PlannerConfig):
            if base_config.max_tasks >= max_tasks:
                return base_config
            payload = base_config.to_dict()
            payload["max_tasks"] = max_tasks
            return PlannerConfig.from_dict(payload)
        return PlannerConfig(max_tasks=max_tasks)

    def _get_tabu_planner(self) -> TabuRoutePlanner:
        max_tasks = max(len(self.tasks), 1)
        if self._tabu_planner is None or self._tabu_max_tasks < max_tasks:
            config = self._tabu_planner_config(max_tasks)
            self._tabu_planner = TabuRoutePlanner(config, self._tabu_config)
            self._tabu_max_tasks = max_tasks
        return self._tabu_planner

    def _plan_route_with_tabu(self):
        if len(self.tasks) <= 1:
            return None
        planner = self._get_tabu_planner()
        try:
            planned_tasks, _info = planner.plan_route(
                self.tasks, self.max_weight, self.dep_x, self.dep_y
            )
        except Exception:
            return None
        return planned_tasks

    def first_step(self):
        planned_tasks = self._plan_route_with_tabu()
        if planned_tasks:
            planned_ids = {task.id for task in planned_tasks}
            remaining = [task for task in self.tasks if task.id not in planned_ids]
            self.tasks = planned_tasks + remaining
        self._refresh_arrival_time_list()

    def replan_route_with_planner(self):
        planned_tasks = self._plan_route_with_tabu()
        if not planned_tasks:
            return False
        planned_ids = {task.id for task in planned_tasks}
        remaining = [task for task in self.tasks if task.id not in planned_ids]
        self.tasks = planned_tasks + remaining
        self._refresh_arrival_time_list()
        return True
