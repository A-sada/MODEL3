from __future__ import annotations

from typing import Dict, List, Optional

from Vehicle import Vehicle_BASE
from Vehicle_NegotiatiorX import VehicleNegotiatorX
from balletin_are_search import calculate_dynamic_area
from classes import Agree, Offer, pac_task
from VRPTW_functions import (
    earliest_start_time_list,
    find_time_zone,
    find_vehicles_in_neighboring_areas,
    find_vehicle_by_id,
    latest_start_time_list,
)


class Vehicle(Vehicle_BASE):
    def __init__(self, id, max_weight, dep_x, dep_y):
        super().__init__(id, max_weight, dep_x, dep_y)

    def _task_value_scores(self, tasks: Optional[List] = None) -> Dict[int, float]:
        task_list = list(tasks) if tasks is not None else list(self.tasks)
        if not task_list:
            return {}
        scores = self.standardized_q_task_scores(task_list)
        if scores is None:
            return {task.id: 0.0 for task in task_list}
        return scores

    def _tasks_sorted_by_value(self) -> List:
        scores = self._task_value_scores()
        return sorted(self.tasks, key=lambda task: scores.get(task.id, 0.0))

    def _task_value_rank_ratio(self, task_id: int, scores: Dict[int, float]) -> float:
        if task_id not in scores:
            return 1.0
        sorted_scores = sorted(scores.values(), reverse=True)
        if not sorted_scores:
            return 1.0
        target_score = scores[task_id]
        rank = 1 + sum(1 for score in sorted_scores if score > target_score)
        return rank / float(len(sorted_scores))

    def offer_on_negotiation(self, run_cars, offer_id, vehicles):
        ordered_tasks = self._tasks_sorted_by_value()
        for task in ordered_tasks:
            vehicles_in_neighbors = []
            task_area = calculate_dynamic_area(
                task.x_coordinate,
                task.y_coordinate,
                self.bulletin_board.X,
                self.bulletin_board.n,
            )
            task_time_zone = find_time_zone(task.ready_time, self.bulletin_board.zones)
            vehicles_in_neighbors = find_vehicles_in_neighboring_areas(
                task_time_zone, task_area, self.bulletin_board.area_board
            )
            vehicles_in_neighbors.extend(
                self.find_available_vehicles(
                    self.bulletin_board.time_board, task.ready_time, task.due_date, vehicles
                )
            )
            if self.id in vehicles_in_neighbors:
                vehicles_in_neighbors.remove(self.id)

            for vehicle in vehicles_in_neighbors:
                car = find_vehicle_by_id(vehicle, run_cars)
                if car is not None:
                    offer = Offer(offer_id, self.id, car.id, task)
                    offer_id += 1
                    self.offer_nego_list.append(offer)
        return self.offer_nego_list

    def find_available_vehicles(self, b_board, task_ready_time, task_due_date, vehicles):
        available_vehicles = []
        for _, row in b_board.iterrows():
            if row["departure_time"] > task_due_date or row["return_time"] < task_ready_time:
                available_vehicles.append(row["id"])
        return [vehicle for vehicle in vehicles if vehicle.id in available_vehicles]

    def before_negotiation(self):
        self.over_task = self._tasks_sorted_by_value()

    def sign_contracts(self, agreements: List[Agree]):
        signed = []
        for agreement in agreements:
            task_a = agreement.taskA
            candidate_tasks = list(self.tasks)
            if task_a not in candidate_tasks:
                candidate_tasks.append(task_a)
            scores = self._task_value_scores(candidate_tasks)
            if not scores:
                signed.append(agreement)
                continue
            rank_ratio = self._task_value_rank_ratio(task_a.id, scores)
            if rank_ratio <= 0.3:
                signed.append(agreement)
        return signed

    def first_step(self):
        if self.route_planner is not None and len(self.tasks) > 1:
            try:
                planned_tasks, _info = self.route_planner.plan_route(
                    self.tasks, self.max_weight, self.dep_x, self.dep_y
                )
            except Exception:
                planned_tasks = None
            if planned_tasks:
                planned_ids = {task.id for task in planned_tasks}
                remaining = [task for task in self.tasks if task.id not in planned_ids]
                self.tasks = planned_tasks + remaining
        self.arrival_time_list = []
        self.current_weight = 0
        for task in self.tasks:
            self.arrival_time_list.append(pac_task(task))
            self.current_weight += task.weight
        earliest_start_time_list(self.arrival_time_list, self.dep_x, self.dep_y)
        latest_start_time_list(self.arrival_time_list)

    def make_neg_agent(self, negotiation_id: Optional[int] = None, counterparty_id: Optional[int] = None):
        initial_can_propose = bool(self.offer_flag)
        self.Neg = VehicleNegotiatorX(
            self.id,
            self.tasks,
            is_vehicle_a=initial_can_propose,
            task_a=self.propose_task,
            route_planner=self.route_planner,
            dep_x=self.dep_x,
            dep_y=self.dep_y,
            max_weight=self.max_weight,
            q_importance_weight=self.q_importance_weight,
            negotiation_id=negotiation_id,
            log_path=self.negotiation_log_path,
            counterparty_id=counterparty_id,
            name=self.id,
            can_propose=True,
        )
        self.Neg.bulletin_board = self.bulletin_board
        self.before_negotiation()
        self.Neg.remove_list = self.over_task
        self.Neg.arrival_time_list = self.arrival_time_list
        return self.Neg
