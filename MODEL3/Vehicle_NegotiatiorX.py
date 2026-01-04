from __future__ import annotations

from typing import Optional

from negmas import ResponseType

from Vehicle_Negotiatior import VehicleNegotiator


class VehicleNegotiatorX(VehicleNegotiator):
    def _standardized_q_task_scores(self, tasks):
        if self.route_planner is None or tasks is None:
            return None
        task_list = list(tasks)
        if not task_list:
            return {}
        cache_key = ("standardized",) + self._importance_cache_key(task_list)
        cached = self._q_importance_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            scores = self.route_planner.standardized_discounted_q_task_scores(
                task_list, self.max_weight, self.dep_x, self.dep_y
            )
        except Exception:
            return None
        self._q_importance_cache[cache_key] = scores
        return scores

    def _task_value_scores(self, tasks):
        scores = self._standardized_q_task_scores(tasks)
        if scores is None:
            return None
        return scores

    def _task_value_rank_ratio(self, task_id: int, scores: dict[int, float]) -> float:
        if task_id not in scores:
            return 1.0
        sorted_scores = sorted(scores.values(), reverse=True)
        if not sorted_scores:
            return 1.0
        target_score = scores[task_id]
        rank = 1 + sum(1 for score in sorted_scores if score > target_score)
        return rank / float(len(sorted_scores))

    def _should_accept_task(self, task_id: int, scores: dict[int, float]) -> ResponseType:
        rank_ratio = self._task_value_rank_ratio(task_id, scores)
        if rank_ratio <= 0.3:
            return ResponseType.ACCEPT_OFFER
        if rank_ratio <= 0.7:
            return ResponseType.REJECT_OFFER
        return ResponseType.END_NEGOTIATION

    def make_remove_list(self, taskA):
        scores = self._task_value_scores(self.tasks)
        if scores is None:
            self.remove_list = [task for task in self.tasks]
            return
        ordered = sorted(self.tasks, key=lambda task: scores.get(task.id, 0.0))
        ordered.append("0")
        self.remove_list = ordered

    def respond(
        self,
        state,
        offer=None,
        source: Optional[str] = None,
    ):
        if offer is None and state is not None:
            offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if not self.initial_offer_received:
            self.initial_offer_received = offer

        task_a = offer.get("taskA") if offer else None
        if task_a is None:
            return ResponseType.REJECT_OFFER
        if task_a not in self.tasks and self.current_weight + task_a.weight > self.max_weight:
            return ResponseType.REJECT_OFFER

        candidate_tasks = list(self.tasks)
        if task_a not in candidate_tasks:
            candidate_tasks.append(task_a)
        scores = self._task_value_scores(candidate_tasks)
        if scores is None:
            return ResponseType.REJECT_OFFER
        return self._should_accept_task(task_a.id, scores)
