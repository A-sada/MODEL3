"""Reinforcement-learning based routing support for VRPTW negotiation agents."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Optional dependency
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - torch might not be installed on all setups
    torch = None
    nn = None
    optim = None

from classes import Task


def _euclidean_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class PlannerConfig:
    max_tasks: int
    gamma: float = 0.99
    hidden_size: int = 128
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    replay_size: int = 50000
    batch_size: int = 64
    warmup: int = 512
    target_sync: int = 250
    wait_penalty: float = 0.1
    late_penalty: float = 5.0
    completion_bonus: float = 10.0
    infeasible_penalty: float = 25.0

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, float | int | None]) -> "PlannerConfig":
        if payload is None:
            raise ValueError("PlannerConfig payload is None and cannot be reconstructed")
        field_names = {field.name for field in fields(cls)}
        config_kwargs: Dict[str, float | int] = {}
        for key, value in payload.items():
            if key in field_names and value is not None:
                config_kwargs[key] = value
        missing = {name for name in field_names if name not in config_kwargs}
        if missing:
            raise ValueError(f"PlannerConfig missing keys in checkpoint: {sorted(missing)}")
        return cls(**config_kwargs)  # type: ignore[arg-type]


class VRPTWRoutingEnv:
    """Environment that scores routes for a single vehicle."""

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.state_dim = 7
        self.action_dim = 8
        self._tasks: List[Task] = []
        self._remaining: List[Task] = []
        self._feasible_indices: List[int] = []
        self.capacity: float = 0.0
        self.remaining_capacity: float = 0.0
        self.depot: Tuple[float, float] = (0.0, 0.0)
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_time: float = 0.0
        self._time_scale: float = 1.0
        self._distance_scale: float = 1.0
        self.route: List[Task] = []
        self.total_reward: float = 0.0
        self.total_distance: float = 0.0
        self.total_lateness: float = 0.0

    def reset(self, tasks: Sequence[Task], capacity: float, dep_x: float, dep_y: float) -> Tuple[np.ndarray, np.ndarray]:
        self._tasks = list(tasks)
        self._remaining = list(tasks)
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.depot = (dep_x, dep_y)
        self.current_x, self.current_y = dep_x, dep_y
        self.current_time = 0.0
        self.route = []
        self.total_reward = 0.0
        self.total_distance = 0.0
        self.total_lateness = 0.0
        if not self._tasks:
            self._time_scale = 1.0
            self._distance_scale = 1.0
        else:
            due_dates = [task.due_date for task in self._tasks]
            ready_times = [task.ready_time for task in self._tasks]
            self._time_scale = max(max(due_dates), max(ready_times), 1.0)
            xs = [task.x_coordinate for task in self._tasks]
            ys = [task.y_coordinate for task in self._tasks]
            xs.append(dep_x)
            ys.append(dep_y)
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            self._distance_scale = max(span, 1.0)
        return self._state_vector(), self._candidate_matrix()

    def step(self, candidate_index: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, float]]:
        if candidate_index >= len(self._feasible_indices):
            # Selected an infeasible action; give strong penalty and end episode
            penalty = -self.config.infeasible_penalty
            self.total_reward += penalty
            return self._state_vector(), np.zeros((0, self.action_dim), dtype=np.float32), penalty, True, {
                "total_distance": self.total_distance,
                "total_lateness": self.total_lateness,
            }
        task_idx = self._feasible_indices[candidate_index]
        task = self._remaining.pop(task_idx)
        travel_distance = _euclidean_xy(self.current_x, self.current_y, task.x_coordinate, task.y_coordinate)
        arrival_time = self.current_time + travel_distance
        wait_time = max(0.0, task.ready_time - arrival_time)
        start_service = arrival_time + wait_time
        finish_time = start_service + task.service_time
        lateness = max(0.0, finish_time - task.due_date)

        reward = -travel_distance - self.config.wait_penalty * wait_time - self.config.late_penalty * lateness
        self.total_distance += travel_distance
        self.total_lateness += lateness

        self.current_x = task.x_coordinate
        self.current_y = task.y_coordinate
        self.current_time = finish_time
        self.remaining_capacity -= task.weight
        self.route.append(task)

        if not self._remaining:
            reward += self.config.completion_bonus

        self.total_reward += reward
        next_candidates = self._candidate_matrix()
        done = len(next_candidates) == 0
        next_state = self._state_vector()
        info = {
            "total_distance": self.total_distance,
            "total_lateness": self.total_lateness,
        }
        return next_state, next_candidates, reward, done, info

    def _candidate_matrix(self) -> np.ndarray:
        self._feasible_indices = []
        features: List[List[float]] = []
        for idx, task in enumerate(self._remaining):
            if task.weight > self.remaining_capacity:
                continue
            travel_distance = _euclidean_xy(self.current_x, self.current_y, task.x_coordinate, task.y_coordinate)
            arrival_time = self.current_time + travel_distance
            start_service = max(arrival_time, task.ready_time)
            finish_time = start_service + task.service_time
            if finish_time > task.due_date and lateness_exceeds_window(finish_time, task.due_date):
                continue
            slack = task.due_date - finish_time
            span = task.due_date - task.ready_time
            features.append([
                task.weight / max(self.capacity, 1.0),
                task.ready_time / self._time_scale,
                task.due_date / self._time_scale,
                task.service_time / self._time_scale,
                travel_distance / self._distance_scale,
                wait_ratio(arrival_time, task.ready_time, self._time_scale),
                slack / max(self._time_scale, 1.0),
                span / max(self._time_scale, 1.0),
            ])
            self._feasible_indices.append(idx)
        if not features:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        return np.asarray(features, dtype=np.float32)

    def _state_vector(self) -> np.ndarray:
        remaining = len(self._remaining)
        capacity_ratio = self.remaining_capacity / max(self.capacity, 1.0)
        time_ratio = self.current_time / max(self._time_scale, 1.0)
        tasks_ratio = remaining / max(self.config.max_tasks, 1)
        if remaining:
            ready_times = np.array([t.ready_time for t in self._remaining], dtype=np.float32)
            due_dates = np.array([t.due_date for t in self._remaining], dtype=np.float32)
            weights = np.array([t.weight for t in self._remaining], dtype=np.float32)
            ready_mean = float(ready_times.mean() / max(self._time_scale, 1.0))
            due_mean = float(due_dates.mean() / max(self._time_scale, 1.0))
            weight_mean = float(weights.mean() / max(self.capacity, 1.0))
        else:
            ready_mean = 0.0
            due_mean = 0.0
            weight_mean = 0.0
        dist_home = _euclidean_xy(self.current_x, self.current_y, *self.depot) / max(self._distance_scale, 1.0)
        return np.asarray(
            [capacity_ratio, time_ratio, tasks_ratio, ready_mean, due_mean, weight_mean, dist_home], dtype=np.float32
        )

    def candidate_count(self) -> int:
        return len(self._feasible_indices)


def wait_ratio(arrival: float, ready: float, scale: float) -> float:
    if scale <= 0:
        scale = 1.0
    return max(0.0, ready - arrival) / scale


def lateness_exceeds_window(finish_time: float, due_date: float, tolerance: float = 0.0) -> bool:
    return finish_time - due_date > tolerance


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_candidates: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, next_candidates, done))

    def sample(self, batch_size: int):
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, next_candidates, done = zip(*batch)
        return (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            next_candidates,
            np.asarray(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


if torch is not None:

    class RouteValueNetwork(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_size: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.layers(x).squeeze(-1)


    class DQNRoutePlanner:
        """Deep-Q-learning agent that proposes routes for a vehicle."""

        def __init__(self, env: VRPTWRoutingEnv, config: Optional[PlannerConfig] = None):
            self.env = env
            self.config = config or env.config
            input_dim = env.state_dim + env.action_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = RouteValueNetwork(input_dim, self.config.hidden_size).to(self.device)
            self.target_net = RouteValueNetwork(input_dim, self.config.hidden_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
            self.replay_buffer = ReplayBuffer(self.config.replay_size)
            self.steps_done = 0
            self.epsilon = self.config.epsilon_start

        def select_action(
            self, state: np.ndarray, candidates: np.ndarray, exploration: bool = True
        ) -> Tuple[int, Optional[np.ndarray]]:
            if candidates.size == 0:
                return -1, None
            if exploration and random.random() < self.epsilon:
                index = random.randrange(len(candidates))
                return index, candidates[index]
            state_tensor = torch.from_numpy(state).float().to(self.device)
            candidates_tensor = torch.from_numpy(candidates).float().to(self.device)
            state_repeat = state_tensor.unsqueeze(0).repeat(candidates_tensor.size(0), 1)
            q_values = self.policy_net(torch.cat([state_repeat, candidates_tensor], dim=1))
            index = int(torch.argmax(q_values).item())
            return index, candidates[index]

        def train_episode(
            self,
            tasks: Sequence[Task],
            capacity: float,
            dep_x: float,
            dep_y: float,
        ) -> float:
            state, candidates = self.env.reset(tasks, capacity, dep_x, dep_y)
            done = False
            episode_reward = 0.0
            while not done and candidates.size:
                action_index, action_feat = self.select_action(state, candidates, exploration=True)
                next_state, next_candidates, reward, done, _ = self.env.step(action_index)
                if action_feat is not None:
                    self.replay_buffer.push(state, action_feat, reward, next_state, next_candidates, done)
                state = next_state
                candidates = next_candidates
                episode_reward += reward
                self.steps_done += 1
                if len(self.replay_buffer) >= self.config.warmup:
                    self._learn_step()
                if self.steps_done % self.config.target_sync == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            self._decay_epsilon()
            task_count = max(len(tasks), 1)
            return episode_reward / float(task_count)

        def _learn_step(self) -> None:
            if len(self.replay_buffer) < self.config.batch_size:
                return
            state_batch, action_batch, reward_batch, next_state_batch, next_candidates_batch, done_batch = (
                self.replay_buffer.sample(self.config.batch_size)
            )
            state_tensor = torch.from_numpy(state_batch).float().to(self.device)
            action_tensor = torch.from_numpy(action_batch).float().to(self.device)
            reward_tensor = torch.from_numpy(reward_batch).float().to(self.device)
            next_state_tensor = torch.from_numpy(next_state_batch).float().to(self.device)
            done_tensor = torch.from_numpy(done_batch).float().to(self.device)

            inputs = torch.cat([state_tensor, action_tensor], dim=1)
            current_q = self.policy_net(inputs)

            target_q_list: List[torch.Tensor] = []
            for i, candidates in enumerate(next_candidates_batch):
                if len(candidates) == 0 or done_tensor[i].item() > 0.5:
                    target_q_list.append(torch.zeros(1, device=self.device))
                    continue
                candidate_tensor = torch.from_numpy(np.asarray(candidates, dtype=np.float32)).float().to(self.device)
                state_rep = next_state_tensor[i].unsqueeze(0).repeat(candidate_tensor.size(0), 1)
                q_values = self.target_net(torch.cat([state_rep, candidate_tensor], dim=1))
                target_q_list.append(q_values.max().unsqueeze(0))
            if target_q_list:
                max_next_q = torch.cat(target_q_list)
            else:
                max_next_q = torch.zeros(len(state_batch), device=self.device)
            expected_q = reward_tensor + self.config.gamma * (1.0 - done_tensor) * max_next_q

            loss = nn.functional.mse_loss(current_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
            self.optimizer.step()

        def _decay_epsilon(self) -> None:
            if self.epsilon > self.config.epsilon_min:
                self.epsilon *= self.config.epsilon_decay
                self.epsilon = max(self.epsilon, self.config.epsilon_min)

        def plan_route(
            self,
            tasks: Sequence[Task],
            capacity: float,
            dep_x: float,
            dep_y: float,
            greedy: bool = True,
        ) -> Tuple[List[Task], Dict[str, float]]:
            state, candidates = self.env.reset(tasks, capacity, dep_x, dep_y)
            done = False
            while not done and candidates.size:
                action_index, _ = self.select_action(state, candidates, exploration=not greedy)
                state, candidates, _, done, info = self.env.step(action_index)
            route = list(self.env.route)
            remaining_ids = {task.id for task in tasks} - {task.id for task in route}
            task_count = max(len(tasks), 1)
            averaged_reward = self.env.total_reward / float(task_count)
            info = {
                "total_reward": averaged_reward,
                "raw_total_reward": self.env.total_reward,
                "total_distance": self.env.total_distance,
                "total_lateness": self.env.total_lateness,
                "unassigned_tasks": len(remaining_ids),
            }
            if remaining_ids:
                leftovers = [task for task in tasks if task.id in remaining_ids]
                route.extend(leftovers)
            return route, info

        def save(self, path: str | Path) -> None:
            path = Path(path)
            torch.save(
                {
                    "policy_state_dict": self.policy_net.state_dict(),
                    "target_state_dict": self.target_net.state_dict(),
                    "epsilon": self.epsilon,
                    "steps_done": self.steps_done,
                    "config": self.config.to_dict(),
                },
                path,
            )

        def _apply_checkpoint(self, checkpoint: Dict[str, object]) -> None:
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])  # type: ignore[index]
            self.target_net.load_state_dict(checkpoint["target_state_dict"])  # type: ignore[index]
            self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_min))  # type: ignore[arg-type]
            self.steps_done = int(checkpoint.get("steps_done", 0))
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def load(self, path: str | Path) -> None:
            path = Path(path)
            checkpoint = torch.load(path, map_location=self.device)
            config_payload = checkpoint.get("config")
            if config_payload is not None:
                loaded_config = PlannerConfig.from_dict(config_payload)  # type: ignore[arg-type]
                if loaded_config.max_tasks != self.config.max_tasks:
                    raise ValueError(
                        "Checkpoint max_tasks does not match current planner configuration: "
                        f"{loaded_config.max_tasks} != {self.config.max_tasks}"
                    )
                self.config = loaded_config
                self.env.config = loaded_config
            self._apply_checkpoint(checkpoint)

        @classmethod
        def from_checkpoint(cls, path: Union[str, Path]) -> "DQNRoutePlanner":
            path = Path(path)
            checkpoint = torch.load(path, map_location="cpu")
            config_payload = checkpoint.get("config")
            if config_payload is None:
                raise ValueError("Checkpoint does not embed planner configuration; retrain with updated trainer.")
            config = PlannerConfig.from_dict(config_payload)  # type: ignore[arg-type]
            env = VRPTWRoutingEnv(config)
            planner = cls(env, config)
            planner._apply_checkpoint(checkpoint)
            return planner


else:

    class DQNRoutePlanner:
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, env: VRPTWRoutingEnv, config: Optional[PlannerConfig] = None):
            raise ImportError("PyTorch is required for DQNRoutePlanner. Install torch to enable RL routing.")

        @classmethod
        def from_checkpoint(cls, path: Union[str, Path]) -> "DQNRoutePlanner":
            raise ImportError("PyTorch is required to load pretrained planners. Install torch to enable RL routing.")


def build_default_planner(max_tasks: int) -> Tuple[VRPTWRoutingEnv, DQNRoutePlanner]:
    config = PlannerConfig(max_tasks=max_tasks)
    env = VRPTWRoutingEnv(config)
    planner = DQNRoutePlanner(env, config)
    return env, planner


def load_pretrained_planner(checkpoint_path: Union[str, Path]) -> Tuple[VRPTWRoutingEnv, DQNRoutePlanner]:
    if torch is None:
        raise ImportError("PyTorch is required to load pretrained planners. Install torch to enable RL routing.")
    planner = DQNRoutePlanner.from_checkpoint(checkpoint_path)
    return planner.env, planner
