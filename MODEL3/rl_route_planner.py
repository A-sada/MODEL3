"""Reinforcement-learning based routing support for VRPTW negotiation agents."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import MISSING, dataclass, asdict, fields
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

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


@dataclass
class GraphState:
    """Order-invariant vehicle state built from per-task graph features."""

    node_features: np.ndarray
    node_mask: np.ndarray
    visited_mask: np.ndarray
    feasible_mask: np.ndarray
    context: np.ndarray

    def feasible_indices(self) -> np.ndarray:
        """Return indices of tasks that are feasible to visit next."""

        return np.nonzero(self.feasible_mask)[0]


_NODE_STATIC_DIM = 7
_NODE_DYNAMIC_DIM = 15
NODE_FEATURE_DIM = _NODE_STATIC_DIM + _NODE_DYNAMIC_DIM
CONTEXT_FEATURE_DIM = 15


def _euclidean_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class PlannerConfig:
    max_tasks: int
    gamma: float = 0.99
    hidden_size: int = 128
    attention_heads: int = 4
    encoder_layers: int = 2
    attention_dropout: float = 0.1
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
        config_kwargs: Dict[str, Any] = {}
        for field_def in fields(cls):
            key = field_def.name
            if key in payload and payload[key] is not None:
                config_kwargs[key] = payload[key]
            elif field_def.default is not MISSING:
                config_kwargs[key] = field_def.default
            elif getattr(field_def, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                config_kwargs[key] = field_def.default_factory()  # type: ignore[call-arg]
            else:
                raise ValueError(f"PlannerConfig checkpoint missing required key: {key}")
        return cls(**config_kwargs)  # type: ignore[arg-type]


class VRPTWRoutingEnv:
    """Environment that scores routes for a single vehicle."""

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.state_dim = 7  # legacy interface
        self.action_dim = 8  # legacy interface
        self._tasks: List[Task] = []
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
        self._node_count: int = 0
        self.node_feature_dim: int = 0
        self.context_dim: int = 0
        self._node_static: Optional[np.ndarray] = None
        self._node_mask = np.zeros(self.config.max_tasks, dtype=np.float32)
        self._visited_mask = np.zeros(self.config.max_tasks, dtype=np.float32)
        self._feasible_mask = np.zeros(self.config.max_tasks, dtype=np.float32)
        self._feasible_order: List[int] = []
        self._cached_graph_state: Optional[GraphState] = None

    def reset(self, tasks: Sequence[Task], capacity: float, dep_x: float, dep_y: float) -> GraphState:
        self._tasks = list(tasks)
        self._node_count = len(self._tasks)
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.depot = (dep_x, dep_y)
        self.current_x, self.current_y = dep_x, dep_y
        self.current_time = 0.0
        self.route = []
        self.total_reward = 0.0
        self.total_distance = 0.0
        self.total_lateness = 0.0
        if self._node_count > self.config.max_tasks:
            raise ValueError(
                f"Number of tasks ({self._node_count}) exceeds planner max_tasks ({self.config.max_tasks})."
            )
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

        self._node_mask.fill(0.0)
        self._visited_mask.fill(0.0)
        self._feasible_mask.fill(0.0)
        self._feasible_order = []
        if self._tasks:
            self._node_mask[: self._node_count] = 1.0
        self._node_static = self._build_static_node_features()
        state = self._build_graph_state()
        self._cached_graph_state = state
        return state

    def step(self, node_index: int) -> Tuple[GraphState, float, bool, Dict[str, float]]:
        if node_index < 0 or node_index >= len(self._tasks) or self._feasible_mask[node_index] < 0.5:
            # Selected an infeasible action; give strong penalty and end episode
            penalty = -self.config.infeasible_penalty
            self.total_reward += penalty
            self._cached_graph_state = self._build_graph_state()
            return self._cached_graph_state, penalty, True, {
                "total_distance": self.total_distance,
                "total_lateness": self.total_lateness,
            }
        task = self._tasks[node_index]
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
        self._visited_mask[node_index] = 1.0

        if self._visited_mask.sum() >= len(self._tasks):
            reward += self.config.completion_bonus

        self.total_reward += reward
        next_state = self._build_graph_state()
        self._cached_graph_state = next_state
        done = not bool(self._feasible_order)
        info = {
            "total_distance": self.total_distance,
            "total_lateness": self.total_lateness,
        }
        return next_state, reward, done, info

    def _build_static_node_features(self) -> np.ndarray:
        static = np.zeros((self.config.max_tasks, _NODE_STATIC_DIM), dtype=np.float32)
        if not self._tasks:
            return static
        capacity_norm = max(self.capacity, 1.0)
        time_norm = max(self._time_scale, 1.0)
        distance_norm = max(self._distance_scale, 1.0)
        dep_x, dep_y = self.depot
        for idx, task in enumerate(self._tasks):
            norm_x = (task.x_coordinate - dep_x) / distance_norm
            norm_y = (task.y_coordinate - dep_y) / distance_norm
            demand_ratio = task.weight / capacity_norm
            ready_norm = task.ready_time / time_norm
            due_norm = task.due_date / time_norm
            service_norm = task.service_time / time_norm
            dist_depot = _euclidean_xy(task.x_coordinate, task.y_coordinate, dep_x, dep_y) / distance_norm
            static[idx] = np.asarray(
                [
                    norm_x,
                    norm_y,
                    demand_ratio,
                    ready_norm,
                    due_norm,
                    service_norm,
                    dist_depot,
                ],
                dtype=np.float32,
            )
        return static

    def _compute_dynamic_features(
        self, task: Task, visited: bool
    ) -> Tuple[np.ndarray, bool, float, float, float, float]:
        distance_norm = max(self._distance_scale, 1.0)
        time_norm = max(self._time_scale, 1.0)
        travel_distance = _euclidean_xy(self.current_x, self.current_y, task.x_coordinate, task.y_coordinate)
        arrival_time = self.current_time + travel_distance
        wait_time = max(0.0, task.ready_time - arrival_time)
        start_service = arrival_time + wait_time
        finish_time = start_service + task.service_time
        lateness = max(0.0, finish_time - task.due_date)
        slack = task.due_date - finish_time
        span = task.due_date - task.ready_time
        due_margin = task.due_date - self.current_time
        ready_margin = task.ready_time - self.current_time
        regret = task.due_date - (arrival_time + task.service_time)

        capacity_ok = 1.0 if task.weight <= self.remaining_capacity + 1e-6 else 0.0
        time_ok = 0.0 if lateness_exceeds_window(finish_time, task.due_date) else 1.0
        feasible_now = 1.0 if (not visited and capacity_ok > 0.5 and time_ok > 0.5) else 0.0

        if visited:
            capacity_ok = 0.0
            time_ok = 0.0
            feasible_now = 0.0

        dynamic = np.asarray(
            [
                1.0 if visited else 0.0,
                feasible_now,
                capacity_ok,
                time_ok,
                travel_distance / distance_norm,
                arrival_time / time_norm,
                wait_time / time_norm,
                start_service / time_norm,
                finish_time / time_norm,
                slack / time_norm,
                span / time_norm,
                lateness / time_norm,
                due_margin / time_norm,
                ready_margin / time_norm,
                regret / time_norm,
            ],
            dtype=np.float32,
        )

        return dynamic, bool(feasible_now), travel_distance / distance_norm, slack / time_norm, due_margin / time_norm, ready_margin / time_norm

    def _build_context_features(
        self,
        travel_vals: List[float],
        slack_vals: List[float],
        due_vals: List[float],
        ready_vals: List[float],
        feasible_mask: np.ndarray,
    ) -> np.ndarray:
        time_norm = max(self._time_scale, 1.0)
        distance_norm = max(self._distance_scale, 1.0)
        capacity_norm = max(self.capacity, 1.0)
        node_count = self._node_count
        remaining_count = node_count - int(self._visited_mask[:node_count].sum())
        visited_ratio = 0.0 if node_count == 0 else len(self.route) / float(node_count)
        remaining_ratio = 0.0 if node_count == 0 else remaining_count / float(node_count)
        feasible_ratio = 0.0 if node_count == 0 else float(feasible_mask[:node_count].sum()) / float(node_count)

        def _aggregate(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            arr = np.asarray(values, dtype=np.float32)
            return float(arr.min()), float(arr.mean())

        min_slack, mean_slack = _aggregate(slack_vals)
        min_due, mean_due = _aggregate(due_vals)
        min_ready, mean_ready = _aggregate(ready_vals)
        _, mean_travel = _aggregate(travel_vals)

        distance_to_depot = _euclidean_xy(self.current_x, self.current_y, *self.depot) / distance_norm

        context = np.asarray(
            [
                self.current_time / time_norm,
                self.remaining_capacity / capacity_norm,
                visited_ratio,
                remaining_ratio,
                distance_to_depot,
                self.total_distance / distance_norm,
                self.total_lateness / time_norm,
                mean_travel,
                min_slack,
            ],
            dtype=np.float32,
        )

        tail = np.asarray(
            [
                mean_slack,
                min_due,
                mean_due,
                min_ready,
                mean_ready,
                feasible_ratio,
            ],
            dtype=np.float32,
        )

        return np.concatenate([context, tail]).astype(np.float32, copy=False)

    def _build_graph_state(self) -> GraphState:
        node_features = np.zeros((self.config.max_tasks, NODE_FEATURE_DIM), dtype=np.float32)
        visited_mask = self._visited_mask.copy()
        feasible_mask = np.zeros(self.config.max_tasks, dtype=np.float32)
        travel_vals: List[float] = []
        slack_vals: List[float] = []
        due_vals: List[float] = []
        ready_vals: List[float] = []
        feasible_order: List[int] = []

        if not self._tasks:
            context = np.zeros(CONTEXT_FEATURE_DIM, dtype=np.float32)
            self.node_feature_dim = NODE_FEATURE_DIM
            self.context_dim = CONTEXT_FEATURE_DIM
            self._feasible_mask = feasible_mask
            self._feasible_order = feasible_order
            return GraphState(node_features, self._node_mask.copy(), visited_mask, feasible_mask.copy(), context)

        if self._node_static is None:
            self._node_static = self._build_static_node_features()

        for idx, task in enumerate(self._tasks):
            node_features[idx, :_NODE_STATIC_DIM] = self._node_static[idx]
            dynamic, feasible, travel_norm, slack_norm, due_norm, ready_norm = self._compute_dynamic_features(
                task, bool(visited_mask[idx])
            )
            node_features[idx, _NODE_STATIC_DIM:] = dynamic
            if visited_mask[idx] < 0.5:
                travel_vals.append(travel_norm)
                slack_vals.append(slack_norm)
                due_vals.append(due_norm)
                ready_vals.append(ready_norm)
            if feasible:
                feasible_mask[idx] = 1.0
                feasible_order.append(idx)

        context = self._build_context_features(travel_vals, slack_vals, due_vals, ready_vals, feasible_mask)
        if context.shape[0] != CONTEXT_FEATURE_DIM:
            padded_context = np.zeros(CONTEXT_FEATURE_DIM, dtype=np.float32)
            padded_context[: context.shape[0]] = context
            context = padded_context

        self.node_feature_dim = NODE_FEATURE_DIM
        self.context_dim = CONTEXT_FEATURE_DIM
        self._feasible_mask = feasible_mask
        self._feasible_order = feasible_order

        return GraphState(node_features, self._node_mask.copy(), visited_mask, feasible_mask.copy(), context)

    def _state_vector(self) -> np.ndarray:
        remaining_indices = np.where(self._visited_mask[: self._node_count] < 0.5)[0]
        remaining = int(remaining_indices.size)
        capacity_ratio = self.remaining_capacity / max(self.capacity, 1.0)
        time_ratio = self.current_time / max(self._time_scale, 1.0)
        tasks_ratio = remaining / max(self.config.max_tasks, 1)
        if remaining:
            ready_times = np.array([self._tasks[i].ready_time for i in remaining_indices], dtype=np.float32)
            due_dates = np.array([self._tasks[i].due_date for i in remaining_indices], dtype=np.float32)
            weights = np.array([self._tasks[i].weight for i in remaining_indices], dtype=np.float32)
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
        return int(self._feasible_mask[: self._node_count].sum())


def wait_ratio(arrival: float, ready: float, scale: float) -> float:
    if scale <= 0:
        scale = 1.0
    return max(0.0, ready - arrival) / scale


def lateness_exceeds_window(finish_time: float, due_date: float, tolerance: float = 0.0) -> bool:
    return finish_time - due_date > tolerance


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                int,
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                bool,
            ]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: GraphState,
        action_index: int,
        reward: float,
        next_state: GraphState,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                state.node_features.astype(np.float32, copy=True),
                state.node_mask.astype(np.float32, copy=True),
                state.context.astype(np.float32, copy=True),
                state.feasible_mask.astype(np.float32, copy=True),
                int(action_index),
                float(reward),
                next_state.node_features.astype(np.float32, copy=True),
                next_state.node_mask.astype(np.float32, copy=True),
                next_state.context.astype(np.float32, copy=True),
                next_state.feasible_mask.astype(np.float32, copy=True),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        (
            state_nodes,
            state_masks,
            state_context,
            state_feasible,
            action_indices,
            rewards,
            next_nodes,
            next_masks,
            next_context,
            next_feasible,
            done_flags,
        ) = zip(*batch)
        return (
            np.stack(state_nodes, axis=0),
            np.stack(state_masks, axis=0),
            np.stack(state_context, axis=0),
            np.stack(state_feasible, axis=0),
            np.asarray(action_indices, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_nodes, axis=0),
            np.stack(next_masks, axis=0),
            np.stack(next_context, axis=0),
            np.stack(next_feasible, axis=0),
            np.asarray(done_flags, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


if torch is not None:

    class EncoderBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, hidden_size: int, num_heads: int, dropout: float):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(hidden_size)
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(hidden_size)

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
            attn_out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            return x


    class SetEncoder(nn.Module):  # type: ignore[misc]
        def __init__(self, hidden_size: int, num_heads: int, num_layers: int, dropout: float):
            super().__init__()
            self.layers = nn.ModuleList(
                [EncoderBlock(hidden_size, num_heads, dropout) for _ in range(max(1, num_layers))]
            )

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
            for layer in self.layers:
                x = layer(x, padding_mask)
            return x


    class PoolingByMultiheadAttention(nn.Module):  # type: ignore[misc]
        def __init__(self, hidden_size: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
            super().__init__()
            self.seeds = nn.Parameter(torch.randn(num_seeds, hidden_size))
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
            batch_size = x.size(0)
            seeds = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)
            pooled, _ = self.attn(seeds, x, x, key_padding_mask=padding_mask)
            return self.norm(pooled)


    class GraphRouteValueNetwork(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            node_dim: int,
            context_dim: int,
            hidden_size: int,
            num_heads: int,
            num_encoder_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.node_proj = nn.Sequential(
                nn.Linear(node_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
            self.encoder = SetEncoder(hidden_size, num_heads, num_encoder_layers, dropout)
            self.pma = PoolingByMultiheadAttention(hidden_size, num_heads, num_seeds=1, dropout=dropout)
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.context_fuse = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(
            self,
            node_features: torch.Tensor,
            node_mask: torch.Tensor,
            context_features: torch.Tensor,
        ) -> torch.Tensor:  # type: ignore[override]
            padding_mask = node_mask < 0.5
            encoded = self.node_proj(node_features)
            encoded = self.encoder(encoded, padding_mask)
            pooled = self.pma(encoded, padding_mask).squeeze(1)
            context_emb = self.context_proj(context_features)
            fused_context = self.context_fuse(torch.cat([pooled, context_emb], dim=-1))
            context_expand = fused_context.unsqueeze(1).expand_as(encoded)
            scores = self.scorer(torch.cat([encoded, context_expand], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(padding_mask, float("-inf"))
            return scores

    class DQNRoutePlanner:
        """Deep-Q-learning agent that proposes routes for a vehicle."""

        def __init__(self, env: VRPTWRoutingEnv, config: Optional[PlannerConfig] = None):
            self.env = env
            self.config = config or env.config
            self.env.config = self.config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = GraphRouteValueNetwork(
                NODE_FEATURE_DIM,
                CONTEXT_FEATURE_DIM,
                self.config.hidden_size,
                self.config.attention_heads,
                self.config.encoder_layers,
                self.config.attention_dropout,
            ).to(self.device)
            self.target_net = GraphRouteValueNetwork(
                NODE_FEATURE_DIM,
                CONTEXT_FEATURE_DIM,
                self.config.hidden_size,
                self.config.attention_heads,
                self.config.encoder_layers,
                self.config.attention_dropout,
            ).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
            self.replay_buffer = ReplayBuffer(self.config.replay_size)
            self.steps_done = 0
            self.epsilon = self.config.epsilon_start

        def _state_to_tensors(
            self, state: GraphState
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            node_features = torch.from_numpy(state.node_features).unsqueeze(0).float().to(self.device)
            node_mask = torch.from_numpy(state.node_mask).unsqueeze(0).float().to(self.device)
            context = torch.from_numpy(state.context).unsqueeze(0).float().to(self.device)
            return node_features, node_mask, context

        def select_action(self, state: GraphState, exploration: bool = True) -> int:
            feasible_indices = state.feasible_indices()
            if feasible_indices.size == 0:
                return -1
            if exploration and random.random() < self.epsilon:
                return int(random.choice(feasible_indices.tolist()))
            node_features, node_mask, context = self._state_to_tensors(state)
            with torch.no_grad():
                q_values = self.policy_net(node_features, node_mask, context).squeeze(0)
            feasible_tensor = torch.from_numpy(feasible_indices).long().to(self.device)
            feasible_q = q_values.index_select(0, feasible_tensor)
            best_local = int(torch.argmax(feasible_q).item())
            return int(feasible_indices[best_local])

        def train_episode(
            self,
            tasks: Sequence[Task],
            capacity: float,
            dep_x: float,
            dep_y: float,
        ) -> float:
            state = self.env.reset(tasks, capacity, dep_x, dep_y)
            done = False
            episode_reward = 0.0
            while not done:
                feasible = state.feasible_indices()
                if feasible.size == 0:
                    break
                action_index = self.select_action(state, exploration=True)
                next_state, reward, done, _ = self.env.step(action_index)
                self.replay_buffer.push(state, action_index, reward, next_state, done)
                state = next_state
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
            (
                state_nodes,
                state_masks,
                state_context,
                _state_feasible,
                action_indices,
                reward_batch,
                next_nodes,
                next_masks,
                next_context,
                next_feasible,
                done_batch,
            ) = self.replay_buffer.sample(self.config.batch_size)

            state_nodes_t = torch.from_numpy(state_nodes).float().to(self.device)
            state_masks_t = torch.from_numpy(state_masks).float().to(self.device)
            state_context_t = torch.from_numpy(state_context).float().to(self.device)
            action_indices_t = torch.from_numpy(action_indices).long().unsqueeze(1).to(self.device)
            reward_t = torch.from_numpy(reward_batch).float().to(self.device)
            next_nodes_t = torch.from_numpy(next_nodes).float().to(self.device)
            next_masks_t = torch.from_numpy(next_masks).float().to(self.device)
            next_context_t = torch.from_numpy(next_context).float().to(self.device)
            next_feasible_t = torch.from_numpy(next_feasible).float().to(self.device)
            done_t = torch.from_numpy(done_batch).float().to(self.device)

            q_all = self.policy_net(state_nodes_t, state_masks_t, state_context_t)
            current_q = q_all.gather(1, action_indices_t).squeeze(1)

            next_q_all = self.target_net(next_nodes_t, next_masks_t, next_context_t)
            infeasible_mask = next_feasible_t < 0.5
            next_q_all = next_q_all.masked_fill(infeasible_mask, float("-inf"))
            max_next_q = next_q_all.max(dim=1).values
            max_next_q = torch.where(torch.isfinite(max_next_q), max_next_q, torch.zeros_like(max_next_q))
            expected_q = reward_t + self.config.gamma * (1.0 - done_t) * max_next_q

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
            state = self.env.reset(tasks, capacity, dep_x, dep_y)
            done = False
            info: Dict[str, float] = {}
            while not done:
                feasible = state.feasible_indices()
                if feasible.size == 0:
                    break
                action_index = self.select_action(state, exploration=not greedy)
                state, _, done, info = self.env.step(action_index)
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
