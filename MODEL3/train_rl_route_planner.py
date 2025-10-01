from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Toggle this flag to switch between CLI arguments and the in-script configuration
USE_CLI = False

# Update these values when USE_CLI is False to control training/evaluation without the CLI
USER_RUN_CONFIG = {
    "dataset_root": SCRIPT_DIR / "Dateset" / "In",
    "instances": [],
    "instance_sample_size": 0,
    "use_all_instances": True,
    "train_samples": 200,
    "test_samples": 50,
    "subset_min": 3,
    "subset_max": 20,
    "train_episodes": 100000,
    "seed": 42,
    "log_interval": 50,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.995,
    "warmup": 256,
    "target_sync": 250,
    "batch_size": 64,
    "output_dir": None,
    "checkpoint": None,
    "no_train": False,
    "train_subsets": None,
    "test_subsets": SCRIPT_DIR.parent
    / "output_files"
    / "rl_planner"
    / "20250925-124137"
    / "test_subsets.json",
}

try:
    from classes import Task
    from rl_route_planner import DQNRoutePlanner, PlannerConfig, VRPTWRoutingEnv
except ImportError as exc:  # pragma: no cover - surfacing missing dependency early
    raise RuntimeError("Failed to import RL planner dependencies. Run from the repository root or adjust PYTHONPATH.") from exc

try:
    import torch  # noqa: F401  # imported to ensure availability for planner
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to train the RL route planner. Install torch before running this script.") from exc


def _clone_task(task: Task) -> Task:
    return Task(
        task.id,
        task.x_coordinate,
        task.y_coordinate,
        task.weight,
        task.ready_time,
        task.due_date,
        task.service_time,
    )


def _task_to_dict(task: Task) -> dict:
    return {
        "id": int(task.id),
        "x": float(task.x_coordinate),
        "y": float(task.y_coordinate),
        "weight": float(task.weight),
        "ready_time": float(task.ready_time),
        "due_date": float(task.due_date),
        "service_time": float(task.service_time),
    }


def _dict_to_task(data: dict) -> Task:
    return Task(
        data["id"],
        data["x"],
        data["y"],
        data["weight"],
        data["ready_time"],
        data["due_date"],
        data["service_time"],
    )


@dataclass
class InstanceData:
    name: str
    tasks: List[Task]
    capacity: float
    depot: Tuple[float, float]


@dataclass
class TaskSubset:
    instance: str
    subset_id: int
    tasks: List[Task]
    capacity: float
    depot: Tuple[float, float]

    def to_serializable(self) -> dict:
        return {
            "instance": self.instance,
            "subset_id": self.subset_id,
            "capacity": float(self.capacity),
            "depot": [float(self.depot[0]), float(self.depot[1])],
            "tasks": [_task_to_dict(task) for task in self.tasks],
        }

    def fresh_tasks(self) -> List[Task]:
        return [_clone_task(task) for task in self.tasks]


@dataclass
class EvaluationResult:
    subset_key: str
    num_tasks: int
    total_reward: float
    raw_total_reward: float
    total_distance: float
    total_lateness: float
    unassigned_tasks: int

    def to_serializable(self) -> dict:
        return {
            "subset": self.subset_key,
            "num_tasks": self.num_tasks,
            "total_reward": self.total_reward,
            "raw_total_reward": self.raw_total_reward,
            "total_distance": self.total_distance,
            "total_lateness": self.total_lateness,
            "unassigned_tasks": self.unassigned_tasks,
        }


def parse_vrptw_instance(file_path: Path, label: str | None = None) -> InstanceData:
    tasks: List[Task] = []
    capacity: float | None = None
    depot_x: float | None = None
    depot_y: float | None = None
    section: str | None = None

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("VEHICLE"):
                section = "VEHICLE"
                continue
            if upper.startswith("CUSTOMER"):
                section = "CUSTOMER"
                continue
            if section == "VEHICLE":
                if "NUMBER" in upper or "CAPACITY" in upper:
                    continue
                pieces = line.split()
                if len(pieces) >= 2:
                    capacity = float(pieces[-1])
                continue
            if section == "CUSTOMER":
                if upper.startswith("CUST"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                cust_id = int(float(parts[0]))
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])
                if cust_id == 0:
                    depot_x = x_coord
                    depot_y = y_coord
                    continue
                tasks.append(Task(cust_id, x_coord, y_coord, demand, ready_time, due_date, service_time))

    if capacity is None:
        raise ValueError(f"Failed to locate vehicle capacity in {file_path}")
    if depot_x is None or depot_y is None:
        raise ValueError(f"Failed to locate depot coordinates in {file_path}")

    display_name = label if label is not None else file_path.name

    return InstanceData(display_name, tasks, capacity, (depot_x, depot_y))


def _sample_task_subset(
    tasks: Sequence[Task],
    capacity: float,
    subset_min: int,
    subset_max: int,
    rng: random.Random,
    max_attempts: int = 500,
) -> List[Task]:
    subset_min = max(1, subset_min)
    subset_max = max(subset_min, subset_max)
    subset_max = min(subset_max, len(tasks))
    task_list = list(tasks)
    for _ in range(max_attempts):
        size = rng.randint(subset_min, subset_max)
        candidate = rng.sample(task_list, size)
        total_weight = sum(task.weight for task in candidate)
        if total_weight <= capacity:
            return candidate
    ordered = task_list[:]
    rng.shuffle(ordered)
    picked: List[Task] = []
    total_weight = 0.0
    for task in ordered:
        if len(picked) >= subset_max:
            break
        if total_weight + task.weight <= capacity:
            picked.append(task)
            total_weight += task.weight
        if len(picked) >= subset_min:
            break
    if len(picked) < subset_min:
        picked = ordered[:subset_min]
    return picked


def generate_task_subsets(
    instance: InstanceData,
    count: int,
    subset_min: int,
    subset_max: int,
    rng: random.Random,
    start_id: int = 0,
) -> List[TaskSubset]:
    subsets: List[TaskSubset] = []
    for idx in range(count):
        chosen = _sample_task_subset(instance.tasks, instance.capacity, subset_min, subset_max, rng)
        subsets.append(
            TaskSubset(
                instance=instance.name,
                subset_id=start_id + idx,
                tasks=[_clone_task(task) for task in chosen],
                capacity=instance.capacity,
                depot=instance.depot,
            )
        )
    return subsets


def ensure_output_dir(base_dir: Path | None) -> Path:
    if base_dir is None:
        root_dir = SCRIPT_DIR.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = root_dir / "output_files" / "rl_planner" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def write_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_task_subsets(path: Path) -> List[TaskSubset]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in subset file {path}")
    subsets: List[TaskSubset] = []
    for idx, item in enumerate(raw):
        instance = item.get("instance", "unknown")
        subset_id = int(item.get("subset_id", idx))
        capacity = float(item.get("capacity", 0.0))
        depot_values = item.get("depot", [0.0, 0.0])
        if len(depot_values) != 2:
            raise ValueError(f"Invalid depot specification in {path}: {depot_values}")
        depot = (float(depot_values[0]), float(depot_values[1]))
        task_dicts = item.get("tasks", [])
        tasks = [_dict_to_task(task_data) for task_data in task_dicts]
        subsets.append(
            TaskSubset(
                instance=instance,
                subset_id=subset_id,
                tasks=tasks,
                capacity=capacity,
                depot=depot,
            )
        )
    return subsets


def evaluate_subsets(planner: DQNRoutePlanner, subsets: Iterable[TaskSubset]) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for subset in subsets:
        tasks = subset.fresh_tasks()
        route, info = planner.plan_route(tasks, subset.capacity, subset.depot[0], subset.depot[1], greedy=True)
        key = f"{subset.instance}#subset{subset.subset_id}"
        results.append(
            EvaluationResult(
                subset_key=key,
                num_tasks=len(route),
                total_reward=float(info.get("total_reward", 0.0)),
                raw_total_reward=float(info.get("raw_total_reward", info.get("total_reward", 0.0))),
                total_distance=float(info.get("total_distance", 0.0)),
                total_lateness=float(info.get("total_lateness", 0.0)),
                unassigned_tasks=int(info.get("unassigned_tasks", 0)),
            )
        )
    return results


def summarise_results(results: Sequence[EvaluationResult]) -> dict:
    if not results:
        return {
            "count": 0,
            "avg_reward": 0.0,
            "avg_raw_reward": 0.0,
            "avg_distance": 0.0,
            "avg_lateness": 0.0,
            "avg_unassigned": 0.0,
        }
    return {
        "count": len(results),
        "avg_reward": statistics.mean(res.total_reward for res in results),
        "avg_raw_reward": statistics.mean(res.raw_total_reward for res in results),
        "avg_distance": statistics.mean(res.total_distance for res in results),
        "avg_lateness": statistics.mean(res.total_lateness for res in results),
        "avg_unassigned": statistics.mean(res.unassigned_tasks for res in results),
    }


def train_planner(
    planner: DQNRoutePlanner,
    subsets: Sequence[TaskSubset],
    episodes: int,
    rng: random.Random,
    log_interval: int,
) -> List[float]:
    rewards: List[float] = []
    if episodes <= 0 or not subsets:
        return rewards
    order = list(range(len(subsets)))
    for episode in range(episodes):
        rng.shuffle(order)
        subset = subsets[order[episode % len(order)]]
        tasks = subset.fresh_tasks()
        reward = planner.train_episode(tasks, subset.capacity, subset.depot[0], subset.depot[1])
        rewards.append(reward)
        if (episode + 1) % log_interval == 0 or episode == 0:
            epsilon = getattr(planner, "epsilon", 0.0)
            print(f"Episode {episode + 1}/{episodes}: reward={reward:.3f} epsilon={epsilon:.3f}")
    return rewards


def export_evaluation(path: Path, results: Sequence[EvaluationResult]) -> None:
    header = "subset,num_tasks,total_reward,raw_total_reward,total_distance,total_lateness,unassigned_tasks\n"
    lines = [header]
    for res in results:
        lines.append(
            f"{res.subset_key},{res.num_tasks},{res.total_reward:.4f},{res.raw_total_reward:.4f},{res.total_distance:.4f},{res.total_lateness:.4f},{res.unassigned_tasks}\n"
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone trainer for the RL route planner")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DIR / "Dateset" / "In",
        help="Path to directory containing Solomon VRPTW instance files",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=[],
        help="Instance filenames (within dataset root) used for sampling tasks",
    )
    parser.add_argument(
        "--instance-sample-size",
        type=int,
        default=0,
        help="When >0, randomly sample this many instance files from the dataset root (recursively)",
    )
    parser.add_argument(
        "--use-all-instances",
        action="store_true",
        help="Ignore --instances/--instance-sample-size and use every instance in the dataset root",
    )
    parser.add_argument("--train-samples", type=int, default=200, help="Number of sampled task subsets per instance for training")
    parser.add_argument("--test-samples", type=int, default=50, help="Number of sampled task subsets per instance for testing")
    parser.add_argument("--subset-min", type=int, default=5, help="Minimum number of tasks in a sampled subset")
    parser.add_argument("--subset-max", type=int, default=15, help="Maximum number of tasks in a sampled subset")
    parser.add_argument("--train-episodes", type=int, default=800, help="Number of RL training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-interval", type=int, default=50, help="How often to print training progress")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.1, help="Minimum epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay multiplier per episode")
    parser.add_argument("--warmup", type=int, default=256, help="Replay buffer warmup before training starts")
    parser.add_argument("--target-sync", type=int, default=250, help="Target network sync interval")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for DQN updates")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store checkpoints, datasets, and metrics",
    )
    parser.add_argument(
        "--train-subsets",
        type=Path,
        default=None,
        help="Path to existing train_subsets.json to reuse instead of sampling",
    )
    parser.add_argument(
        "--test-subsets",
        type=Path,
        default=None,
        help="Path to existing test_subsets.json to reuse for evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to load before training (allows fine-tuning or evaluation only)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training and only evaluate using the provided checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    if USE_CLI:
        args = parse_args()
    else:
        args = argparse.Namespace(**USER_RUN_CONFIG)

    args.dataset_root = Path(args.dataset_root)
    raw_instances = getattr(args, "instances", [])
    if raw_instances is None:
        raw_instances = []
    args.instances = list(raw_instances)
    args.instance_sample_size = int(getattr(args, "instance_sample_size", 0) or 0)
    args.use_all_instances = bool(getattr(args, "use_all_instances", False))
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
    if getattr(args, "train_subsets", None) is not None:
        args.train_subsets = Path(args.train_subsets)
    if getattr(args, "test_subsets", None) is not None:
        args.test_subsets = Path(args.test_subsets)

    rng = random.Random(args.seed)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    instance_files: List[Path]
    if args.use_all_instances:
        instance_files = sorted(
            {
                path.resolve()
                for path in dataset_root.rglob("*.txt")
                if path.is_file() and "roadtopology" not in {part.lower() for part in path.parts}
            }
        )
        if not instance_files:
            raise RuntimeError(f"No instance files found under {dataset_root} when --use-all-instances is set")
    elif args.instance_sample_size and args.instance_sample_size > 0:
        candidates = [
            path
            for path in dataset_root.rglob("*.txt")
            if path.is_file() and "roadtopology" not in {part.lower() for part in path.parts}
        ]
        if not candidates:
            raise RuntimeError(f"No instance files found under {dataset_root} for sampling")
        rng.shuffle(candidates)
        sample_count = min(args.instance_sample_size, len(candidates))
        instance_files = candidates[:sample_count]
    else:
        if not args.instances:
            raise RuntimeError("No instances provided; specify --instances, --instance-sample-size, or --use-all-instances")
        instance_files = [dataset_root / name for name in args.instances]

    for file in instance_files:
        if not file.exists():
            raise FileNotFoundError(f"Instance file {file} does not exist")

    def _instance_label(path: Path) -> str:
        try:
            return path.relative_to(dataset_root).as_posix()
        except ValueError:
            return path.name

    if args.instance_sample_size and args.instance_sample_size > 0:
        print("Sampled instances:")
        for chosen in instance_files:
            print(f"  - {_instance_label(chosen)}")

    instances = [parse_vrptw_instance(path, label=_instance_label(path)) for path in instance_files]
    if not instances:
        raise RuntimeError("No instances were loaded for training")

    output_dir = ensure_output_dir(args.output_dir)

    if getattr(args, "train_subsets", None) is not None:
        if not args.train_subsets.exists():
            raise FileNotFoundError(f"Train subset file {args.train_subsets} does not exist")
        train_subsets = load_task_subsets(args.train_subsets)
        write_json(output_dir / "train_subsets.json", [subset.to_serializable() for subset in train_subsets])
    else:
        train_subsets = []
        for inst in instances:
            train_subsets.extend(
                generate_task_subsets(
                    inst,
                    max(0, args.train_samples),
                    args.subset_min,
                    args.subset_max,
                    rng,
                    start_id=0,
                )
            )
        write_json(output_dir / "train_subsets.json", [subset.to_serializable() for subset in train_subsets])

    if getattr(args, "test_subsets", None) is not None:
        if not args.test_subsets.exists():
            raise FileNotFoundError(f"Test subset file {args.test_subsets} does not exist")
        test_subsets = load_task_subsets(args.test_subsets)
        write_json(output_dir / "test_subsets.json", [subset.to_serializable() for subset in test_subsets])
    else:
        test_subsets = []
        for inst in instances:
            start_id = max(0, args.train_samples)
            test_subsets.extend(
                generate_task_subsets(
                    inst,
                    max(0, args.test_samples),
                    args.subset_min,
                    args.subset_max,
                    rng,
                    start_id=start_id,
                )
            )
        write_json(output_dir / "test_subsets.json", [subset.to_serializable() for subset in test_subsets])

    config = PlannerConfig(
        max_tasks=args.subset_max,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        warmup=args.warmup,
        target_sync=args.target_sync,
    )

    env = VRPTWRoutingEnv(config)
    planner = DQNRoutePlanner(env, config)

    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found")
        planner.load(str(args.checkpoint))
        print(f"Loaded checkpoint from {args.checkpoint}")

    if not args.no_train and args.train_episodes > 0 and train_subsets:
        rewards = train_planner(planner, train_subsets, args.train_episodes, rng, max(1, args.log_interval))
        write_json(output_dir / "training_rewards.json", rewards)
        checkpoint_path = output_dir / "planner_checkpoint.pt"
        planner.save(str(checkpoint_path))
        print(f"Checkpoint saved to {checkpoint_path}")
    elif args.no_train and args.checkpoint is None:
        print("--no-train specified without a checkpoint; skipping training and evaluation")
        return

    if test_subsets:
        results = evaluate_subsets(planner, test_subsets)
        export_evaluation(output_dir / "test_metrics.csv", results)
        write_json(output_dir / "test_metrics.json", [res.to_serializable() for res in results])
        summary = summarise_results(results)
        write_json(output_dir / "test_summary.json", summary)
        print("Test evaluation summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("No test subsets generated; skipping evaluation")


if __name__ == "__main__":
    main()
