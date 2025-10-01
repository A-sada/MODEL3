#!/usr/bin/env python3
"""Evaluate the pretrained RL route planner on the predefined R101 routes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from classes import Task
from rl_route_planner import load_pretrained_planner
from train_rl_route_planner import parse_vrptw_instance


@dataclass
class RouteEvaluation:
    route_index: int
    requested_tasks: List[int]
    planned_route: List[int]
    total_reward: float
    raw_total_reward: float
    total_distance: float
    total_lateness: float
    unassigned_tasks: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "route_index": self.route_index,
            "requested_tasks": self.requested_tasks,
            "planned_route": self.planned_route,
            "total_reward": self.total_reward,
            "raw_total_reward": self.raw_total_reward,
            "total_distance": self.total_distance,
            "total_lateness": self.total_lateness,
            "unassigned_tasks": self.unassigned_tasks,
        }


def clone_task(task: Task) -> Task:
    return Task(
        task.id,
        task.x_coordinate,
        task.y_coordinate,
        task.weight,
        task.ready_time,
        task.due_date,
        task.service_time,
    )


def parse_route_file(path: Path) -> List[List[int]]:
    routes: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or not stripped.lower().startswith("route"):
                continue
            try:
                _, tasks_part = stripped.split(":", 1)
            except ValueError:
                continue
            task_ids = [int(token) for token in tasks_part.strip().split()] if tasks_part.strip() else []
            routes.append(task_ids)
    return routes


def resolve_checkpoint(explicit: Path | None, script_dir: Path) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_path = os.getenv("RL_PLANNER_CHECKPOINT")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return candidate
    candidates = [
        script_dir / "pretrained" / "planner_checkpoint.pt",
        script_dir / "pretrained_models" / "route_planner.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("事前学習済みルートプランナーのチェックポイントが見つかりません")


def build_tasks(route_ids: Sequence[int], task_map: Dict[int, Task]) -> List[Task]:
    tasks: List[Task] = []
    missing: List[int] = []
    for task_id in route_ids:
        task = task_map.get(task_id)
        if task is None:
            missing.append(task_id)
            continue
        tasks.append(clone_task(task))
    if missing:
        raise KeyError(f"存在しないタスクIDが指定されました: {missing}")
    return tasks


def evaluate_routes(
    routes: Sequence[Sequence[int]],
    instance_tasks: Dict[int, Task],
    capacity: float,
    depot_x: float,
    depot_y: float,
    checkpoint: Path,
) -> List[RouteEvaluation]:
    _, planner = load_pretrained_planner(checkpoint)
    results: List[RouteEvaluation] = []
    for index, route_ids in enumerate(routes, start=1):
        requested = list(route_ids)
        tasks = build_tasks(route_ids, instance_tasks)
        planned, info = planner.plan_route(tasks, capacity, depot_x, depot_y, greedy=True)
        planned_ids = [task.id for task in planned]
        evaluation = RouteEvaluation(
            route_index=index,
            requested_tasks=requested,
            planned_route=planned_ids,
            total_reward=float(info.get("total_reward", 0.0)),
            raw_total_reward=float(info.get("raw_total_reward", info.get("total_reward", 0.0))),
            total_distance=float(info.get("total_distance", 0.0)),
            total_lateness=float(info.get("total_lateness", 0.0)),
            unassigned_tasks=int(info.get("unassigned_tasks", 0)),
        )
        results.append(evaluation)
    return results


def summarise(results: Sequence[RouteEvaluation]) -> Dict[str, float]:
    if not results:
        return {
            "count": 0,
            "avg_reward": 0.0,
            "avg_raw_reward": 0.0,
            "avg_distance": 0.0,
            "avg_lateness": 0.0,
            "avg_unassigned": 0.0,
        }
    count = float(len(results))
    return {
        "count": len(results),
        "avg_reward": sum(item.total_reward for item in results) / count,
        "avg_raw_reward": sum(item.raw_total_reward for item in results) / count,
        "avg_distance": sum(item.total_distance for item in results) / count,
        "avg_lateness": sum(item.total_lateness for item in results) / count,
        "avg_unassigned": sum(item.unassigned_tasks for item in results) / count,
    }


def write_outputs(
    results: Sequence[RouteEvaluation],
    summary: Dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "routes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "route_index",
                "requested_tasks",
                "planned_route",
                "total_reward",
                "raw_total_reward",
                "total_distance",
                "total_lateness",
                "unassigned_tasks",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.route_index,
                    " ".join(str(task) for task in item.requested_tasks),
                    " ".join(str(task) for task in item.planned_route),
                    f"{item.total_reward:.6f}",
                    f"{item.raw_total_reward:.6f}",
                    f"{item.total_distance:.6f}",
                    f"{item.total_lateness:.6f}",
                    item.unassigned_tasks,
                ]
            )
    json_path = output_dir / "routes.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in results], handle, indent=2, ensure_ascii=False)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> None:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="Evaluate RL planner on predefined R101 routes")
    parser.add_argument(
        "--route-file",
        type=Path,
        default=root_dir / "r101_test.txt",
        help="Path to the file describing target routes",
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        default=script_dir / "r101.txt",
        help="VRPTW instance file providing task definitions",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the pretrained planner checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store evaluation outputs",
    )
    args = parser.parse_args(argv)

    route_file = args.route_file.expanduser().resolve()
    instance_file = args.instance_file.expanduser().resolve()
    if not route_file.exists():
        raise FileNotFoundError(f"ルート定義ファイルが見つかりません: {route_file}")
    if not instance_file.exists():
        raise FileNotFoundError(f"インスタンスファイルが見つかりません: {instance_file}")

    checkpoint = resolve_checkpoint(args.checkpoint, script_dir)

    routes = parse_route_file(route_file)
    if len(routes) != 19:
        print(f"警告: ルート数が19ではありません ({len(routes)}件) -> ファイル内容を確認してください")
    instance = parse_vrptw_instance(instance_file, label=instance_file.name)
    task_map: Dict[int, Task] = {task.id: task for task in instance.tasks}

    results = evaluate_routes(
        routes,
        task_map,
        instance.capacity,
        instance.depot[0],
        instance.depot[1],
        checkpoint,
    )
    summary = summarise(results)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = root_dir / "output_files" / "rl_planner" / "r101_routes" / timestamp
    write_outputs(results, summary, output_dir)

    print(f"評価結果を保存しました: {output_dir}")
    for item in results:
        print(
            f"Route {item.route_index:02d}: requested={item.requested_tasks} planned={item.planned_route} "
            f"reward={item.total_reward:.3f} distance={item.total_distance:.3f} "
            f"lateness={item.total_lateness:.3f} unassigned={item.unassigned_tasks}"
        )
    print("平均統計:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
