#!/usr/bin/env bash
set -euo pipefail

IMAGE=rl-route:cpu

docker build -f Dockerfile.cpu -t "${IMAGE}" .
mkdir -p outputs

docker run --rm -it   -e PYTHONPATH=/app:/app/MODEL3   -v "$PWD":/app   -v "$PWD/outputs":/app/output_files   "${IMAGE}"   bash -lc 'python MODEL3/train_rl_route_planner.py --use-all-instances --train-episodes 2000 --output-dir /app/output_files/rl_planner_run'
