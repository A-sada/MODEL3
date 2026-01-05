#!/usr/bin/env bash
set -euo pipefail

IMAGE=rl-route:cpu

docker build -f Dockerfile.cpu -t "${IMAGE}" .
mkdir -p new_output

docker run --rm -it   -e PYTHONPATH=/app:/app/MODEL3   -v "$PWD":/app   -v "$PWD/new_output":/app/new_output   "${IMAGE}"   bash -lc 'python MODEL3/train_rl_route_planner.py --use-all-instances --train-episodes 2000 --output-dir /app/new_output/rl_planner_run'
