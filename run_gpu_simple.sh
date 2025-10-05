#!/usr/bin/env bash
set -euo pipefail

IMAGE=rl-route:gpu

# Rebuild (safe if layers are cached)
docker build -f Dockerfile.gpu -t "${IMAGE}" .

mkdir -p outputs

docker run --rm -it --gpus all   -e PYTHONPATH=/app:/app/MODEL3   -v "$PWD":/app   -v "$PWD/outputs":/app/output_files   "${IMAGE}"   bash -lc 'sed -i "s/^USE_CLI\s*=\s*False/USE_CLI = True/" MODEL3/train_rl_route_planner.py &&             python MODEL3/train_rl_route_planner.py               --use-all-instances               --train-episodes 20000               --output-dir /app/output_files/rl_planner_run'
