#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_data_dir = script_dir / "In"
    vrptw_main = script_dir / "VRPTW-main.py"

    parser = argparse.ArgumentParser(
        description="Run VRPTW-main.py for each instance in a directory, sequentially."
    )
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory containing instance files (default: MODEL3/In).",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for instance files (default: *.txt).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {data_dir}")
    if not vrptw_main.exists():
        raise FileNotFoundError(f"VRPTW-main.py not found: {vrptw_main}")

    files = sorted(
        path
        for path in data_dir.glob(args.pattern)
        if path.is_file() and not path.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No instance files found in {data_dir} with {args.pattern}")

    total = len(files)
    for index, instance_path in enumerate(files, start=1):
        cmd = [args.python, str(vrptw_main), "--data", str(instance_path)]
        print(f"[{index}/{total}] {instance_path.name}")
        if args.dry_run:
            print(" ".join(cmd))
            continue
        env = os.environ.copy()
        env["VRPTW_OUTPUT_SUFFIX"] = instance_path.stem
        subprocess.run(cmd, cwd=project_root, check=True, env=env)


if __name__ == "__main__":
    main()
