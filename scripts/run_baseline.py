from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments.csv"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_yolo26.py"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts" / "check_environment.py"


def load_valid_workloads(csv_path: Path) -> set[str]:
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        return {
            row["workload_id"].strip()
            for row in reader
        }


def parse_workloads(text: str) -> list[str]:
    workloads = [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]

    if not workloads:
        raise ValueError("No workloads specified.")

    return workloads


def run_with_log(
    command: list[str],
    log_path: Path,
) -> int:
    """
    Run a subprocess while simultaneously:
      1. showing output in the terminal
      2. saving output to run.log
    """

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return process.wait()


def print_command(command: list[str]) -> None:
    print(
        " ".join(
            f'"{arg}"' if " " in arg else arg
            for arg in command
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO26 baseline ground-truth experiments."
        )
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help=(
            "Logical hardware name, "
            "for example RTX5090 or RTX4090."
        ),
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics CUDA device. Default: 0",
    )

    parser.add_argument(
        "--workloads",
        default="C01",
        help=(
            "Comma-separated workload IDs. "
            "Example: C01,C04,C09"
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of baseline repeats per workload.",
    )

    parser.add_argument(
        "--experiments",
        type=Path,
        default=DEFAULT_EXPERIMENTS,
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print commands only. "
            "No GPU or training is required."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete an existing run directory "
            "and rerun the experiment."
        ),
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    experiments_path = args.experiments.resolve()
    runs_root = args.runs_root.resolve()

    valid_workloads = load_valid_workloads(
        experiments_path
    )

    workloads = parse_workloads(
        args.workloads
    )

    unknown = [
        workload
        for workload in workloads
        if workload not in valid_workloads
    ]

    if unknown:
        raise ValueError(
            f"Unknown workload IDs: {unknown}"
        )

    print("=" * 72)
    print("YOLO26 Baseline Runner")
    print("=" * 72)
    print(f"Device ID : {args.device_id}")
    print(f"CUDA dev  : {args.device}")
    print(f"Workloads : {', '.join(workloads)}")
    print(f"Repeats   : {args.repeats}")
    print(f"Dry run   : {args.dry_run}")
    print("=" * 72)
    print()

    # ------------------------------------------------------------
    # Formal GPU runs must first pass the software environment check.
    # ------------------------------------------------------------

    environment_command = [
        sys.executable,
        str(CHECK_ENV_SCRIPT),
        "--gpu",
    ]

    print("[Environment check]")
    print_command(environment_command)
    print()

    if not args.dry_run:
        result = subprocess.run(
            environment_command,
            check=False,
        )

        if result.returncode != 0:
            raise SystemExit(
                "GPU environment check failed. "
                "Baseline collection aborted."
            )

    completed = 0
    skipped = 0

    # ------------------------------------------------------------
    # Run each workload independently.
    # ------------------------------------------------------------

    for workload_id in workloads:

        for repeat in range(1, args.repeats + 1):

            run_dir = (
                runs_root
                / args.device_id
                / workload_id
                / f"baseline_{repeat:02d}"
            )

            summary_file = (
                run_dir
                / "summary.json"
            )

            if summary_file.exists() and not args.force:
                print(
                    f"[SKIP] {workload_id} "
                    f"repeat={repeat}: "
                    f"summary.json already exists"
                )

                skipped += 1
                continue

            if run_dir.exists() and args.force:
                if not args.dry_run:
                    shutil.rmtree(run_dir)

            command = [
                sys.executable,
                str(TRAIN_SCRIPT),

                "--workload-id",
                workload_id,

                "--device-id",
                args.device_id,

                "--device",
                args.device,

                "--run-type",
                "baseline",

                "--repeat",
                str(repeat),

                "--experiments",
                str(experiments_path),

                "--runs-root",
                str(runs_root),
            ]

            print()
            print("-" * 72)
            print(
                f"[RUN] {workload_id} "
                f"repeat={repeat}"
            )
            print("-" * 72)

            print_command(command)

            if args.dry_run:
                continue

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            log_path = (
                run_dir
                / "run.log"
            )

            return_code = run_with_log(
                command,
                log_path,
            )

            if return_code != 0:
                print()
                print(
                    f"[FAILED] {workload_id} "
                    f"repeat={repeat}"
                )

                print(
                    f"Log: {log_path}"
                )

                raise SystemExit(
                    return_code
                )

            if not summary_file.exists():
                raise RuntimeError(
                    "Training command returned successfully "
                    "but summary.json was not generated:\n"
                    f"{summary_file}"
                )

            print(
                f"[OK] {workload_id} "
                f"repeat={repeat}"
            )

            completed += 1

    print()
    print("=" * 72)

    if args.dry_run:
        print(
            "DRY RUN COMPLETE - "
            "no training was executed."
        )
    else:
        print("BASELINE COLLECTION COMPLETE")
        print(f"Completed : {completed}")
        print(f"Skipped   : {skipped}")

    print("=" * 72)


if __name__ == "__main__":
    main()