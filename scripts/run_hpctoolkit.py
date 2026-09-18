from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments.csv"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_CAPABILITIES_ROOT = PROJECT_ROOT / "capabilities"

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_yolo26.py"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts" / "check_environment.py"


def load_valid_workloads(csv_path: Path) -> set[str]:
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return {
            row["workload_id"].strip()
            for row in csv.DictReader(f)
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


def resolve_binary(
    name: str,
    hpct_root: Path | None,
) -> str | None:

    if hpct_root is not None:
        candidate = hpct_root / "bin" / name

        if candidate.exists():
            return str(candidate)

    return shutil.which(name)


def print_command(command: list[str]) -> None:
    print(
        " ".join(
            f'"{arg}"' if " " in arg else arg
            for arg in command
        )
    )


def run_with_log(
    command: list[str],
    cwd: Path,
    log_path: Path,
) -> int:

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
            cwd=cwd,
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


def load_validation(
    validation_file: Path,
) -> dict:

    return json.loads(
        validation_file.read_text(
            encoding="utf-8-sig"
        )
    )


def get_version(
    binary: str,
) -> str | None:

    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        return (
            result.stdout.strip()
            or result.stderr.strip()
        )

    except Exception:
        return None


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO26 HPCToolkit GPU profiling experiments."
        )
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="Logical GPU label, e.g. RTX5090.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device used by Ultralytics.",
    )

    parser.add_argument(
        "--workloads",
        default="C01",
        help="Comma-separated workload IDs.",
    )

    parser.add_argument(
        "--profile-mode",
        choices=[
            "cuda",
            "cuda-trace",
        ],
        default="cuda",
        help=(
            "cuda       : hpcrun -e gpu=cuda\n"
            "cuda-trace : hpcrun -e gpu=cuda -t"
        ),
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--hpct-root",
        type=Path,
        default=None,
        help=(
            "HPCToolkit installation root. "
            "If omitted, hpcrun is searched in PATH."
        ),
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
        "--capabilities-root",
        type=Path,
        default=DEFAULT_CAPABILITIES_ROOT,
    )

    parser.add_argument(
        "--postprocess",
        action="store_true",
        help=(
            "After collection, run hpcstruct and hpcprof. "
            "Disabled by default to reduce disk/time usage."
        ),
    )

    parser.add_argument(
        "--hpcstruct-jobs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--hpcprof-jobs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help=(
            "Allow profiling without a successful "
            "validate_hpctoolkit.py result. "
            "Not recommended for formal data collection."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    hpct_root = (
        args.hpct_root.resolve()
        if args.hpct_root
        else None
    )

    experiments_path = (
        args.experiments.resolve()
    )

    runs_root = (
        args.runs_root.resolve()
    )

    capabilities_root = (
        args.capabilities_root.resolve()
    )

    workloads = parse_workloads(
        args.workloads
    )

    valid_workloads = load_valid_workloads(
        experiments_path
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

    hpcrun = resolve_binary(
        "hpcrun",
        hpct_root,
    )

    hpcstruct = resolve_binary(
        "hpcstruct",
        hpct_root,
    )

    hpcprof = resolve_binary(
        "hpcprof",
        hpct_root,
    )

    # In dry-run mode we allow placeholder paths because
    # development laptops normally have no HPCToolkit.
    if args.dry_run:

        if hpcrun is None:
            hpcrun = "hpcrun"

        if hpcstruct is None:
            hpcstruct = "hpcstruct"

        if hpcprof is None:
            hpcprof = "hpcprof"

    else:

        if hpcrun is None:
            raise SystemExit(
                "hpcrun was not found."
            )

        if args.postprocess:
            if hpcstruct is None:
                raise SystemExit(
                    "hpcstruct was not found."
                )

            if hpcprof is None:
                raise SystemExit(
                    "hpcprof was not found."
                )

    # ==========================================================
    # Validation gate
    # ==========================================================

    validation_file = (
        capabilities_root
        / args.device_id
        / "hpctoolkit"
        / "validation"
        / "validation.json"
    )

    validation = None

    if validation_file.exists():

        validation = load_validation(
            validation_file
        )

    elif not args.dry_run and not args.allow_unvalidated:

        raise SystemExit(
            "HPCToolkit runtime validation file is missing:\n"
            f"  {validation_file}\n\n"
            "Run validate_hpctoolkit.py first."
        )

    if (
        validation is not None
        and not args.allow_unvalidated
    ):

        if not validation.get(
            "gpu_cuda_pass",
            False,
        ):

            raise SystemExit(
                "HPCToolkit gpu=cuda validation "
                "did not pass on this device."
            )

    # ==========================================================
    # Environment
    # ==========================================================

    print("=" * 72)
    print("YOLO26 HPCToolkit Runner")
    print("=" * 72)

    print(f"Device ID    : {args.device_id}")
    print(f"CUDA dev     : {args.device}")
    print(f"Workloads    : {', '.join(workloads)}")
    print(f"Profile mode : {args.profile_mode}")
    print(f"Repeat       : {args.repeat}")
    print(f"HPCT root    : {hpct_root}")
    print(f"Postprocess  : {args.postprocess}")
    print(f"Dry run      : {args.dry_run}")

    print("=" * 72)

    env_command = [
        sys.executable,
        str(CHECK_ENV_SCRIPT),
        "--gpu",
    ]

    print()
    print("[Environment check]")
    print_command(env_command)

    hpct_version = None

    if not args.dry_run:

        result = subprocess.run(
            env_command,
            check=False,
        )

        if result.returncode != 0:

            raise SystemExit(
                "GPU environment check failed."
            )

        hpct_version = get_version(
            hpcrun
        )

        print()
        print("[HPCToolkit]")
        print(
            hpct_version
            or "version unavailable"
        )

    # ==========================================================
    # Workloads
    # ==========================================================

    completed = 0
    skipped = 0

    for workload_id in workloads:

        mode_name = {
            "cuda": "hpctoolkit_cuda",
            "cuda-trace": "hpctoolkit_cuda_trace",
        }[args.profile_mode]

        run_dir = (
            runs_root
            / args.device_id
            / workload_id
            / f"{mode_name}_{args.repeat:02d}"
        )

        measurements_dir = (
            run_dir
            / "hpctoolkit-measurements"
        )

        complete_marker = (
            run_dir
            / "collection_complete.json"
        )

        if (
            complete_marker.exists()
            and not args.force
        ):

            print(
                f"[SKIP] {workload_id}: "
                "collection already complete"
            )

            skipped += 1
            continue

        if run_dir.exists() and args.force:

            if not args.dry_run:
                shutil.rmtree(
                    run_dir
                )

        # ------------------------------------------------------
        # hpcrun command
        # ------------------------------------------------------

        command = [
            hpcrun,

            "-e",
            "gpu=cuda",
        ]

        if args.profile_mode == "cuda-trace":

            command.append(
                "-t"
            )

        command.extend([
            "-o",
            str(measurements_dir),

            sys.executable,
            str(TRAIN_SCRIPT),

            "--workload-id",
            workload_id,

            "--device-id",
            args.device_id,

            "--device",
            args.device,

            "--run-type",
            mode_name,

            "--repeat",
            str(args.repeat),

            "--experiments",
            str(experiments_path),

            "--runs-root",
            str(runs_root),
        ])

        print()
        print("-" * 72)
        print(
            f"[HPCTOOLKIT-{args.profile_mode.upper()}] "
            f"{workload_id} "
            f"repeat={args.repeat}"
        )
        print("-" * 72)

        print_command(
            command
        )

        if args.dry_run:
            continue

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save exact settings.
        config = {
            "device_id": args.device_id,
            "workload_id": workload_id,

            "profile_mode": (
                args.profile_mode
            ),

            "repeat": args.repeat,

            "hpctoolkit_version": (
                hpct_version
            ),

            "hpct_root": (
                str(hpct_root)
                if hpct_root
                else None
            ),

            "event": "gpu=cuda",

            "trace": (
                args.profile_mode
                == "cuda-trace"
            ),

            "measurements_dir": (
                str(measurements_dir)
            ),

            "postprocess": (
                args.postprocess
            ),

            "validation_file": (
                str(validation_file)
            ),

            "command": command,
        }

        (
            run_dir
            / "profiling_config.json"
        ).write_text(
            json.dumps(
                config,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------
        # Collection
        # ------------------------------------------------------

        rc = run_with_log(
            command,
            PROJECT_ROOT,
            run_dir / "hpcrun.log",
        )

        if rc != 0:

            raise SystemExit(
                f"hpcrun failed for "
                f"{workload_id}.\n"
                f"See {run_dir / 'hpcrun.log'}"
            )

        if not measurements_dir.exists():

            raise RuntimeError(
                "hpcrun returned successfully but "
                "measurements directory is missing:\n"
                f"{measurements_dir}"
            )

        # ------------------------------------------------------
        # Optional hpcstruct / hpcprof
        # ------------------------------------------------------

        struct_pass = None
        prof_pass = None

        if args.postprocess:

            struct_command = [
                hpcstruct,
                "-j",
                str(args.hpcstruct_jobs),
                str(measurements_dir),
            ]

            print()
            print("[hpcstruct]")
            print_command(
                struct_command
            )

            rc = run_with_log(
                struct_command,
                PROJECT_ROOT,
                run_dir / "hpcstruct.log",
            )

            struct_pass = (
                rc == 0
            )

            if not struct_pass:

                raise SystemExit(
                    f"hpcstruct failed for "
                    f"{workload_id}."
                )

            prof_command = [
                hpcprof,
                "-j",
                str(args.hpcprof_jobs),
                str(measurements_dir),
            ]

            print()
            print("[hpcprof]")
            print_command(
                prof_command
            )

            rc = run_with_log(
                prof_command,
                run_dir,
                run_dir / "hpcprof.log",
            )

            prof_pass = (
                rc == 0
            )

            if not prof_pass:

                raise SystemExit(
                    f"hpcprof failed for "
                    f"{workload_id}."
                )

        # ------------------------------------------------------
        # Complete marker
        # ------------------------------------------------------

        result = {
            "device_id": args.device_id,
            "workload_id": workload_id,

            "profile_mode": (
                args.profile_mode
            ),

            "repeat": args.repeat,

            "hpcrun_pass": True,

            "hpcstruct_pass": (
                struct_pass
            ),

            "hpcprof_pass": (
                prof_pass
            ),

            "measurements_dir": (
                str(measurements_dir)
            ),
        }

        complete_marker.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[OK] {workload_id}"
        )

        completed += 1

    print()
    print("=" * 72)

    if args.dry_run:

        print(
            "DRY RUN COMPLETE - "
            "no HPCToolkit profiling was executed."
        )

    else:

        print(
            "HPCTOOLKIT COLLECTION COMPLETE"
        )

        print(
            f"Completed : {completed}"
        )

        print(
            f"Skipped   : {skipped}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()