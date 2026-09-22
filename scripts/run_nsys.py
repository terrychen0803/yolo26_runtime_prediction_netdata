from __future__ import annotations

import argparse
import csv
import getpass
import json
import pwd
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments.csv"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_yolo26.py"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts" / "check_environment.py"
NETDATA_SCRIPT = PROJECT_ROOT / "scripts" / "collect_netdata.py"


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


def print_command(command: list[str]) -> None:
    print(
        " ".join(
            f'"{arg}"' if " " in arg else arg
            for arg in command
        )
    )


def run_with_log(
    command: list[str],
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


def get_nsys_version(
    nsys_path: str,
) -> str | None:
    """
    Return the version string for the exact Nsight Systems
    executable selected by this runner.
    """

    try:
        result = subprocess.run(
            [nsys_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        output = (
            result.stdout.strip()
            or result.stderr.strip()
        )

        return output

    except Exception:
        return None


def restore_file_ownership(
    path: Path,
    username: str,
) -> None:
    """
    Nsight Systems runs as root when --sudo-nsys is used.
    The generated .nsys-rep is therefore normally owned by root.

    Restore ownership to the user who owns/runs the experiment
    so that later export, feature extraction, deletion, etc.
    can be done without sudo.
    """

    try:
        user_info = pwd.getpwnam(username)
    except KeyError as exc:
        raise RuntimeError(
            f"Cannot resolve Linux user: {username}"
        ) from exc

    owner = (
        f"{user_info.pw_uid}:"
        f"{user_info.pw_gid}"
    )

    result = subprocess.run(
        [
            "sudo",
            "chown",
            owner,
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Nsight report was generated, but ownership "
            "could not be restored.\n"
            f"File: {path}\n"
            f"User: {username}\n"
            f"Error: {result.stderr.strip()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO26 Nsight Systems profiling experiments."
        )
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="Logical hardware label, e.g. RTX5090.",
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
            "trace",
            "gpu-metrics",
            "cpu-metrics",
        ],
        default="trace",
        help=(
            "trace       : CUDA/NVTX operation timeline\n"
            "gpu-metrics : CUDA/NVTX + GPU metric sampling\n"
            "cpu-metrics : CUDA/NVTX + system-wide CPU PMU metrics"
        ),
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--nsys-path",
        type=Path,
        default=None,
        help=(
            "Explicit Nsight Systems CLI executable. "
            "If omitted, resolve 'nsys' from PATH."
        ),
    )

    # ----------------------------------------------------------
    # GPU Metrics configuration
    # ----------------------------------------------------------

    parser.add_argument(
        "--gpu-metrics-devices",
        default="0",
        help=(
            "Nsight Systems GPU metrics device selector. "
            "Example: 0, all, cuda-visible."
        ),
    )

    parser.add_argument(
        "--gpu-metrics-set",
        default=None,
        help=(
            "GPU metrics set alias. "
            "If omitted, Nsight Systems selects the first "
            "suitable metric set."
        ),
    )

    parser.add_argument(
        "--gpu-metrics-frequency",
        type=int,
        default=1000,
        help=(
            "GPU metric sampling frequency in Hz. "
            "Default for this experiment: 1000 Hz."
        ),
    )

    # ----------------------------------------------------------
    # CPU Metrics configuration
    # ----------------------------------------------------------

    parser.add_argument(
        "--cpu-metrics",
        default=None,
        help=(
            "Comma-separated CPU metrics/events/metric-set "
            "supported by the target node. "
            "Discover them first using "
            "'nsys profile --cpu-metrics=help:all'."
        ),
    )

    parser.add_argument(
        "--event-sampling-interval",
        type=int,
        default=5,
        help=(
            "CPU event sampling interval in milliseconds. "
            "Default: 5 ms."
        ),
    )

    # ----------------------------------------------------------
    # Privileged Nsight configuration
    # ----------------------------------------------------------

    parser.add_argument(
        "--sudo-nsys",
        action="store_true",
        help=(
            "Run Nsight Systems itself with sudo. "
            "Useful when NVIDIA GPU performance counters "
            "require elevated privilege. "
            "The target application is still run as the user "
            "specified by --run-as."
        ),
    )

    parser.add_argument(
        "--run-as",
        default=getpass.getuser(),
        help=(
            "Linux user used to run the profiled target "
            "when --sudo-nsys is enabled. "
            "Default: current user."
        ),
    )

    # ----------------------------------------------------------
    # Experiment paths / control
    # ----------------------------------------------------------

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
        "--netdata",
        action="store_true",
        help=(
            "Collect node-level Netdata metrics "
            "during each profiling run."
        ),
    )

    parser.add_argument(
        "--netdata-url",
        default="http://127.0.0.1:19999",
        help="Local Netdata Agent base URL.",
    )

    parser.add_argument(
        "--netdata-interval",
        type=float,
        default=1.0,
        help="Netdata sampling interval in seconds. Default: 1.0.",
    )

    parser.add_argument(
        "--netdata-preroll",
        type=float,
        default=2.0,
        help=(
            "Seconds to collect Netdata before launching Nsight. "
            "Default: 2.0."
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

    # ----------------------------------------------------------
    # Argument validation
    # ----------------------------------------------------------

    if args.repeat < 1:
        raise ValueError(
            "--repeat must be >= 1"
        )

    if args.gpu_metrics_frequency < 10:
        raise ValueError(
            "--gpu-metrics-frequency must be >= 10 Hz"
        )

    if args.event_sampling_interval <= 0:
        raise ValueError(
            "--event-sampling-interval must be > 0"
        )

    if args.netdata_interval < 1.0:
        raise ValueError(
            "--netdata-interval must be >= 1.0 second"
        )

    if args.netdata_preroll < 0.0:
        raise ValueError(
            "--netdata-preroll must be >= 0"
        )

    if args.sudo_nsys and not args.run_as:
        raise ValueError(
            "--run-as must be specified "
            "when --sudo-nsys is used"
        )

    if (
        args.sudo_nsys
        and args.profile_mode != "gpu-metrics"
    ):
        print(
            "WARNING: --sudo-nsys is normally only "
            "required for gpu-metrics mode."
        )

    experiments_path = (
        args.experiments.resolve()
    )

    runs_root = (
        args.runs_root.resolve()
    )

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

    # ----------------------------------------------------------
    # CPU Metrics validation
    # ----------------------------------------------------------

    cpu_metrics_value = args.cpu_metrics

    if args.profile_mode == "cpu-metrics":
        if not cpu_metrics_value:

            if args.dry_run:
                cpu_metrics_value = (
                    "NODE_SUPPORTED_CPU_METRICS"
                )

            else:
                raise SystemExit(
                    "CPU metrics mode requires "
                    "--cpu-metrics.\n"
                    "First run on the target node:\n"
                    "  nsys profile "
                    "--cpu-metrics=help:all"
                )

    # ----------------------------------------------------------
    # Resolve Nsight executable
    # ----------------------------------------------------------

    if args.nsys_path is not None:
        requested_nsys = args.nsys_path.expanduser().resolve()

        if not requested_nsys.exists():
            raise SystemExit(
                "Explicit Nsight Systems CLI does not exist:\n"
                f"  {requested_nsys}"
            )

        if not requested_nsys.is_file():
            raise SystemExit(
                "Explicit Nsight Systems path is not a file:\n"
                f"  {requested_nsys}"
            )

        nsys_path = str(requested_nsys)

    else:
        discovered_nsys = shutil.which("nsys")

        if discovered_nsys is None:
            if args.dry_run:
                nsys_path = "nsys"
            else:
                raise SystemExit(
                    "Nsight Systems CLI 'nsys' "
                    "was not found in PATH."
                )
        else:
            nsys_path = str(
                Path(discovered_nsys).resolve()
            )

    # ----------------------------------------------------------
    # Configuration summary
    # ----------------------------------------------------------

    print("=" * 72)
    print("YOLO26 Nsight Systems Runner")
    print("=" * 72)

    print(
        f"Device ID    : {args.device_id}"
    )

    print(
        f"CUDA dev     : {args.device}"
    )

    print(
        f"Workloads    : "
        f"{', '.join(workloads)}"
    )

    print(
        f"Profile mode : {args.profile_mode}"
    )

    print(
        f"Repeat       : {args.repeat}"
    )

    print(
        f"Dry run      : {args.dry_run}"
    )

    print(
        f"Netdata      : {args.netdata}"
    )

    if args.netdata:
        print(
            f"Netdata URL  : {args.netdata_url}"
        )
        print(
            f"Netdata intv : {args.netdata_interval} s"
        )
        print(
            f"Netdata pre  : {args.netdata_preroll} s"
        )

    print(
        f"Nsight path  : {nsys_path}"
    )

    print(
        f"Sudo Nsight  : {args.sudo_nsys}"
    )

    if args.sudo_nsys:
        print(
            f"Run target as: {args.run_as}"
        )

    if args.profile_mode == "gpu-metrics":

        print(
            f"GPU metrics  : devices="
            f"{args.gpu_metrics_devices}"
        )

        print(
            f"GPU frequency: "
            f"{args.gpu_metrics_frequency} Hz"
        )

        print(
            f"GPU set      : "
            f"{args.gpu_metrics_set or 'auto'}"
        )

    if args.profile_mode == "cpu-metrics":

        print(
            f"CPU metrics  : "
            f"{cpu_metrics_value}"
        )

        print(
            f"CPU interval : "
            f"{args.event_sampling_interval} ms"
        )

    print("=" * 72)

    # ----------------------------------------------------------
    # Environment validation
    # ----------------------------------------------------------

    env_command = [
        sys.executable,
        str(CHECK_ENV_SCRIPT),
        "--gpu",
    ]

    print()
    print("[Environment check]")
    print_command(env_command)

    nsys_version = None

    if not args.dry_run:

        result = subprocess.run(
            env_command,
            check=False,
        )

        if result.returncode != 0:
            raise SystemExit(
                "GPU environment check failed."
            )

        # Resolve again after environment validation,
        # ensuring the exact executable used below is recorded.
        discovered_nsys = shutil.which("nsys")

        if discovered_nsys is None:
            raise SystemExit(
                "Nsight Systems CLI 'nsys' "
                "was not found in PATH."
            )

        nsys_path = str(
            Path(discovered_nsys).resolve()
        )

        nsys_version = get_nsys_version(
            nsys_path
        )

        if nsys_version:
            print()
            print("[Nsight Systems]")
            print(
                f"Path    : {nsys_path}"
            )
            print(
                f"Version : {nsys_version}"
            )

    completed = 0
    skipped = 0

    # ----------------------------------------------------------
    # Run workloads
    # ----------------------------------------------------------

    for workload_id in workloads:

        mode_dir_name = {
            "trace": "nsys_trace",
            "gpu-metrics": "nsys_gpu_metrics",
            "cpu-metrics": "nsys_cpu_metrics",
        }[args.profile_mode]

        run_type = mode_dir_name

        run_dir = (
            runs_root
            / args.device_id
            / workload_id
            / f"{mode_dir_name}_{args.repeat:02d}"
        )

        report_base = (
            run_dir
            / "profile"
        )

        report_file = Path(
            str(report_base) + ".nsys-rep"
        )

        netdata_csv = run_dir / "netdata.csv"
        netdata_log = run_dir / "netdata.log"

        if (
            report_file.exists()
            and not args.force
        ):

            print(
                f"[SKIP] {workload_id}: "
                "profile.nsys-rep already exists"
            )

            skipped += 1
            continue

        if (
            run_dir.exists()
            and args.force
        ):
            if not args.dry_run:
                shutil.rmtree(
                    run_dir
                )

        # ======================================================
        # Nsight launcher
        # ======================================================

        if args.sudo_nsys:

            user_info = pwd.getpwnam(
                args.run_as
            )

            target_home = user_info.pw_dir

            command = [
                "sudo",
                nsys_path,
                "profile",

                f"--run-as={args.run_as}",

                "--env-var="
                f"HOME={target_home},"
                f"USER={args.run_as},"
                f"LOGNAME={args.run_as},"
                f"XDG_CONFIG_HOME={target_home}/.config",
            ]

        else:

            command = [
                nsys_path,
                "profile",
            ]

        # ======================================================
        # Common trace configuration
        # ======================================================

        command.extend([
            "--trace=cuda,nvtx",

            # Disable CPU IP/backtrace sampling.
            "--sample=none",

            # Disable context-switch tracing for v1.
            "--cpuctxsw=none",
        ])

        # ======================================================
        # GPU Metrics mode
        # ======================================================

        if args.profile_mode == "gpu-metrics":

            command.append(
                "--gpu-metrics-devices="
                f"{args.gpu_metrics_devices}"
            )

            command.append(
                "--gpu-metrics-frequency="
                f"{args.gpu_metrics_frequency}"
            )

            if args.gpu_metrics_set:

                command.append(
                    "--gpu-metrics-set="
                    f"{args.gpu_metrics_set}"
                )

        # ======================================================
        # CPU Metrics mode
        # ======================================================

        elif args.profile_mode == "cpu-metrics":

            command.append(
                "--event-sample=system-wide"
            )

            command.append(
                "--cpu-metrics="
                f"{cpu_metrics_value}"
            )

            command.append(
                "--event-sampling-interval="
                f"{args.event_sampling_interval}"
            )

        # ======================================================
        # Output and target application
        # ======================================================

        command.extend([
            "--output",
            str(report_base),

            "--",

            sys.executable,
            str(TRAIN_SCRIPT),

            "--workload-id",
            workload_id,

            "--device-id",
            args.device_id,

            "--device",
            args.device,

            "--run-type",
            run_type,

            "--repeat",
            str(args.repeat),

            "--nvtx",

            "--experiments",
            str(experiments_path),

            "--runs-root",
            str(runs_root),
        ])

        print()
        print("-" * 72)

        print(
            f"[NSYS-{args.profile_mode.upper()}] "
            f"{workload_id} "
            f"repeat={args.repeat}"
        )

        print("-" * 72)

        print_command(command)

        if args.dry_run:
            continue

        # ------------------------------------------------------
        # Create run directory as the normal user BEFORE sudo.
        #
        # This keeps profiling_config.json and nsys.log owned by
        # the experiment user rather than root.
        # ------------------------------------------------------

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------
        # Save exact profiler configuration for reproducibility.
        # ------------------------------------------------------

        profiling_config = {
            "device_id": args.device_id,
            "workload_id": workload_id,
            "profile_mode": args.profile_mode,
            "repeat": args.repeat,

            "nsys_path": nsys_path,
            "nsys_version": nsys_version,

            "sudo_nsys": args.sudo_nsys,

            "run_as": (
                args.run_as
                if args.sudo_nsys
                else None
            ),

            "trace": [
                "cuda",
                "nvtx",
            ],

            "sample": "none",
            "cpuctxsw": "none",

            "gpu_metrics_devices": (
                args.gpu_metrics_devices
                if args.profile_mode
                == "gpu-metrics"
                else None
            ),

            "gpu_metrics_set": (
                args.gpu_metrics_set
                if args.profile_mode
                == "gpu-metrics"
                else None
            ),

            "gpu_metrics_frequency_hz": (
                args.gpu_metrics_frequency
                if args.profile_mode
                == "gpu-metrics"
                else None
            ),

            "cpu_metrics": (
                cpu_metrics_value
                if args.profile_mode
                == "cpu-metrics"
                else None
            ),

            "event_sampling_interval_ms": (
                args.event_sampling_interval
                if args.profile_mode
                == "cpu-metrics"
                else None
            ),

            "netdata_enabled": args.netdata,
            "netdata_url": (
                args.netdata_url
                if args.netdata
                else None
            ),
            "netdata_interval_s": (
                args.netdata_interval
                if args.netdata
                else None
            ),
            "netdata_preroll_s": (
                args.netdata_preroll
                if args.netdata
                else None
            ),

            "command": command,
        }

        (
            run_dir
            / "profiling_config.json"
        ).write_text(
            json.dumps(
                profiling_config,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------
        # Start optional Netdata sidecar before Nsight.
        # ------------------------------------------------------

        netdata_process = None
        netdata_log_file = None

        if args.netdata:
            netdata_command = [
                sys.executable,
                str(NETDATA_SCRIPT),
                "--output",
                str(netdata_csv),
                "--url",
                args.netdata_url,
                "--interval",
                str(args.netdata_interval),
            ]

            print()
            print("[Netdata monitor]")
            print_command(netdata_command)

            netdata_log_file = netdata_log.open(
                "w",
                encoding="utf-8",
            )

            netdata_process = subprocess.Popen(
                netdata_command,
                stdout=netdata_log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            time.sleep(args.netdata_preroll)

            if netdata_process.poll() is not None:
                netdata_log_file.close()
                raise RuntimeError(
                    "Netdata collector exited before "
                    "the workload started. "
                    f"Check: {netdata_log}"
                )

        # ------------------------------------------------------
        # Run Nsight Systems.
        # Always stop the Netdata sidecar, even on errors.
        # ------------------------------------------------------

        log_path = (
            run_dir
            / "nsys.log"
        )

        try:
            return_code = run_with_log(
                command,
                log_path,
            )

        finally:
            if netdata_process is not None:
                print()
                print("[Netdata monitor] stopping...")

                netdata_process.terminate()

                try:
                    netdata_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    netdata_process.kill()
                    netdata_process.wait()

                print("[Netdata monitor] stopped")

            if netdata_log_file is not None:
                netdata_log_file.close()

        # ------------------------------------------------------
        # Restore report ownership whenever a report exists.
        #
        # Nsight may still generate a report even when the
        # profiled target application exits with an error.
        # Therefore ownership restoration must happen BEFORE
        # checking the Nsight return code.
        # ------------------------------------------------------

        if (
            args.sudo_nsys
            and report_file.exists()
        ):

            restore_file_ownership(
                report_file,
                args.run_as,
            )

            print(
                "[Ownership restored] "
                f"{args.run_as}: "
                f"{report_file}"
            )

        # ------------------------------------------------------
        # Check profiling result
        # ------------------------------------------------------

        if return_code != 0:

            raise SystemExit(
                f"Nsight profiling failed "
                f"for {workload_id}.\n"
                f"Mode: {args.profile_mode}\n"
                f"Log: {log_path}"
            )

        if not report_file.exists():

            raise RuntimeError(
                "Nsight command completed but "
                "profile.nsys-rep was not generated:\n"
                f"{report_file}"
            )

        print(
            f"[OK] {workload_id}: "
            f"{report_file}"
        )

        completed += 1

    print()
    print("=" * 72)

    if args.dry_run:

        print(
            "DRY RUN COMPLETE - "
            "no profiling was executed."
        )

    else:

        print(
            "NSIGHT COLLECTION COMPLETE"
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