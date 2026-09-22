from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DEFAULT_FIELDS = {
    "gpu_util": "GPU Util%",
    "gpu_mem": "GPU Mem Used(MB)",
    "gpu_temp": "GPU Temp(°C)",
    "gpu_power": "GPU Power(W)",
    "cpu_user": "CPU User%",
    "cpu_system": "CPU System%",
    "cpu_iowait": "CPU IOWait%",
    "load1": "Load 1min",
    "mem_used": "Mem Used(MB)",
    "top1_cpu": "Top1 CPU%",
}


def finite_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []

    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue

        if math.isfinite(value):
            values.append(value)

    return values


def mean_or_nan(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def std_or_nan(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def min_or_nan(values: list[float]) -> float:
    return min(values) if values else math.nan


def max_or_nan(values: list[float]) -> float:
    return max(values) if values else math.nan


def metric_stats(rows: list[dict], key: str, prefix: str) -> dict:
    values = finite_values(rows, key)

    return {
        f"{prefix}_mean": mean_or_nan(values),
        f"{prefix}_std": std_or_nan(values),
        f"{prefix}_min": min_or_nan(values),
        f"{prefix}_max": max_or_nan(values),
    }


def load_csv(path: Path) -> list[dict]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def is_steady(row: dict) -> bool:
    value = str(row.get("warmup", "")).strip().lower()
    return value in {"false", "0", "no"}


def build_row(
    condition_id: str,
    device_id: str,
    workload_id: str,
    repeat_dir: Path,
    pre_window_s: float,
) -> dict:
    summary_path = repeat_dir / "summary.json"
    iterations_path = repeat_dir / "iterations.csv"
    netdata_path = repeat_dir / "netdata.csv"

    for path in (
        summary_path,
        iterations_path,
        netdata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    iterations = load_csv(iterations_path)
    netdata = load_csv(netdata_path)

    if not iterations:
        raise RuntimeError(
            f"No iteration rows: {iterations_path}"
        )

    for row in netdata:
        row["_timestamp_ns"] = int(
            row["TimestampUnixNs"]
        )

    first_iter_start_ns = min(
        int(row["start_unix_ns"])
        for row in iterations
    )

    last_iter_end_ns = max(
        int(row["end_unix_ns"])
        for row in iterations
    )

    steady_iterations = [
        row
        for row in iterations
        if is_steady(row)
    ]

    if not steady_iterations:
        raise RuntimeError(
            f"No steady-state iterations: {iterations_path}"
        )

    steady_start_ns = min(
        int(row["start_unix_ns"])
        for row in steady_iterations
    )

    steady_end_ns = max(
        int(row["end_unix_ns"])
        for row in steady_iterations
    )

    pre_start_ns = (
        first_iter_start_ns
        - int(pre_window_s * 1_000_000_000)
    )

    pre_rows = [
        row
        for row in netdata
        if (
            pre_start_ns
            <= row["_timestamp_ns"]
            < first_iter_start_ns
        )
    ]

    all_runtime_rows = [
        row
        for row in netdata
        if (
            first_iter_start_ns
            <= row["_timestamp_ns"]
            <= last_iter_end_ns
        )
    ]

    steady_rows = [
        row
        for row in netdata
        if (
            steady_start_ns
            <= row["_timestamp_ns"]
            <= steady_end_ns
        )
    ]

    repeat_text = repeat_dir.name
    repeat = int(repeat_text.split("_")[-1])

    row = {
        "condition_id": condition_id,
        "device_id": device_id,
        "workload_id": workload_id,
        "repeat": repeat,
        "pre_window_s": pre_window_s,
        "pre_samples": len(pre_rows),
        "runtime_samples": len(all_runtime_rows),
        "steady_samples": len(steady_rows),

        # Ground-truth targets from the unprofiled baseline.
        "gt_steady_total_ms": float(
            summary["steady_window_total_ms"]
        ),
        "gt_mean_iter_ms": float(
            summary["steady_window_mean_iter_ms"]
        ),
        "gt_gpu_mean_ms": (
            float(summary["gpu_mean_ms"])
            if summary.get("gpu_mean_ms") is not None
            else math.nan
        ),
    }

    # Deployable pre-run features.
    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["gpu_util"],
            "pre_gpu_util_pct",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["gpu_mem"],
            "pre_gpu_mem_used_mb",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["gpu_power"],
            "pre_gpu_power_w",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["gpu_temp"],
            "pre_gpu_temp_c",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["cpu_user"],
            "pre_cpu_user_pct",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["cpu_system"],
            "pre_cpu_system_pct",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["cpu_iowait"],
            "pre_cpu_iowait_pct",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["load1"],
            "pre_load1",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["mem_used"],
            "pre_mem_used_mb",
        )
    )

    row.update(
        metric_stats(
            pre_rows,
            DEFAULT_FIELDS["top1_cpu"],
            "pre_top1_cpu_pct",
        )
    )

    # Diagnostic-only during-run features.
    # Keep these separate from deployable prediction inputs.
    row.update(
        metric_stats(
            steady_rows,
            DEFAULT_FIELDS["gpu_util"],
            "diag_steady_gpu_util_pct",
        )
    )

    row.update(
        metric_stats(
            steady_rows,
            DEFAULT_FIELDS["gpu_mem"],
            "diag_steady_gpu_mem_used_mb",
        )
    )

    row.update(
        metric_stats(
            steady_rows,
            DEFAULT_FIELDS["gpu_power"],
            "diag_steady_gpu_power_w",
        )
    )

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one-row-per-baseline-repeat data from "
            "ground-truth summaries and paired Netdata traces."
        )
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help=(
            "Condition root, e.g. runs_loaded/high_load_01"
        ),
    )

    parser.add_argument(
        "--device-id",
        required=True,
    )

    parser.add_argument(
        "--workloads",
        default="C01",
        help="Comma-separated workload IDs.",
    )

    parser.add_argument(
        "--pre-window",
        type=float,
        default=5.0,
        help=(
            "Seconds immediately before the first YOLO "
            "iteration used as current-load features."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.pre_window <= 0:
        raise ValueError("--pre-window must be > 0")

    runs_root = args.runs_root.resolve()
    condition_id = runs_root.name

    workloads = [
        item.strip()
        for item in args.workloads.split(",")
        if item.strip()
    ]

    rows: list[dict] = []

    for workload_id in workloads:
        workload_root = (
            runs_root
            / args.device_id
            / workload_id
        )

        repeat_dirs = sorted(
            p
            for p in workload_root.glob("baseline_*")
            if p.is_dir()
        )

        if not repeat_dirs:
            raise FileNotFoundError(
                f"No baseline_* directories found: "
                f"{workload_root}"
            )

        for repeat_dir in repeat_dirs:
            row = build_row(
                condition_id=condition_id,
                device_id=args.device_id,
                workload_id=workload_id,
                repeat_dir=repeat_dir,
                pre_window_s=args.pre_window,
            )

            rows.append(row)

    output = (
        args.output.resolve()
        if args.output is not None
        else runs_root / "baseline_netdata_dataset.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 78)
    print("Loaded baseline + Netdata dataset")
    print("=" * 78)
    print(f"Condition : {condition_id}")
    print(f"Device    : {args.device_id}")
    print(f"Rows      : {len(rows)}")
    print(f"Pre-window: {args.pre_window:.1f} s")
    print(f"Output    : {output}")
    print()

    runtimes = [
        float(row["gt_steady_total_ms"])
        for row in rows
    ]

    print(
        f"Runtime mean   : "
        f"{statistics.mean(runtimes):.3f} ms"
    )

    if len(runtimes) >= 2:
        runtime_std = statistics.stdev(runtimes)

        print(
            f"Runtime std    : "
            f"{runtime_std:.3f} ms"
        )

        print(
            f"Runtime CV     : "
            f"{runtime_std / statistics.mean(runtimes) * 100:.2f} %"
        )

    print()
    print(
        "repeat  pre_GPU_mean  pre_GPU_std  "
        "steady_GPU_mean  GT_runtime_ms"
    )

    for row in rows:
        print(
            f"{int(row['repeat']):>6d}  "
            f"{row['pre_gpu_util_pct_mean']:>12.2f}  "
            f"{row['pre_gpu_util_pct_std']:>11.2f}  "
            f"{row['diag_steady_gpu_util_pct_mean']:>15.2f}  "
            f"{row['gt_steady_total_ms']:>13.3f}"
        )

    low_sample_rows = [
        row
        for row in rows
        if int(row["pre_samples"]) < 3
    ]

    if low_sample_rows:
        print()
        print(
            "WARNING: Some repeats contain fewer than "
            "3 pre-run Netdata samples."
        )


if __name__ == "__main__":
    main()
