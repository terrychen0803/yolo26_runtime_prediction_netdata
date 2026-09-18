from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic_yolo"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


def str_to_bool(value: str) -> bool:
    value = value.strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Invalid boolean value: {value}")


def sanitize_name(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

    result = "".join(
        ch if ch in allowed else "_"
        for ch in text
    )

    return result.strip("_") or "unknown"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return values[low]

    fraction = position - low

    return (
        values[low] * (1.0 - fraction)
        + values[high] * fraction
    )


def load_workload(
    csv_path: Path,
    workload_id: str,
) -> dict:
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["workload_id"] == workload_id:
                return {
                    "workload_id": row["workload_id"],
                    "model": row["model"],
                    "batch": int(row["batch"]),
                    "imgsz": int(row["imgsz"]),
                    "amp": str_to_bool(row["amp"]),
                    "max_iters": int(row["max_iters"]),
                    "warmup_iters": int(row["warmup_iters"]),
                    "nbs": int(row["nbs"]),
                }

    raise ValueError(
        f"Workload {workload_id!r} not found in {csv_path}"
    )


def create_runtime_data_yaml(
    dataset_root: Path,
    output_path: Path,
) -> None:
    dataset_posix = dataset_root.resolve().as_posix()

    content = (
        f'path: "{dataset_posix}"\n'
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: synthetic_object\n"
    )

    output_path.write_text(
        content,
        encoding="utf-8",
    )


class IterationRecorder:
    """
    Measure a fixed steady-state training window.

    Definitions
    -----------
    warm-up:
        iteration_id <= warmup_iters

    valid steady-state window:
        warmup_iters < iteration_id <= max_iters

    Primary ground truth:
        synchronized wall-clock duration of the complete steady-state
        window divided by the number of valid iterations.

    CUDA events:
        recorded per iteration without synchronizing after each batch.
        They are resolved only once at the end of the steady-state window.
    """

    def __init__(
        self,
        torch_module,
        workload: dict,
        output_dir: Path,
        device_id: str,
        run_type: str,
        repeat: int,
        enable_nvtx: bool,
    ) -> None:
        self.torch = torch_module
        self.workload = workload
        self.output_dir = output_dir
        self.device_id = device_id
        self.run_type = run_type
        self.repeat = repeat
        self.enable_nvtx = enable_nvtx

        self.iteration_id = 0
        self.batch_in_epoch = 0

        self.current_host_start_ns: int | None = None
        self.current_wall_start_ns: int | None = None
        self.current_cuda_start = None

        self.records: list[dict] = []
        self.cuda_event_pairs: list[tuple] = []

        self.window_start_ns: int | None = None
        self.window_end_ns: int | None = None

        self.finalized = False

    @property
    def cuda_enabled(self) -> bool:
        return self.torch.cuda.is_available()

    def on_train_epoch_start(self, trainer) -> None:
        self.batch_in_epoch = 0

    def on_train_batch_start(self, trainer) -> None:
        self.iteration_id += 1
        self.batch_in_epoch += 1

        requested_batch = self.workload["batch"]
        actual_batch = int(trainer.batch_size)

        # A runtime OOM may cause recent Ultralytics versions to
        # automatically rebuild the training pipeline at a smaller batch.
        # That would invalidate the workload identity, so fail explicitly.
        if actual_batch != requested_batch:
            raise RuntimeError(
                "Actual batch size changed during training: "
                f"requested={requested_batch}, actual={actual_batch}. "
                "This run must not be used in the prediction dataset."
            )

        # Start the clean steady-state timing window only after warm-up.
        if self.iteration_id == self.workload["warmup_iters"] + 1:
            if self.cuda_enabled:
                self.torch.cuda.synchronize()

            self.window_start_ns = time.perf_counter_ns()

        if self.enable_nvtx and self.cuda_enabled:
            self.torch.cuda.nvtx.range_push(
                f"TRAIN_ITER_{self.iteration_id:04d}"
            )

        self.current_host_start_ns = time.perf_counter_ns()
        self.current_wall_start_ns = time.time_ns()

        if self.cuda_enabled:
            self.current_cuda_start = self.torch.cuda.Event(
                enable_timing=True
            )
            self.current_cuda_start.record()

    def on_train_batch_end(self, trainer) -> None:
        host_end_ns = time.perf_counter_ns()
        wall_end_ns = time.time_ns()

        cuda_end = None

        if self.cuda_enabled:
            cuda_end = self.torch.cuda.Event(
                enable_timing=True
            )
            cuda_end.record()

        if self.enable_nvtx and self.cuda_enabled:
            self.torch.cuda.nvtx.range_pop()

        warmup = (
            self.iteration_id
            <= self.workload["warmup_iters"]
        )

        host_callback_ms = (
            host_end_ns - self.current_host_start_ns
        ) / 1e6

        record = {
            "workload_id": self.workload["workload_id"],
            "device_id": self.device_id,
            "run_type": self.run_type,
            "repeat": self.repeat,
            "iteration_id": self.iteration_id,
            "epoch": int(trainer.epoch),
            "batch_in_epoch": self.batch_in_epoch,
            "warmup": warmup,
            "requested_batch": self.workload["batch"],
            "actual_batch": int(trainer.batch_size),
            "imgsz": self.workload["imgsz"],
            "amp": self.workload["amp"],
            "start_unix_ns": self.current_wall_start_ns,
            "end_unix_ns": wall_end_ns,
            "host_callback_ms": host_callback_ms,
            "gpu_ms": None,
        }

        self.records.append(record)

        if self.cuda_enabled:
            self.cuda_event_pairs.append(
                (
                    len(self.records) - 1,
                    self.current_cuda_start,
                    cuda_end,
                )
            )

        # Finish after exactly max_iters training batches.
        if self.iteration_id >= self.workload["max_iters"]:
            if self.cuda_enabled:
                self.torch.cuda.synchronize()

            self.window_end_ns = time.perf_counter_ns()

            self.finalize()

            # Current Ultralytics checks trainer.stop immediately after
            # on_train_batch_end and breaks cleanly between batches.
            trainer.stop = True

    def finalize(self) -> None:
        if self.finalized:
            return

        if self.cuda_enabled:
            self.torch.cuda.synchronize()

            for (
                record_index,
                start_event,
                end_event,
            ) in self.cuda_event_pairs:
                self.records[record_index]["gpu_ms"] = (
                    start_event.elapsed_time(end_event)
                )

        self._write_iteration_csv()
        self._write_summary()

        self.finalized = True

    def _write_iteration_csv(self) -> None:
        output_path = self.output_dir / "iterations.csv"

        fieldnames = [
            "workload_id",
            "device_id",
            "run_type",
            "repeat",
            "iteration_id",
            "epoch",
            "batch_in_epoch",
            "warmup",
            "requested_batch",
            "actual_batch",
            "imgsz",
            "amp",
            "start_unix_ns",
            "end_unix_ns",
            "host_callback_ms",
            "gpu_ms",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(self.records)

    def _write_summary(self) -> None:
        valid_records = [
            row
            for row in self.records
            if not row["warmup"]
            and row["iteration_id"]
            <= self.workload["max_iters"]
        ]

        gpu_values = [
            float(row["gpu_ms"])
            for row in valid_records
            if row["gpu_ms"] is not None
        ]

        host_values = [
            float(row["host_callback_ms"])
            for row in valid_records
        ]

        valid_count = len(valid_records)

        window_total_ms = None
        window_mean_iter_ms = None

        if (
            self.window_start_ns is not None
            and self.window_end_ns is not None
            and valid_count > 0
        ):
            window_total_ms = (
                self.window_end_ns
                - self.window_start_ns
            ) / 1e6

            window_mean_iter_ms = (
                window_total_ms / valid_count
            )

        summary = {
            "workload_id": self.workload["workload_id"],
            "device_id": self.device_id,
            "run_type": self.run_type,
            "repeat": self.repeat,

            "model": self.workload["model"],
            "batch": self.workload["batch"],
            "nbs": self.workload["nbs"],
            "imgsz": self.workload["imgsz"],
            "amp": self.workload["amp"],

            "max_iters": self.workload["max_iters"],
            "warmup_iters": self.workload["warmup_iters"],
            "valid_iters": valid_count,

            # Primary cross-device ground-truth candidate.
            "steady_window_total_ms": window_total_ms,
            "steady_window_mean_iter_ms": window_mean_iter_ms,

            # CPU-side callback duration. This does not force CUDA sync.
            "host_callback_mean_ms": (
                statistics.mean(host_values)
                if host_values
                else None
            ),
            "host_callback_median_ms": (
                statistics.median(host_values)
                if host_values
                else None
            ),

            # Device-side per-iteration timings from CUDA Events.
            "gpu_mean_ms": (
                statistics.mean(gpu_values)
                if gpu_values
                else None
            ),
            "gpu_median_ms": (
                statistics.median(gpu_values)
                if gpu_values
                else None
            ),
            "gpu_std_ms": (
                statistics.stdev(gpu_values)
                if len(gpu_values) >= 2
                else None
            ),
            "gpu_p90_ms": (
                percentile(gpu_values, 0.90)
                if gpu_values
                else None
            ),
            "gpu_p95_ms": (
                percentile(gpu_values, 0.95)
                if gpu_values
                else None
            ),
        }

        output_path = self.output_dir / "summary.json"

        output_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO26 training wrapper for cross-hardware "
            "iteration runtime prediction."
        )
    )

    parser.add_argument(
        "--workload-id",
        required=True,
        help="Workload ID such as C01.",
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="Logical hardware label, e.g. RTX5090.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device argument. Default: 0",
    )

    parser.add_argument(
        "--run-type",
        choices=[
            "baseline",
            "nsys_trace",
            "nsys_gpu_metrics",
            "nsys_cpu_metrics",
            "hpctoolkit_cuda",
            "hpctoolkit_cuda_trace",
        ],
        default="baseline",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit TRAIN_ITER_xxxx NVTX ranges.",
    )

    parser.add_argument(
        "--experiments",
        type=Path,
        default=DEFAULT_EXPERIMENTS,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workload = load_workload(
        args.experiments.resolve(),
        args.workload_id,
    )

    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_root}"
        )

    device_label = sanitize_name(args.device_id)

    run_name = (
        f"{args.run_type}_{args.repeat:02d}"
    )

    output_dir = (
        args.runs_root.resolve()
        / device_label
        / workload["workload_id"]
        / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_yaml = (
        output_dir
        / "data.runtime.yaml"
    )

    create_runtime_data_yaml(
        dataset_root,
        runtime_yaml,
    )

    # Import heavy GPU/framework dependencies only when an actual
    # training run is requested. This lets the script be syntax-checked
    # and --help tested on the local development laptop.
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run the actual experiment on a GPU node."
        )

    recorder = IterationRecorder(
        torch_module=torch,
        workload=workload,
        output_dir=output_dir,
        device_id=device_label,
        run_type=args.run_type,
        repeat=args.repeat,
        enable_nvtx=args.nvtx,
    )

    model = YOLO(
        workload["model"]
    )

    model.add_callback(
        "on_train_epoch_start",
        recorder.on_train_epoch_start,
    )

    model.add_callback(
        "on_train_batch_start",
        recorder.on_train_batch_start,
    )

    model.add_callback(
        "on_train_batch_end",
        recorder.on_train_batch_end,
    )

    metadata = {
        "workload": workload,
        "device_id": device_label,
        "device_argument": args.device,
        "run_type": args.run_type,
        "repeat": args.repeat,
        "nvtx": args.nvtx,

        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,

        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),

        "dataset_root": str(dataset_root),
    }

    (
        output_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("YOLO26 Runtime Prediction Experiment")
    print("=" * 70)
    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )
    print("=" * 70)

    try:
        model.train(
            data=str(runtime_yaml),

            model=workload["model"],

            epochs=20,

            imgsz=workload["imgsz"],
            batch=workload["batch"],
            nbs=workload["nbs"],

            workers=0,
            device=args.device,

            pretrained=False,
            amp=workload["amp"],

            optimizer="AdamW",

            warmup_epochs=0.0,

            cache="ram",

            val=False,
            save=False,
            plots=False,

            multi_scale=0.0,

            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,

            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,

            flipud=0.0,
            fliplr=0.0,

            seed=0,
            deterministic=True,

            compile=False,

            project=str(output_dir),
            name="ultralytics",
            exist_ok=True,
        )

    except FileNotFoundError as exc:
        message = str(exc)

        expected_checkpoint_error = (
            "Training completed but no checkpoint was saved."
            in message
        )

        completed_expected_iterations = (
            recorder.iteration_id
            >= workload["max_iters"]
        )

        if (
            expected_checkpoint_error
            and completed_expected_iterations
        ):
            print()
            print(
                "NOTE: Ultralytics attempted to reload a "
                "checkpoint after training."
            )
            print(
                "Checkpoint saving is intentionally disabled "
                "for this performance benchmark."
            )
            print(
                "The requested iteration window completed "
                "successfully; ignoring this expected error."
            )
        else:
            raise

    finally:
        recorder.finalize()

    print()
    print("Experiment complete")
    print(f"Output: {output_dir}")
    print(f"Iterations: {output_dir / 'iterations.csv'}")
    print(f"Summary:    {output_dir / 'summary.json'}")
    print(f"Metadata:   {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()