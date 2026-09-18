from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "capabilities"


def run_command(command: list[str]) -> dict:
    """
    Run command and preserve stdout/stderr/return code.

    Some Nsight help commands may print useful information to stderr,
    so both streams are retained.
    """

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        return {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    except Exception as exc:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def combined_output(result: dict) -> str:
    parts = []

    if result["stdout"]:
        parts.append(result["stdout"])

    if result["stderr"]:
        parts.append(result["stderr"])

    return "\n".join(parts)


def save_text(path: Path, result: dict) -> None:
    content = (
        f"COMMAND:\n"
        f"{' '.join(result['command'])}\n\n"
        f"RETURN CODE:\n"
        f"{result['return_code']}\n\n"
        f"STDOUT:\n"
        f"{result['stdout']}\n\n"
        f"STDERR:\n"
        f"{result['stderr']}\n"
    )

    path.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Nsight Systems GPU/CPU profiling "
            "capabilities on a target node."
        )
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="Logical node/device label, e.g. RTX5090.",
    )

    parser.add_argument(
        "--gpu-device",
        default="0",
        help="GPU index to inspect. Default: 0.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    if shutil.which("nsys") is None:
        raise SystemExit(
            "ERROR: 'nsys' was not found in PATH."
        )

    output_dir = (
        args.output_root.resolve()
        / args.device_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Nsight Systems Capability Discovery")
    print("=" * 72)
    print(f"Device ID  : {args.device_id}")
    print(f"GPU index  : {args.gpu_device}")
    print(f"Output     : {output_dir}")
    print("=" * 72)

    # ----------------------------------------------------------
    # Basic environment
    # ----------------------------------------------------------

    commands = {
        "nsys_version": [
            "nsys",
            "--version",
        ],

        "nvidia_smi": [
            "nvidia-smi",
        ],

        # List GPUs supported by Nsight GPU Metrics.
        "gpu_metrics_devices": [
            "nsys",
            "profile",
            "--gpu-metrics-devices=help",
        ],

        # Metric sets depend on selected GPU architecture.
        "gpu_metrics_sets": [
            "nsys",
            "profile",
            f"--gpu-metrics-devices={args.gpu_device}",
            "--gpu-metrics-set=help",
        ],

        # Compact CPU capabilities.
        "cpu_metrics_help": [
            "nsys",
            "profile",
            "--cpu-metrics=help",
        ],

        # Complete CPU core/uncore/derived metric listing.
        "cpu_metrics_help_all": [
            "nsys",
            "profile",
            "--cpu-metrics=help:all",
        ],
    }

    results = {}

    for name, command in commands.items():

        print()
        print("-" * 72)
        print(f"[{name}]")
        print(" ".join(command))
        print("-" * 72)

        result = run_command(command)

        results[name] = result

        text = combined_output(result)

        if text:
            print(text)
        else:
            print("(no output)")

        save_text(
            output_dir / f"{name}.txt",
            result,
        )

    # ----------------------------------------------------------
    # Save machine-readable summary
    # ----------------------------------------------------------

    summary = {
        "device_id": args.device_id,
        "gpu_device_index": args.gpu_device,

        "timestamp_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),

        "platform": platform.platform(),
        "machine": platform.machine(),

        "nsys_version": combined_output(
            results["nsys_version"]
        ),

        "commands": results,
    }

    (
        output_dir
        / "capabilities.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("CAPABILITY DISCOVERY COMPLETE")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print()
    print("Files:")

    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()