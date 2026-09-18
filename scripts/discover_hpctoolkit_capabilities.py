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


def resolve_binary(
    name: str,
    hpct_root: Path | None,
) -> str | None:

    if hpct_root is not None:
        candidate = hpct_root / "bin" / name

        if candidate.exists():
            return str(candidate)

    return shutil.which(name)


def save_result(
    path: Path,
    result: dict,
) -> None:

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
            "Discover HPCToolkit CPU/GPU profiling "
            "capabilities on a target node."
        )
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="Logical device label, e.g. RTX5090.",
    )

    parser.add_argument(
        "--hpct-root",
        type=Path,
        default=None,
        help=(
            "HPCToolkit installation root. "
            "Example: $(spack location -i /HASH)"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    hpct_root = (
        args.hpct_root.resolve()
        if args.hpct_root is not None
        else None
    )

    output_dir = (
        args.output_root.resolve()
        / args.device_id
        / "hpctoolkit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    print("=" * 72)
    print("HPCToolkit Capability Discovery")
    print("=" * 72)
    print(f"Device ID : {args.device_id}")
    print(f"HPCT root : {hpct_root}")
    print(f"hpcrun    : {hpcrun}")
    print(f"hpcstruct : {hpcstruct}")
    print(f"hpcprof   : {hpcprof}")
    print("=" * 72)

    installed = all(
        binary is not None
        for binary in (
            hpcrun,
            hpcstruct,
            hpcprof,
        )
    )

    # ----------------------------------------------------------
    # HPCToolkit not installed
    # ----------------------------------------------------------

    if not installed:

        summary = {
            "device_id": args.device_id,
            "timestamp_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "platform": platform.platform(),
            "machine": platform.machine(),

            "installed": False,

            "hpct_root": (
                str(hpct_root)
                if hpct_root
                else None
            ),

            "hpcrun": hpcrun,
            "hpcstruct": hpcstruct,
            "hpcprof": hpcprof,

            "gpu_cuda_listed": False,
            "gpu_cuda_trace_possible": False,
            "gpu_cuda_pc_listed": False,
        }

        (
            output_dir / "capabilities.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("HPCToolkit installation incomplete or unavailable.")
        print(
            "Capability record saved, but no profiling "
            "capability is assumed."
        )

        return

    # ----------------------------------------------------------
    # Capability queries
    # ----------------------------------------------------------

    commands = {
        "hpcrun_version": [
            hpcrun,
            "--version",
        ],

        "hpcrun_events": [
            hpcrun,
            "-L",
        ],

        "hpcstruct_version": [
            hpcstruct,
            "--version",
        ],

        "hpcprof_version": [
            hpcprof,
            "--version",
        ],

        "nvidia_smi": [
            "nvidia-smi",
        ],
    }

    nvcc = shutil.which("nvcc")

    if nvcc:
        commands["nvcc_version"] = [
            nvcc,
            "--version",
        ]

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

        save_result(
            output_dir / f"{name}.txt",
            result,
        )

    # ----------------------------------------------------------
    # Parse hpcrun -L
    # ----------------------------------------------------------

    event_text = combined_output(
        results["hpcrun_events"]
    ).lower()

    gpu_cuda_listed = (
        "gpu=cuda" in event_text
    )

    gpu_cuda_pc_listed = (
        "gpu=cuda,pc" in event_text
    )

    hpcrun_version_text = combined_output(
        results["hpcrun_version"]
    )

    summary = {
        "device_id": args.device_id,

        "timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),

        "platform": platform.platform(),
        "machine": platform.machine(),

        "installed": True,

        "hpct_root": (
            str(hpct_root)
            if hpct_root
            else None
        ),

        "hpcrun": hpcrun,
        "hpcstruct": hpcstruct,
        "hpcprof": hpcprof,

        "hpcrun_version_output": (
            hpcrun_version_text
        ),

        # Listed means the installed HPCToolkit build exposes
        # the event. It does NOT prove that the target GPU can
        # successfully execute the mode.
        "gpu_cuda_listed": gpu_cuda_listed,

        "gpu_cuda_trace_possible": gpu_cuda_listed,

        "gpu_cuda_pc_listed": gpu_cuda_pc_listed,

        # Runtime validation must still be performed separately.
        "gpu_cuda_runtime_validated": None,
        "gpu_cuda_pc_runtime_validated": None,

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
    print("HPCTOOLKIT CAPABILITY DISCOVERY COMPLETE")
    print("=" * 72)

    print(
        f"gpu=cuda listed    : "
        f"{gpu_cuda_listed}"
    )

    print(
        f"gpu=cuda,pc listed : "
        f"{gpu_cuda_pc_listed}"
    )

    print()
    print(
        "NOTE: 'listed' only means hpcrun exposes the mode."
    )

    print(
        "A CUDA smoke test is still required to validate "
        "actual GPU/runtime compatibility."
    )

    print()
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()