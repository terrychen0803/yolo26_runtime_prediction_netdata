from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STACK_FILE = PROJECT_ROOT / "configs" / "software_stack.json"


def run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Also require CUDA/GPU environment.",
    )

    args = parser.parse_args()

    expected = json.loads(
        STACK_FILE.read_text(encoding="utf-8-sig")
    )

    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    print(f"Platform : {platform.platform()}")
    print(f"Python   : {platform.python_version()}")

    expected_python = expected["python_major_minor"]

    actual_python = ".".join(
        platform.python_version().split(".")[:2]
    )

    python_ok = actual_python == expected_python

    print(
        f"Python target : {expected_python} "
        f"[{'OK' if python_ok else 'MISMATCH'}]"
    )

    print()

    # Ultralytics
    try:
        import ultralytics

        actual_ultralytics = ultralytics.__version__

        ultra_ok = (
            actual_ultralytics
            == expected["ultralytics_version"]
        )

        print(
            f"Ultralytics : {actual_ultralytics} "
            f"[{'OK' if ultra_ok else 'MISMATCH'}]"
        )

    except Exception as e:
        actual_ultralytics = None
        ultra_ok = False
        print(f"Ultralytics : NOT INSTALLED ({e})")

    # PyTorch
    try:
        import torch

        actual_torch = torch.__version__
        actual_torch_cuda = torch.version.cuda

        torch_ok = (
            actual_torch
            == expected["torch_version"]
        )

        torch_cuda_ok = (
            actual_torch_cuda
            == expected["torch_cuda_version"]
        )

        print(
            f"PyTorch     : {actual_torch} "
            f"[{'OK' if torch_ok else 'MISMATCH'}]"
        )

        print(
            f"PyTorch CUDA: {actual_torch_cuda} "
            f"[{'OK' if torch_cuda_ok else 'MISMATCH'}]"
        )

        cuda_available = torch.cuda.is_available()

        print(f"CUDA usable : {cuda_available}")

        if cuda_available:
            print(
                f"GPU         : "
                f"{torch.cuda.get_device_name(0)}"
            )

    except Exception as e:
        actual_torch = None
        actual_torch_cuda = None
        torch_ok = False
        torch_cuda_ok = False
        cuda_available = False

        print(f"PyTorch     : NOT INSTALLED ({e})")

    print()

    nvidia_smi = run_command(["nvidia-smi"])

    if nvidia_smi:
        print("nvidia-smi:")
        print(nvidia_smi)
    else:
        print("nvidia-smi: unavailable")

    print()
    print("=" * 70)

    if args.gpu:
        passed = (
            python_ok
            and ultra_ok
            and torch_ok
            and torch_cuda_ok
            and cuda_available
        )

        if passed:
            print("RESULT: GPU experiment environment PASS")
        else:
            print("RESULT: GPU experiment environment FAIL")
            sys.exit(1)

    else:
        print(
            "RESULT: development-machine check complete "
            "(GPU environment not required)"
        )


if __name__ == "__main__":
    main()