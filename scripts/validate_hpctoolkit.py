from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "capabilities"


CUDA_SOURCE = r"""
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(
        d_a,
        h_a,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        h_b,
        bytes,
        cudaMemcpyHostToDevice
    );

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Repeat enough times to generate meaningful GPU activity.
    for (int r = 0; r < 100; ++r) {
        vector_add<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            n
        );
    }

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    printf("result = %.1f\n", h_c[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
"""


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
) -> int:

    print()
    print("$ " + " ".join(command))

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


def resolve_binary(
    name: str,
    hpct_root: Path | None,
) -> str | None:

    if hpct_root is not None:

        candidate = hpct_root / "bin" / name

        if candidate.exists():
            return str(candidate)

    return shutil.which(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate HPCToolkit CUDA profiling "
            "on a target GPU."
        )
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="Logical GPU label, e.g. RTX5090.",
    )

    parser.add_argument(
        "--hpct-root",
        type=Path,
        default=None,
        help="HPCToolkit installation root.",
    )

    parser.add_argument(
        "--nvcc",
        default=None,
        help=(
            "nvcc path. If omitted, search PATH."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--test-pc",
        action="store_true",
        help=(
            "Also test gpu=cuda,pc. "
            "Do not enable on a device with a known "
            "PC-sampling incompatibility."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    hpct_root = (
        args.hpct_root.resolve()
        if args.hpct_root
        else None
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

    nvcc = (
        args.nvcc
        if args.nvcc
        else shutil.which("nvcc")
    )

    required = {
        "hpcrun": hpcrun,
        "hpcstruct": hpcstruct,
        "hpcprof": hpcprof,
        "nvcc": nvcc,
    }

    missing = [
        name
        for name, path in required.items()
        if path is None
    ]

    if missing:
        raise SystemExit(
            "Missing required tools: "
            + ", ".join(missing)
        )

    output_dir = (
        args.output_root.resolve()
        / args.device_id
        / "hpctoolkit"
        / "validation"
    )

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_file = (
        output_dir / "cuda_smoke.cu"
    )

    binary_file = (
        output_dir / "cuda_smoke"
    )

    source_file.write_text(
        CUDA_SOURCE,
        encoding="utf-8",
    )

    results = {
        "device_id": args.device_id,
        "timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),

        "hpct_root": (
            str(hpct_root)
            if hpct_root
            else None
        ),

        "hpcrun": hpcrun,
        "hpcstruct": hpcstruct,
        "hpcprof": hpcprof,
        "nvcc": nvcc,

        "compile_pass": False,
        "native_cuda_pass": False,

        "gpu_cuda_pass": False,
        "gpu_cuda_struct_pass": False,
        "gpu_cuda_prof_pass": False,

        "gpu_cuda_pc_tested": args.test_pc,
        "gpu_cuda_pc_pass": None,
    }

    # ==========================================================
    # 1. Compile
    # ==========================================================

    compile_cmd = [
        nvcc,
        "-O2",
        "-lineinfo",
        str(source_file),
        "-o",
        str(binary_file),
    ]

    rc = run_command(
        compile_cmd,
        output_dir,
        output_dir / "compile.log",
    )

    results["compile_pass"] = (
        rc == 0 and binary_file.exists()
    )

    if not results["compile_pass"]:
        save_results(output_dir, results)
        raise SystemExit("CUDA compilation failed.")

    # ==========================================================
    # 2. Native CUDA
    # ==========================================================

    rc = run_command(
        [str(binary_file)],
        output_dir,
        output_dir / "native.log",
    )

    results["native_cuda_pass"] = (
        rc == 0
    )

    if not results["native_cuda_pass"]:
        save_results(output_dir, results)
        raise SystemExit(
            "Native CUDA smoke test failed."
        )

    # ==========================================================
    # 3. HPCToolkit gpu=cuda
    # ==========================================================

    measurements = (
        output_dir
        / "hpctoolkit-cuda-measurements"
    )

    cuda_cmd = [
        hpcrun,
        "-e",
        "gpu=cuda",
        "-o",
        str(measurements),
        str(binary_file),
    ]

    rc = run_command(
        cuda_cmd,
        output_dir,
        output_dir / "hpcrun_cuda.log",
    )

    results["gpu_cuda_pass"] = (
        rc == 0
        and measurements.exists()
    )

    # ==========================================================
    # 4. hpcstruct
    # ==========================================================

    if results["gpu_cuda_pass"]:

        struct_cmd = [
            hpcstruct,
            str(measurements),
        ]

        rc = run_command(
            struct_cmd,
            output_dir,
            output_dir / "hpcstruct.log",
        )

        results["gpu_cuda_struct_pass"] = (
            rc == 0
        )

    # ==========================================================
    # 5. hpcprof
    # ==========================================================

    database = (
        output_dir
        / "hpctoolkit-cuda-database"
    )

    if results["gpu_cuda_struct_pass"]:

        prof_cmd = [
            hpcprof,
            "-o",
            str(database),
            str(measurements),
        ]

        rc = run_command(
            prof_cmd,
            output_dir,
            output_dir / "hpcprof.log",
        )

        results["gpu_cuda_prof_pass"] = (
            rc == 0
            and database.exists()
        )

    # ==========================================================
    # 6. Optional PC sampling
    # ==========================================================

    if args.test_pc:

        pc_measurements = (
            output_dir
            / "hpctoolkit-cuda-pc-measurements"
        )

        pc_cmd = [
            hpcrun,
            "-e",
            "gpu=cuda,pc",
            "-o",
            str(pc_measurements),
            str(binary_file),
        ]

        rc = run_command(
            pc_cmd,
            output_dir,
            output_dir / "hpcrun_cuda_pc.log",
        )

        results["gpu_cuda_pc_pass"] = (
            rc == 0
            and pc_measurements.exists()
        )

    save_results(
        output_dir,
        results,
    )

    print()
    print("=" * 72)
    print("HPCToolkit Runtime Validation")
    print("=" * 72)

    print(
        f"Compile             : "
        f"{results['compile_pass']}"
    )

    print(
        f"Native CUDA         : "
        f"{results['native_cuda_pass']}"
    )

    print(
        f"gpu=cuda            : "
        f"{results['gpu_cuda_pass']}"
    )

    print(
        f"hpcstruct           : "
        f"{results['gpu_cuda_struct_pass']}"
    )

    print(
        f"hpcprof             : "
        f"{results['gpu_cuda_prof_pass']}"
    )

    if args.test_pc:
        print(
            f"gpu=cuda,pc         : "
            f"{results['gpu_cuda_pc_pass']}"
        )
    else:
        print(
            "gpu=cuda,pc         : "
            "NOT TESTED"
        )

    print("=" * 72)

    overall_pass = (
        results["compile_pass"]
        and results["native_cuda_pass"]
        and results["gpu_cuda_pass"]
        and results["gpu_cuda_struct_pass"]
        and results["gpu_cuda_prof_pass"]
    )

    if overall_pass:
        print("RESULT: HPCToolkit CUDA validation PASS")
    else:
        print("RESULT: HPCToolkit CUDA validation FAIL")
        raise SystemExit(1)


def save_results(
    output_dir: Path,
    results: dict,
) -> None:

    (
        output_dir
        / "validation.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()