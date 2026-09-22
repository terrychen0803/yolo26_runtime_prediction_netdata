# YOLO26 Runtime Prediction + Netdata

This repository extends `yolo26_runtime_prediction` with synchronized node-level Netdata monitoring for YOLO26 dry-run / profiling experiments.

## Monitoring architecture

Each Nsight run can collect three complementary data streams:

1. **YOLO runtime ground truth** — `iterations.csv`
   - per-iteration host callback time
   - CUDA-event GPU time
   - `start_unix_ns` / `end_unix_ns` for time alignment

2. **Nsight Systems** — `profile.nsys-rep`
   - CUDA + NVTX timeline
   - optional GPU metrics
   - optional CPU PMU metrics

3. **Netdata** — `netdata.csv`
   - CPU User / System / IOWait
   - Load 1 / 5 / 15 min
   - RAM used / free
   - CPU temperature
   - GPU utilization
   - GPU framebuffer memory used
   - GPU temperature
   - GPU power
   - Top CPU application-group utilization
   - Top GPU process utilization columns are currently reserved and emitted as `NaN`

All streams are stored under the same run directory.

## Netdata requirement

Install and start a local Netdata Agent. The collector expects:

```text
http://127.0.0.1:19999
```

For NVIDIA metrics, configure the Netdata `nvidia_smi` collector to update every second:

```yaml
update_every: 1
autodetection_retry: 0

jobs:
  - name: nvidia_smi
```

Then restart Netdata:

```bash
sudo systemctl restart netdata
```

Verify that the collector is using 1-second sampling:

```bash
ps -eo pid,args | grep '[n]vidia-smi'
```

Expected pattern:

```text
/usr/bin/nvidia-smi -q -x -l 1
```

## Standalone Netdata test

```bash
python3 scripts/collect_netdata.py \
  --output /tmp/netdata_test.csv
```

Stop with `Ctrl+C`, then inspect:

```bash
head -5 /tmp/netdata_test.csv
```

## Nsight + Netdata integration

Example RTX 5090 GPU-metrics run:

```bash
python3 scripts/run_nsys.py \
  --device-id RTX5090 \
  --workloads C01 \
  --profile-mode gpu-metrics \
  --gpu-metrics-devices 0 \
  --gpu-metrics-frequency 1000 \
  --sudo-nsys \
  --netdata
```

Netdata options:

```text
--netdata
--netdata-url http://127.0.0.1:19999
--netdata-interval 1.0
--netdata-preroll 2.0
```

The sidecar starts before Nsight, collects pre-run node state, and is terminated automatically after profiling, including error paths.

## Output layout

Example:

```text
runs/RTX5090/C01/nsys_gpu_metrics_01/
├── profile.nsys-rep
├── nsys.log
├── profiling_config.json
├── iterations.csv
├── summary.json
├── metadata.json
├── netdata.csv
└── netdata.log
```

## Workloads

The original C01-C24 workload matrix is preserved:

- batch: 4 / 8 / 16 / 32
- image size: 320 / 480 / 640
- AMP: false / true
- max iterations: 128
- warm-up iterations: 20

## Notes

- Netdata system monitoring is intentionally 1 Hz.
- Nsight GPU metrics may run at a much higher frequency; the datasets are aligned later by timestamp instead of forcing a common sampling rate.
- `Top1 GPU%` and `Top2 GPU%` are placeholders for a future NVML per-process GPU-utilization collector.
- Before large profiling sweeps, verify available disk space because `.nsys-rep` files can be large.


## Ground-truth baseline + Netdata

Each baseline repeat can collect its own Netdata trace so the observed machine state is paired 1:1 with the ground-truth runtime.

```bash
python scripts/run_baseline.py \
  --device-id RTX5090 \
  --workloads C01 \
  --repeats 5 \
  --runs-root runs_loaded/high_load_01 \
  --netdata \
  --netdata-interval 1.0 \
  --netdata-preroll 5.0
```

Each repeat then contains:

```text
baseline_01/
├── iterations.csv
├── summary.json
├── metadata.json
├── baseline_config.json
├── run.log
├── netdata.csv
└── netdata.log
```

For prediction datasets, derive deployable load-state features from a fixed pre-run window (for example, the final 5 seconds before the first YOLO iteration). Keep during-run Netdata features separate as diagnostic features to avoid target leakage.
