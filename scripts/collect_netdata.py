#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_NETDATA_URL = "http://127.0.0.1:19999"
DEFAULT_INTERVAL = 1.0

FIELDS = [
    "Timestamp",
    "TimestampUnixNs",
    "CPU User%",
    "CPU System%",
    "CPU IOWait%",
    "Load 1min",
    "Load 5min",
    "Load 15min",
    "Mem Used(MB)",
    "Mem Free(MB)",
    "CPU Temp(°C)",
    "GPU Util%",
    "GPU Mem Used(MB)",
    "GPU Temp(°C)",
    "GPU Power(W)",
    "Top1 CPU%",
    "Top2 CPU%",
    "Top3 CPU%",
    "Top1 GPU%",
    "Top2 GPU%",
]

running = True

def stop_handler(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)

def fetch_allmetrics(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/api/v1/allmetrics?format=json"
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)

def dim(chart: dict | None, name: str, default: float = math.nan) -> float:
    if not chart:
        return default
    dimensions = chart.get("dimensions", {})
    info = dimensions.get(name)
    if info is None:
        return default
    value = info.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def find_chart_by_context(metrics: dict, context: str) -> dict | None:
    matches = [
        chart
        for chart in metrics.values()
        if chart.get("context") == context
    ]
    return matches[0] if matches else None

def get_cpu_temperature(metrics: dict) -> float:
    preferred = []
    fallback = []
    for chart_id, chart in metrics.items():
        name = chart_id.lower()
        if not (
            name.startswith("sensors.temperature_")
            and name.endswith("_input")
        ):
            continue
        value = dim(chart, "input")
        if math.isnan(value):
            continue
        if "k10temp" in name and "_tctl_input" in name:
            preferred.append(value)
        elif "k10temp" in name:
            fallback.append(value)
    if preferred:
        return preferred[0]
    if fallback:
        return max(fallback)
    return math.nan

def get_top_cpu(metrics: dict) -> list[float]:
    usage = []
    for chart_id, chart in metrics.items():
        if not (
            chart_id.startswith("app.")
            and chart_id.endswith("_cpu_utilization")
        ):
            continue
        user = dim(chart, "user", 0.0)
        system = dim(chart, "system", 0.0)
        usage.append(user + system)
    usage.sort(reverse=True)
    while len(usage) < 3:
        usage.append(0.0)
    return usage[:3]

def collect(metrics: dict) -> dict:
    cpu = metrics.get("system.cpu", {})
    load = metrics.get("system.load", {})
    ram = metrics.get("system.ram", {})

    gpu_util_chart = find_chart_by_context(
        metrics, "nvidia_smi.gpu_utilization"
    )
    gpu_memory_chart = find_chart_by_context(
        metrics, "nvidia_smi.gpu_frame_buffer_memory_usage"
    )
    gpu_temp_chart = find_chart_by_context(
        metrics, "nvidia_smi.gpu_temperature"
    )
    gpu_power_chart = find_chart_by_context(
        metrics, "nvidia_smi.gpu_power_draw"
    )

    gpu_memory_used_bytes = dim(gpu_memory_chart, "used")
    gpu_memory_used_mb = (
        math.nan
        if math.isnan(gpu_memory_used_bytes)
        else gpu_memory_used_bytes / 1024.0 / 1024.0
    )

    top_cpu = get_top_cpu(metrics)
    now_ns = time.time_ns()

    return {
        "Timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "TimestampUnixNs": now_ns,
        "CPU User%": dim(cpu, "user"),
        "CPU System%": dim(cpu, "system"),
        "CPU IOWait%": dim(cpu, "iowait"),
        "Load 1min": dim(load, "load1"),
        "Load 5min": dim(load, "load5"),
        "Load 15min": dim(load, "load15"),
        "Mem Used(MB)": dim(ram, "used"),
        "Mem Free(MB)": dim(ram, "free"),
        "CPU Temp(°C)": get_cpu_temperature(metrics),
        "GPU Util%": dim(gpu_util_chart, "gpu"),
        "GPU Mem Used(MB)": gpu_memory_used_mb,
        "GPU Temp(°C)": dim(gpu_temp_chart, "temperature"),
        "GPU Power(W)": dim(gpu_power_chart, "power_draw"),
        "Top1 CPU%": top_cpu[0],
        "Top2 CPU%": top_cpu[1],
        "Top3 CPU%": top_cpu[2],
        "Top1 GPU%": math.nan,
        "Top2 GPU%": math.nan,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect node-level system metrics from a local Netdata Agent."
    )
    parser.add_argument("--output", type=Path, default=Path("netdata.csv"))
    parser.add_argument("--url", default=DEFAULT_NETDATA_URL)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    args = parser.parse_args()

    if args.interval < 1.0:
        raise ValueError("Netdata collection interval should be >= 1 second.")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists() or output.stat().st_size == 0

    print(f"[NETDATA] URL      : {args.url}", flush=True)
    print(f"[NETDATA] Output   : {output}", flush=True)
    print(f"[NETDATA] Interval : {args.interval:.1f} s", flush=True)

    try:
        metrics = fetch_allmetrics(args.url)
    except Exception as exc:
        raise SystemExit(f"Cannot connect to Netdata: {exc}")

    required_contexts = [
        "nvidia_smi.gpu_utilization",
        "nvidia_smi.gpu_frame_buffer_memory_usage",
        "nvidia_smi.gpu_temperature",
        "nvidia_smi.gpu_power_draw",
    ]
    for context in required_contexts:
        found = find_chart_by_context(metrics, context) is not None
        print(
            f"[NETDATA] {context}: {'OK' if found else 'NOT FOUND'}",
            flush=True,
        )

    with output.open("a", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        next_time = time.monotonic()
        global running

        while running:
            try:
                metrics = fetch_allmetrics(args.url)
                row = collect(metrics)
                writer.writerow(row)
                print(
                    row["Timestamp"],
                    f'CPU={row["CPU User%"]:.1f}%',
                    f'GPU={row["GPU Util%"]:.1f}%',
                    f'VRAM={row["GPU Mem Used(MB)"]:.0f}MB',
                    f'Power={row["GPU Power(W)"]:.1f}W',
                    f'GPU Temp={row["GPU Temp(°C)"]:.1f}C',
                    flush=True,
                )
            except Exception as exc:
                print(f"[NETDATA] collection error: {exc}", file=sys.stderr, flush=True)

            next_time += args.interval
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = time.monotonic()

    print("[NETDATA] Collection stopped.", flush=True)

if __name__ == "__main__":
    main()
