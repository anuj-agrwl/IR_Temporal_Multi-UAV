#!/usr/bin/env python3
"""Measure the computational profile of the segmentation model reported in the paper.

MEASUREMENT DISCIPLINE -- the reason a naive timing loop over-reports throughput:
  * CUDA is asynchronous. Without torch.cuda.synchronize() around the timed region you measure
    kernel LAUNCH time, not execution, and get numbers several times too fast.
  * The first iterations include cuDNN autotuning and lazy allocation, so a warmup is required
    before timing.
  * Batch 1 is the operationally meaningful case: a sensor delivers one frame at a time, and
    the paper's claim is about per-frame latency, not offline batch throughput.

Both arms of the study share one architecture, so latency is identical for all four input
constructions by construction -- the input tensor shape never changes. We nonetheless time a
real trained checkpoint rather than a fresh model, so the reported figure is the deployed one.

    python irstd/bench_latency.py                    # default: temporal_s0 weights, FP32
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=str(ROOT / "irstd/runs/mirsdt_segpre_temporal_s0/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--half", action="store_true", help="FP16; paper reports FP32")
    ap.add_argument("--out", default=str(ROOT / "irstd" / "latency_results.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device -- this measurement is meaningless on CPU")
    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(0)

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    ymodel = YOLO(args.weights)
    net = ymodel.model.to(dev).eval()
    if args.half:
        net = net.half()
    params = get_num_params(net)
    flops = get_flops(net, args.imgsz)

    # fused parameter count is what a deployed model carries
    fused = get_num_params(net.fuse()) if hasattr(net, "fuse") else params

    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=dev,
                    dtype=torch.half if args.half else torch.float)

    with torch.no_grad():
        for _ in range(args.warmup):
            net(x)
        torch.cuda.synchronize()

        # per-iteration timing, so we can report a spread rather than only a mean
        times = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()          # REQUIRED -- see docstring
            times.append((time.perf_counter() - t0) * 1000.0)

    # End-to-end through the ultralytics predict path: letterboxing, NMS, mask assembly.
    # This is the number a system integrator cares about, and it is always slower than the
    # raw forward pass. Uses a real test frame, not the random tensor.
    frame = ROOT / "datasets/MIRSDT_seg/temporal/images/test"
    sample = sorted(frame.glob("*.png"))[0]
    for _ in range(10):
        ymodel.predict(str(sample), imgsz=args.imgsz, device=0, verbose=False)
    torch.cuda.synchronize()
    e2e = []
    for _ in range(100):
        t0 = time.perf_counter()
        ymodel.predict(str(sample), imgsz=args.imgsz, device=0, verbose=False)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1000.0)

    res = {
        "gpu": gpu,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "precision": "FP16" if args.half else "FP32",
        "params": int(params),
        "params_fused": int(fused),
        "gflops": round(float(flops), 1),
        "forward_ms_mean": round(st.mean(times), 3),
        "forward_ms_std": round(st.stdev(times), 3),
        "forward_ms_median": round(st.median(times), 3),
        "forward_fps": round(1000.0 / st.mean(times), 1),
        "end2end_ms_mean": round(st.mean(e2e), 3),
        "end2end_ms_median": round(st.median(e2e), 3),
        "end2end_fps": round(1000.0 / st.mean(e2e), 1),
        "iters": args.iters,
        "note": ("Identical for all four input constructions: the input tensor is 3x640x640 in "
                 "every arm, so the graph and its cost do not change."),
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
