#!/usr/bin/env python3
"""Measure two things the paper currently asserts without evidence.

1. **Is the sensor actually stationary?** The paper claims frame registration is unnecessary here.
   That was an assumption, not a measurement. Phase correlation between consecutive frames gives
   the global background shift directly: near zero means a fixed sensor and registration would be
   a no-op; a consistent non-zero shift would mean the claim is wrong.

2. **How far does a target move between frames?** This bounds the method's weak case. Temporal
   stacking works because the target moves against a static background; a target that does not
   move in the image plane -- hovering, or flying head-on -- gives the temporal arm no extra
   signal. Per-frame mask centroids give the displacement distribution, and the fraction of frames
   with sub-pixel motion quantifies how often that weak case actually occurs.

CPU-only and nice'd; a training job owns the GPU.

    python irstd/analyse_motion.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "NUDT-MIRSDT"
OUT = ROOT / "irstd" / "motion_analysis.json"
MAX_SEQ = 40          # enough for a stable distribution without reading all 120


def centroid(mask):
    ys, xs = np.where(mask > 0)
    return (xs.mean(), ys.mean()) if len(xs) else None


def main():
    bg_shifts, tgt_steps, per_seq = [], [], {}

    seqs = sorted((d for d in SRC.iterdir() if d.is_dir() and d.name.startswith("Sequence")),
                  key=lambda p: int(p.name.replace("Sequence", "")))[:MAX_SEQ]

    for seq in seqs:
        imgs = sorted((seq / "images").glob("*.png"))
        if len(imgs) < 3:
            continue
        prev_img = prev_c = None
        sb, st = [], []
        for ip in imgs:
            g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            mp_ = seq / "masks" / f"{ip.stem}.png"
            m = cv2.imread(str(mp_), cv2.IMREAD_GRAYSCALE) if mp_.exists() else None
            c = centroid(m) if m is not None else None

            if prev_img is not None and g is not None and g.shape == prev_img.shape:
                # global background shift; the target is a handful of pixels so it cannot
                # dominate the correlation peak
                (dx, dy), _ = cv2.phaseCorrelate(prev_img.astype(np.float64),
                                                 g.astype(np.float64))
                sb.append(float(np.hypot(dx, dy)))
            if prev_c is not None and c is not None:
                st.append(float(np.hypot(c[0] - prev_c[0], c[1] - prev_c[1])))
            prev_img, prev_c = g, c

        if sb:
            bg_shifts += sb
        if st:
            tgt_steps += st
            per_seq[seq.name] = round(float(np.median(st)), 3)

    bg = np.array(bg_shifts)
    tg = np.array(tgt_steps)

    res = {
        "n_sequences": len(per_seq),
        "background_shift_px_per_frame": {
            "median": round(float(np.median(bg)), 4),
            "p95": round(float(np.percentile(bg, 95)), 4),
            "max": round(float(bg.max()), 4),
            "frac_under_1px": round(float((bg < 1.0).mean()), 4),
        },
        "target_step_px_per_frame": {
            "median": round(float(np.median(tg)), 3),
            "p05": round(float(np.percentile(tg, 5)), 3),
            "p95": round(float(np.percentile(tg, 95)), 3),
            "frac_under_0p5px": round(float((tg < 0.5).mean()), 4),
            "frac_under_1px": round(float((tg < 1.0).mean()), 4),
            "frac_under_2px": round(float((tg < 2.0).mean()), 4),
        },
        "median_target_step_by_sequence": per_seq,
    }
    OUT.write_text(json.dumps(res, indent=2))

    b, t = res["background_shift_px_per_frame"], res["target_step_px_per_frame"]
    print(f"sequences analysed: {res['n_sequences']}\n")
    print("1. BACKGROUND motion between consecutive frames (is the sensor moving?)")
    print(f"   median {b['median']:.3f} px | p95 {b['p95']:.3f} px | max {b['max']:.3f} px")
    print(f"   {100*b['frac_under_1px']:.1f} % of frame pairs shift < 1 px")
    print("   -> stationary sensor confirmed" if b["median"] < 1.0 else
          "   -> SENSOR MOVES: the paper's registration claim is wrong")
    print("\n2. TARGET motion between consecutive frames (how often is the cue absent?)")
    print(f"   median {t['median']:.2f} px | p05 {t['p05']:.2f} | p95 {t['p95']:.2f}")
    print(f"   {100*t['frac_under_0p5px']:.1f} % of steps < 0.5 px  (effectively static)")
    print(f"   {100*t['frac_under_1px']:.1f} % < 1 px | {100*t['frac_under_2px']:.1f} % < 2 px")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
