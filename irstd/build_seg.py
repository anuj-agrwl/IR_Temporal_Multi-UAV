#!/usr/bin/env python3
"""Build the SEGMENTATION arm of the MIRSDT study, for the journal paper.

Why segmentation and not boxes. Published IRSTD work reports IoU / nIoU over masks -- that is the
field-standard formulation for this benchmark, so a box mAP number cannot be placed next to the
literature no matter how good it is.

Same design as the detection build: identical architecture, recipe, seed, labels and official
split, with only the input channels differing --

    single/    plane 4 replicated        (frame t)
    temporal/  planes 2,3,4              (t-2, t-1, t)

so the single-vs-temporal effect is isolated exactly as before, now on the metric the field uses.

Labels are YOLO polygon format, taken from each mask's outer contour. Targets here are ~7x7 px, so
a contour is only a handful of points; degenerate ones (fewer than 3 vertices after simplification)
are dropped rather than padded, since a padded polygon would be a fabricated annotation.

    python irstd/build_seg.py
"""
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "NUDT-MIRSDT"
OUT = ROOT / "datasets" / "MIRSDT_seg"


def polys_from_mask(mask):
    """Outer contours -> normalised YOLO polygons. Returns [] if nothing usable."""
    H, W = mask.shape
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        c = c.reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in c)
        out.append(pts)
    return out


def main():
    tr = {l.split("/")[0] for l in (SRC / "train.txt").read_text().split()}
    te = {l.split("/")[0] for l in (SRC / "test.txt").read_text().split()}
    allseq = {d.name for d in SRC.iterdir() if d.is_dir() and d.name.startswith("Sequence")}
    split_of = {}
    for s in tr: split_of[s] = "train"
    for s in (allseq - tr - te): split_of[s] = "val"
    for s in te: split_of[s] = "test"
    print(f"  official train {len(tr)} | val {len(allseq-tr-te)} | official test {len(te)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    counts = defaultdict(lambda: [0, 0])
    dropped = 0

    for si, seq in enumerate(sorted(split_of, key=lambda s: int(s.replace("Sequence", ""))), 1):
        sp = split_of[seq]
        for mf in sorted((SRC / seq / "Mix").glob("*.mat")):
            mp = SRC / seq / "masks" / f"{mf.stem}.png"
            if not mp.exists():
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            polys = polys_from_mask(mask)
            if not polys:
                dropped += 1
                continue
            mix = scipy.io.loadmat(str(mf))["Mix"]
            if mix.ndim != 3 or mix.shape[0] < 5:
                continue
            cur, p1, p2 = mix[4], mix[3], mix[2]

            label = "".join(f"0 {p}\n" for p in polys)
            name = f"{seq}_{mf.stem}"
            for variant, img in (("single", cv2.merge([cur, cur, cur])),
                                 ("temporal", cv2.merge([p2, p1, cur]))):
                di = OUT / variant / "images" / sp
                dl = OUT / variant / "labels" / sp
                di.mkdir(parents=True, exist_ok=True)
                dl.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(di / f"{name}.png"), img)
                (dl / f"{name}.txt").write_text(label)
            counts[sp][0] += 1
            counts[sp][1] += len(polys)
        if si % 20 == 0:
            print(f"  [{si}/{len(split_of)}] {seq}", flush=True)

    for variant in ("single", "temporal"):
        (OUT / variant / "data.yaml").write_text(
            f"# NUDT-MIRSDT official split, SEGMENTATION, {variant} input\n"
            f"path: {OUT/variant}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            "nc: 1\nnames:\n  0: target\n")

    print("\n=== built (segmentation) ===")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} {counts[sp][0]:6d} frames  {counts[sp][1]:5d} polygons")
    print(f"  dropped (contour < 3 vertices): {dropped}")
    print(f"\nWrote {OUT}/single and {OUT}/temporal")


if __name__ == "__main__":
    main()
