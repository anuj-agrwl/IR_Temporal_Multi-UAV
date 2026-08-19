#!/usr/bin/env python3
"""Build the single-frame vs multi-frame datasets from the OFFICIAL NUDT-MIRSDT release.

Why this supersedes the earlier build: the copy previously on disk was 1280x1024 with 5 px
targets (0.0019 % of frame) and no official split -- it is not this benchmark. The real release is
~209x157 to 318x231 with 5x6 px targets (~0.09 % of frame) and ships `train.txt` / `test.txt`.
Reporting on the official split is what makes results comparable to published work.

The dataset also ships the temporal stack itself: each `Mix/*.mat` holds 5 consecutive frames
(verified: plane 4 is the current frame, planes 2-3 are t-1 and t-2). So the single-vs-multi-frame
comparison is the benchmark's intended usage rather than something imposed on it.

    single/    plane 4 replicated to 3 channels        (one frame)
    temporal/  planes 2,3,4 = t-2, t-1, t              (three frames)

Splits: official train (80 seq) and test (20 seq), untouched. The 20 sequences that appear in
neither official list are used as validation, so the official protocol is preserved exactly.

    python irstd/build_official.py
"""
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "NUDT-MIRSDT"
OUT = ROOT / "datasets" / "MIRSDT_official"


def boxes_from_mask(mask):
    """One connected target per frame in this benchmark; return normalised YOLO boxes."""
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    H, W = mask.shape
    out = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        out.append(((x0 + x1 + 1) / 2 / W, (y0 + y1 + 1) / 2 / H, bw / W, bh / H))
    return out


def main():
    tr = {l.split("/")[0] for l in (SRC / "train.txt").read_text().split()}
    te = {l.split("/")[0] for l in (SRC / "test.txt").read_text().split()}
    allseq = {d.name for d in SRC.iterdir() if d.is_dir() and d.name.startswith("Sequence")}
    va = allseq - tr - te
    split_of = {}
    for s in tr: split_of[s] = "train"
    for s in va: split_of[s] = "val"
    for s in te: split_of[s] = "test"
    print(f"  official train {len(tr)} seq | val (unused seqs) {len(va)} | official test {len(te)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    counts = defaultdict(lambda: [0, 0])
    sizes = []
    tgt = []

    for si, seq in enumerate(sorted(split_of, key=lambda s: int(s.replace("Sequence", ""))), 1):
        sp = split_of[seq]
        for mf in sorted((SRC / seq / "Mix").glob("*.mat")):
            stem = mf.stem
            mp = SRC / seq / "masks" / f"{stem}.png"
            if not mp.exists():
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            bxs = boxes_from_mask(mask)
            if not bxs:
                continue
            mix = scipy.io.loadmat(str(mf))["Mix"]      # (5, H, W)
            if mix.ndim != 3 or mix.shape[0] < 5:
                continue
            cur, p1, p2 = mix[4], mix[3], mix[2]        # t, t-1, t-2
            sizes.append(mask.shape)
            tgt.extend((b[2] * mask.shape[1], b[3] * mask.shape[0]) for b in bxs)

            label = "".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in bxs)
            name = f"{seq}_{stem}"
            for variant, img in (("single", cv2.merge([cur, cur, cur])),
                                 ("temporal", cv2.merge([p2, p1, cur]))):
                di = OUT / variant / "images" / sp
                dl = OUT / variant / "labels" / sp
                di.mkdir(parents=True, exist_ok=True)
                dl.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(di / f"{name}.png"), img)
                (dl / f"{name}.txt").write_text(label)
            counts[sp][0] += 1
            counts[sp][1] += len(bxs)
        if si % 20 == 0:
            print(f"  [{si}/{len(split_of)}] {seq}", flush=True)

    for variant in ("single", "temporal"):
        (OUT / variant / "data.yaml").write_text(
            f"# NUDT-MIRSDT official split. {variant} input "
            f"({'frames t-2,t-1,t' if variant == 'temporal' else 'frame t replicated'}).\n"
            f"path: {OUT/variant}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            "nc: 1\nnames:\n  0: target\n")

    print("\n=== built ===")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} {counts[sp][0]:6d} frames  {counts[sp][1]:5d} targets")
    hs = [s[0] for s in sizes]; ws = [s[1] for s in sizes]
    tw = [t[0] for t in tgt]; th = [t[1] for t in tgt]
    print(f"\n  image size : {min(ws)}x{min(hs)} to {max(ws)}x{max(hs)}")
    print(f"  target size: median {np.median(tw):.1f}x{np.median(th):.1f} px "
          f"({np.median(tw)*np.median(th)/np.median(ws)/np.median(hs)*100:.3f}% of frame)")
    print(f"\nWrote {OUT}/single and {OUT}/temporal")


if __name__ == "__main__":
    main()
