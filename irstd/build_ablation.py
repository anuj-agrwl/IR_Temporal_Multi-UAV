#!/usr/bin/env python3
"""Build the two ABLATION arms for the paper.

The main study establishes that [I(t-2), I(t-1), I(t)] beats [I(t), I(t), I(t)]. It does NOT
establish *why*. Two alternative explanations survive that experiment, and each gets a control
here. Both controls keep the input at exactly three channels, so the zero-cost property --
the whole point of the paper -- is preserved and the comparison stays like-for-like.

  shuffled : a deterministic per-frame PERMUTATION of the same three consecutive frames.
             Identical pixel content, identical statistics, only the channel ORDER changes.
             -> If the gain survives, the network is using inter-frame *difference*, not
                *directed* motion, and the paper must not claim direction is being exploited.
             -> If the gain collapses, temporal ORDER carries the signal.
             The permutation is seeded per frame name, so it is fixed and reproducible, and it
             is never the identity (that would silently duplicate the temporal arm).

  wide     : [I(t-4), I(t-2), I(t)] -- three frames spanning FIVE, sampled at stride 2.
             Same three channels, same cost, roughly double the temporal baseline.
             -> Tests whether a longer baseline helps, which matters because the median
                two-frame target displacement is only 1.90 px. This is the honest way to ask
                the "why not more frames?" question without adding a fourth channel, which
                would change the first-layer parameter count and end the zero-cost claim.

The source .mat holds five consecutive frames as Mix[0..4] = I(t-4) .. I(t), which is what
makes the `wide` arm possible at all with no extra data.

Labels, split protocol and every other detail are copied unchanged from build_seg.py, so these
arms are directly comparable with the single/temporal arms already trained.

    python irstd/build_ablation.py
"""
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "NUDT-MIRSDT"
OUT = ROOT / "datasets" / "MIRSDT_seg"          # new variants land beside single/ and temporal/
VARIANTS = ("shuffled", "wide")

# The five non-identity permutations of three channels. Chosen per frame by a hash of its
# name, so the assignment is deterministic across rebuilds and machines but carries no
# systematic relationship to sequence, split or time.
PERMS = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def polys_from_mask(mask):
    """Outer contours -> normalised YOLO polygons. Identical to build_seg.py."""
    H, W = mask.shape
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        c = c.reshape(-1, 2)
        if len(c) < 3:
            continue
        out.append(" ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in c))
    return out


def perm_for(name):
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16)
    return PERMS[h % len(PERMS)]


def main():
    tr = {l.split("/")[0] for l in (SRC / "train.txt").read_text().split()}
    te = {l.split("/")[0] for l in (SRC / "test.txt").read_text().split()}
    allseq = {d.name for d in SRC.iterdir() if d.is_dir() and d.name.startswith("Sequence")}
    split_of = {}
    for s in tr:
        split_of[s] = "train"
    for s in (allseq - tr - te):
        split_of[s] = "val"
    for s in te:
        split_of[s] = "test"
    print(f"  official train {len(tr)} | val {len(allseq-tr-te)} | official test {len(te)}")

    for v in VARIANTS:
        if (OUT / v).exists():
            shutil.rmtree(OUT / v)

    counts = defaultdict(lambda: [0, 0])
    dropped = 0
    perm_hist = defaultdict(int)

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

            f0, f1, f2, f3, f4 = mix[0], mix[1], mix[2], mix[3], mix[4]
            name = f"{seq}_{mf.stem}"
            label = "".join(f"0 {p}\n" for p in polys)

            # shuffled: same three frames as the temporal arm, order permuted
            base = [f2, f3, f4]                       # I(t-2), I(t-1), I(t)
            p = perm_for(name)
            perm_hist[p] += 1
            imgs = {
                "shuffled": cv2.merge([base[p[0]], base[p[1]], base[p[2]]]),
                # wide: I(t-4), I(t-2), I(t) -- stride 2, spanning five frames
                "wide": cv2.merge([f0, f2, f4]),
            }

            for v in VARIANTS:
                di, dl = OUT / v / "images" / sp, OUT / v / "labels" / sp
                di.mkdir(parents=True, exist_ok=True)
                dl.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(di / f"{name}.png"), imgs[v])
                (dl / f"{name}.txt").write_text(label)
            counts[sp][0] += 1
            counts[sp][1] += len(polys)
        if si % 20 == 0:
            print(f"  [{si}/{len(split_of)}] {seq}", flush=True)

    for v in VARIANTS:
        desc = ("channel-order control: permuted [I(t-2),I(t-1),I(t)]" if v == "shuffled"
                else "wide-baseline control: [I(t-4),I(t-2),I(t)], stride 2")
        (OUT / v / "data.yaml").write_text(
            f"# NUDT-MIRSDT official split, SEGMENTATION, {v} -- {desc}\n"
            f"path: {OUT/v}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            "nc: 1\nnames:\n  0: target\n")

    print("\n=== built (ablation arms) ===")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} {counts[sp][0]:6d} frames  {counts[sp][1]:5d} polygons")
    print(f"  dropped (contour < 3 vertices): {dropped}")
    print("  permutation histogram (should be roughly uniform over the 5 non-identity perms):")
    for k, n in sorted(perm_hist.items()):
        print(f"    {k}: {n}")
    print(f"\nWrote {OUT}/shuffled and {OUT}/wide")


if __name__ == "__main__":
    main()
