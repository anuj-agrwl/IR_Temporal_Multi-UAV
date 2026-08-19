#!/usr/bin/env python3
"""Build NUDT-MIRSDT in DNANet's expected layout, one directory per input construction.

WHY. The YOLO-based study answers "does temporal channel stacking help?" under a controlled
comparison, but a reviewer will reasonably ask whether the effect survives on an architecture
the field actually uses. DNANet is a purpose-built IRSTD segmentation network with public code
and published NUDT-MIRSDT numbers, and -- crucially -- it already takes a THREE-channel input:

    model/parse_args_train.py:  --in_channels default=3
    model/utils.py:             img = Image.open(img_path).convert('RGB')

On a grayscale infrared frame `convert('RGB')` replicates one frame into all three channels,
which is exactly our single-frame baseline. Temporal stacking therefore drops into DNANet with
no architectural change and no added parameters, and the zero-cost claim carries over intact.
That makes DNANet-vs-DNANet+stacking a like-for-like test of OUR modification, rather than a
cross-technique comparison against a different method's absolute numbers.

WHAT THIS WRITES, per variant:

    datasets/MIRSDT_dnanet/<variant>/
        images/<seq>_<stem>.png     3-channel, channels per the variant (see below)
        masks/<seq>_<stem>.png      binary target mask, 0/255, straight from the source
        idx/train.txt               image ids, one per line
        idx/test.txt                official test sequences only

    single    [ I(t),   I(t),   I(t)   ]   == DNANet's own default behaviour
    temporal  [ I(t-2), I(t-1), I(t)   ]
    wide      [ I(t-4), I(t-2), I(t)   ]   best arm in the YOLO study

INTEGRITY NOTES.
  * Masks are the ORIGINAL mask PNGs, not polygons rasterised back. There is no 1691-vs-2000
    frame loss here: every frame with a mask is kept, including the 1-2 px targets that the
    polygon build had to drop. This evaluation set is the full official one.
  * Channel construction is byte-identical to build_seg.py / build_ablation.py, so the arms are
    the same data the YOLO study used.
  * Splitting is by SEQUENCE using the official train.txt / test.txt, never by frame.

    python irstd/build_dnanet.py
"""
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "NUDT-MIRSDT"
OUT = ROOT / "datasets" / "MIRSDT_dnanet"
VARIANTS = ("single", "temporal", "wide")


def channels(mix, variant):
    """mix[0..4] = I(t-4) .. I(t). Same construction as build_seg.py/build_ablation.py."""
    f0, _, f2, f3, f4 = mix[0], mix[1], mix[2], mix[3], mix[4]
    if variant == "single":
        return cv2.merge([f4, f4, f4])
    if variant == "temporal":
        return cv2.merge([f2, f3, f4])
    if variant == "wide":
        return cv2.merge([f0, f2, f4])
    raise ValueError(variant)


def main():
    tr = {l.split("/")[0] for l in (SRC / "train.txt").read_text().split()}
    te = {l.split("/")[0] for l in (SRC / "test.txt").read_text().split()}
    allseq = {d.name for d in SRC.iterdir() if d.is_dir() and d.name.startswith("Sequence")}
    split_of = {s: "train" for s in tr}
    split_of.update({s: "train" for s in (allseq - tr - te)})   # val folded into train:
    split_of.update({s: "test" for s in te})                    # DNANet uses train/test only
    print(f"  sequences: train+val {len(allseq - te)} | test {len(te)}")

    for v in VARIANTS:
        if (OUT / v).exists():
            shutil.rmtree(OUT / v)
        for sub in ("images", "masks", "idx"):
            (OUT / v / sub).mkdir(parents=True, exist_ok=True)

    ids = defaultdict(list)
    kept = skipped = 0

    for si, seq in enumerate(sorted(split_of, key=lambda s: int(s.replace("Sequence", ""))), 1):
        sp = split_of[seq]
        for mf in sorted((SRC / seq / "Mix").glob("*.mat")):
            mp = SRC / seq / "masks" / f"{mf.stem}.png"
            if not mp.exists():
                skipped += 1
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                skipped += 1
                continue
            mix = scipy.io.loadmat(str(mf))["Mix"]
            if mix.ndim != 3 or mix.shape[0] < 5:
                skipped += 1
                continue

            name = f"{seq}_{mf.stem}"
            binm = ((mask > 0).astype(np.uint8)) * 255
            for v in VARIANTS:
                cv2.imwrite(str(OUT / v / "images" / f"{name}.png"), channels(mix, v))
                cv2.imwrite(str(OUT / v / "masks" / f"{name}.png"), binm)
            ids[sp].append(name)
            kept += 1
        if si % 20 == 0:
            print(f"  [{si}/{len(split_of)}] {seq}", flush=True)

    for v in VARIANTS:
        for sp in ("train", "test"):
            (OUT / v / "idx" / f"{sp}.txt").write_text("\n".join(ids[sp]) + "\n")

    print("\n=== built (DNANet layout) ===")
    print(f"  train {len(ids['train'])} frames | test {len(ids['test'])} frames")
    print(f"  kept {kept}, skipped {skipped} (no mask / short stack)")
    print(f"  variants: {', '.join(VARIANTS)}")
    print(f"\nWrote {OUT}")
    print("NOTE: test set here is the FULL official list -- no polygon-degeneracy loss.")


if __name__ == "__main__":
    main()
