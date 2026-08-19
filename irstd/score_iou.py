#!/usr/bin/env python3
"""IRSTD field-standard scoring for the MIRSDT segmentation study (Paper 5).

WHY THIS EXISTS. `build_seg.py` was written so the study could be placed next to published
IRSTD work, whose docstring says it plainly: "Published IRSTD work reports IoU / nIoU over
masks -- that is the field-standard formulation for this benchmark, so a box mAP number cannot
be placed next to the literature no matter how good it is." But `train_seg.py` reports mask
mAP50 / mAP50-95 / P / R, which are COCO-style metrics. Nothing in the repo computed IoU or
nIoU, so the comparison the segmentation arm exists to enable was still impossible. This fills
that gap. It is training-free: it loads each run's best.pt and scores the held-out test split.

METRIC DEFINITIONS -- these are the ones the IRSTD literature uses, and they are NOT the same
as the mask mAP numbers already reported. Both are computed at a fixed binarisation.

  IoU  (aggregate / dataset-level, as in ACM, Dai et al. WACV 2021 and DNANet, Li et al. 2022)

           IoU = sum_i TP_i / sum_i (T_i + P_i - TP_i)

       One ratio over the pooled pixel counts of the whole split. Large targets dominate it.

  nIoU (normalised IoU, introduced in ACM to stop large targets dominating)

           nIoU = (1/N) * sum_i [ TP_i / (T_i + P_i - TP_i) ]

       The mean of PER-IMAGE IoU. For a small-target benchmark this is the more informative of
       the two, and it is the number most IRSTD papers lead with.

  Pd   probability of detection: a ground-truth target counts as detected when some predicted
       connected component has its centroid within `--pd-dist` pixels of the target's centroid
       (3 px is the common convention). Reported as detected targets / total targets.

  Fa   false-alarm rate: false-positive pixels / total pixels over the split.

  where T_i = ground-truth positive pixels, P_i = predicted positive pixels, TP_i = intersection.

HONEST CAVEAT -- READ BEFORE PUTTING THESE IN A TABLE. Published IRSTD methods are segmentation
networks that emit a per-pixel probability map, and their IoU is computed by thresholding that
map (conventionally at 0.5). This study uses an instance-segmentation model, so the mask is the
union of the instances surviving a CONFIDENCE threshold. That is a genuine methodological
difference in how the binary mask is formed, and the resulting IoU depends on `--conf` in a way
a probability-map IoU does not. Use `--sweep` to see the sensitivity, report which threshold was
used, and state this difference in the paper. Do not present these numbers as though they were
produced by the identical protocol.

Runs on CPU by default so it cannot disturb a training job holding the GPU.

    python irstd/score_iou.py                      # all 6 runs, test split, conf 0.25
    python irstd/score_iou.py --sweep              # threshold sensitivity
    python irstd/score_iou.py --limit 50           # quick smoke test
"""
import argparse
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "MIRSDT_seg"
RUNS = ROOT / "irstd" / "runs"


def official_test_items(variant, cache_root):
    """The OFFICIAL NUDT-MIRSDT test protocol: all 2000 frames of test.txt.

    WHY THIS MODE EXISTS. `build_seg.py` writes a frame only if its mask yields a contour with
    >= 3 vertices, because a 2-point "polygon" would be a fabricated annotation. That is the
    right call for TRAINING labels, but it silently changes the EVALUATION set, and not evenly:

        train   206/8000 dropped (2.57%)
        val      77/2000 dropped (3.85%)
        test    309/2000 dropped (15.45%)   <-- one frame in six

    The dropped frames are the ones whose target is 1-2 px, i.e. the hardest cases in the
    benchmark. Scoring on the 1691 survivors measures a materially easier problem than the one
    published NUDT-MIRSDT numbers are measured on, so any SOTA table built from it would be
    comparing different evaluation sets.

    This mode fixes that. Ground truth is taken from the ORIGINAL mask PNG (always present and
    exact, whatever its contour looks like), and the input frame is rebuilt from the .mat with
    the identical channel construction build_seg.py uses -- verified byte-identical against the
    stored images. Predictions are unchanged; only the evaluation set is corrected.
    """
    src = ROOT / "datasets" / "NUDT-MIRSDT"
    lines = (src / "test.txt").read_text().split()
    cache = cache_root / variant / "official_test_images"
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    for ln in lines:
        seq, _, fn = ln.split("/")
        stem = Path(fn).stem
        mp = src / seq / "masks" / f"{stem}.png"
        if not mp.exists():
            continue
        g = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        out = cache / f"{seq}_{stem}.png"
        if not out.exists():
            import scipy.io
            mix = scipy.io.loadmat(str(src / seq / "Mix" / f"{stem}.mat"))["Mix"]
            if mix.ndim != 3 or mix.shape[0] < 5:
                continue
            cur, p1, p2 = mix[4], mix[3], mix[2]
            img = cv2.merge([cur, cur, cur]) if variant == "single" else cv2.merge([p2, p1, cur])
            cv2.imwrite(str(out), img)
        items.append((out, (g > 0).astype(np.uint8)))
    return items


def polygon_test_items(variant):
    """The reconstructed-polygon subset build_seg.py produced (1691 of 2000 frames)."""
    img_dir = DS / variant / "images" / "test"
    lbl_dir = DS / variant / "labels" / "test"
    items = []
    for p in sorted(img_dir.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        items.append((p, gt_mask(lbl_dir / (p.stem + ".txt"), *img.shape[:2])))
    return items


def gt_mask(label_path, h, w):
    """Rasterise the YOLO polygon label back to a binary mask at image resolution."""
    m = np.zeros((h, w), np.uint8)
    txt = label_path.read_text().strip()
    if not txt:
        return m
    for line in txt.split("\n"):
        v = line.split()
        if len(v) < 7:                      # class + at least 3 (x,y) pairs
            continue
        c = np.array([float(x) for x in v[1:]], np.float32).reshape(-1, 2)
        c[:, 0] *= w
        c[:, 1] *= h
        cv2.fillPoly(m, [c.astype(np.int32)], 1)
    return m


def centroids(mask):
    """Centroids of connected components in a binary mask."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs):
            out.append((xs.mean(), ys.mean()))
    return out


def score_run(run_dir, variant, conf, imgsz, device, limit, pd_dist, items):
    """Score one run against a prepared list of (image_path, gt_mask) pairs."""
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit(f"no test items for variant {variant}")
    imgs = [p for p, _ in items]
    gts = {p: g for p, g in items}

    model = YOLO(str(weights))

    sum_tp = sum_t = sum_p = 0            # pooled pixel counts -> aggregate IoU
    per_img_iou = []                      # -> nIoU
    n_targets = n_detected = 0            # -> Pd
    fa_pixels = total_pixels = 0          # -> Fa

    # Per-sequence accumulators, so results can be split by the DTUM authors' SNR subsets.
    # The DTUM README defines the low-SNR test subset as Sequence[47,56,59,76,92,101,105,119]
    # and reports Pd/Fa/AUC separately for SNR<=3 and SNR>3. Reporting the same breakdown is
    # what makes our numbers placeable next to theirs.
    from collections import defaultdict
    per_seq = defaultdict(lambda: {"tp": 0, "t": 0, "p": 0, "tgt": 0, "det": 0,
                                   "fa": 0, "px": 0, "iou": []})

    for chunk_start in range(0, len(imgs), 64):
        chunk = imgs[chunk_start:chunk_start + 64]
        results = model.predict(
            [str(p) for p in chunk], imgsz=imgsz, conf=conf, device=device,
            retina_masks=True, verbose=False, stream=False,
        )
        for path, r in zip(chunk, results):
            h, w = r.orig_shape
            g = gts[path]

            # Union of the surviving instance masks -> one binary map, comparable to the
            # thresholded probability map a segmentation network would produce.
            if r.masks is not None and len(r.masks.data):
                p = (r.masks.data.cpu().numpy().sum(0) > 0).astype(np.uint8)
                if p.shape != (h, w):                       # safety net if retina_masks is off
                    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                p = np.zeros((h, w), np.uint8)

            tp = int(np.count_nonzero(g & p))
            t = int(np.count_nonzero(g))
            pp = int(np.count_nonzero(p))
            sum_tp += tp
            sum_t += t
            sum_p += pp
            union = t + pp - tp
            per_img_iou.append(tp / union if union else 0.0)

            fa_pixels += pp - tp
            total_pixels += h * w

            gc = centroids(g)
            pc = centroids(p)
            n_targets += len(gc)
            ndet = 0
            for cx, cy in gc:
                if any((cx - px) ** 2 + (cy - py) ** 2 <= pd_dist ** 2 for px, py in pc):
                    ndet += 1
            n_detected += ndet

            s = per_seq[path.name.split("_")[0]]
            s["tp"] += tp; s["t"] += t; s["p"] += pp
            s["tgt"] += len(gc); s["det"] += ndet
            s["fa"] += pp - tp; s["px"] += h * w
            s["iou"].append(per_img_iou[-1])

    denom = sum_t + sum_p - sum_tp
    # DTUM's low-SNR test subset, verbatim from its README.
    LOW_SNR = {f"Sequence{n}" for n in (47, 56, 59, 76, 92, 101, 105, 119)}

    def subset(keys):
        tp = sum(per_seq[k]["tp"] for k in keys)
        t = sum(per_seq[k]["t"] for k in keys)
        p = sum(per_seq[k]["p"] for k in keys)
        tgt = sum(per_seq[k]["tgt"] for k in keys)
        det = sum(per_seq[k]["det"] for k in keys)
        fa = sum(per_seq[k]["fa"] for k in keys)
        px = sum(per_seq[k]["px"] for k in keys)
        ious = [i for k in keys for i in per_seq[k]["iou"]]
        d = t + p - tp
        return {"sequences": len(keys), "IoU": tp / d if d else 0.0,
                "nIoU": float(np.mean(ious)) if ious else 0.0,
                "Pd": det / tgt if tgt else 0.0, "Fa": fa / px if px else 0.0}

    seen = set(per_seq)
    return {
        "run": run_dir.name,
        "variant": variant,
        "images": len(imgs),
        "conf": conf,
        "IoU": sum_tp / denom if denom else 0.0,
        "nIoU": float(np.mean(per_img_iou)) if per_img_iou else 0.0,
        "Pd": n_detected / n_targets if n_targets else 0.0,
        "Fa": fa_pixels / total_pixels if total_pixels else 0.0,
        "targets": n_targets,
        "snr_low": subset(seen & LOW_SNR),
        "snr_high": subset(seen - LOW_SNR),
    }


def summarise(rows, label=""):
    """Print per-run numbers and the mean +/- std across seeds for each arm."""
    print(f"\n=== IRSTD metrics, held-out TEST split {label} ===")
    print(f"{'run':<32}{'IoU':>9}{'nIoU':>9}{'Pd':>9}{'Fa':>12}")
    for r in rows:
        print(f"{r['run']:<32}{r['IoU']:>9.4f}{r['nIoU']:>9.4f}{r['Pd']:>9.4f}{r['Fa']:>12.2e}")

    print(f"\n{'arm':<12}{'IoU (n=3)':>22}{'nIoU (n=3)':>22}{'Pd (n=3)':>22}")
    agg = {}
    # Iterate the arms actually present in `rows`, in first-appearance order, rather than a
    # hardcoded pair -- the study grew to four arms (single/temporal/shuffled/wide) on
    # 13 Aug 2026 and the ablation arms were being silently dropped from this summary.
    seen = []
    for r in rows:
        if r["variant"] not in seen:
            seen.append(r["variant"])
    for v in seen:
        sel = [r for r in rows if r["variant"] == v]
        if len(sel) < 2:
            continue
        cell = []
        for k in ("IoU", "nIoU", "Pd"):
            vals = [r[k] for r in sel]
            m, s = st.mean(vals), st.stdev(vals)
            agg[f"{v}_{k}"] = (m, s)
            cell.append(f"{m:.4f} ±{s:.4f}")
        print(f"{v:<12}" + "".join(f"{c:>22}" for c in cell))
    if "single_IoU" in agg and "temporal_IoU" in agg:
        print()
        for k in ("IoU", "nIoU", "Pd"):
            d = (agg[f"temporal_{k}"][0] - agg[f"single_{k}"][0]) * 100
            print(f"  delta {k:<5} temporal - single = {d:+.1f} points")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so a training job on the GPU is never disturbed")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N images. SMOKE TEST ONLY -- this is the "
                         "alphabetical head, not a sample, so it lands inside a single sequence "
                         "(the first 20 test images are all Sequence101) and is not "
                         "representative. The single arm legitimately scores 0 there.")
    ap.add_argument("--pd-dist", type=float, default=3.0,
                    help="centroid distance in px for a target to count as detected")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the confidence threshold to show IoU sensitivity")
    ap.add_argument("--protocol", choices=("official", "polygon"), default="official",
                    help="'official' = all 2000 frames of NUDT-MIRSDT test.txt, GT from the "
                         "original mask PNGs -- the only mode comparable to published numbers. "
                         "'polygon' = the 1691-frame subset build_seg.py produced, which drops "
                         "15.45%% of test frames (the 1-2 px targets) and is therefore EASIER. "
                         "Report official; use polygon only to quantify the difference.")
    ap.add_argument("--variants", default="single,temporal,shuffled,wide",
                    help="comma-separated arms to score; each needs its own "
                         "datasets/MIRSDT_seg/<variant>/ image set")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = str(ROOT / "irstd" / f"iou_results_{args.protocol}.json")

    import torch
    torch.set_num_threads(8)        # leave cores for the training dataloader

    # All four arms of the study. shuffled/wide are the ablation controls added 13 Aug 2026
    # (irstd/build_ablation.py); each has its own image set under datasets/MIRSDT_seg/<variant>,
    # so items_for must be built per variant -- the frames differ even though the labels do not.
    variants = tuple(args.variants.split(","))
    runs = [(RUNS / f"mirsdt_segpre_{v}_s{s}", v) for v in variants for s in (0, 1, 2)]
    runs = [(d, v) for d, v in runs if d.exists()]
    if not runs:
        raise SystemExit("no mirsdt_segpre_* run directories found")

    items_for = {}
    for v in variants:
        items_for[v] = (official_test_items(v, DS) if args.protocol == "official"
                        else polygon_test_items(v))
        print(f"  {args.protocol} protocol: {v:9s} {len(items_for[v])} test frames", flush=True)

    confs = [0.05, 0.10, 0.25, 0.50] if args.sweep else [args.conf]
    payload = {"protocol": args.protocol}
    for c in confs:
        rows = []
        for d, v in runs:
            r = score_run(d, v, c, args.imgsz, args.device, args.limit, args.pd_dist,
                          items_for[v])
            if r:
                rows.append(r)
                print(f"  scored {r['run']:<32} conf={c:<5} IoU={r['IoU']:.4f} nIoU={r['nIoU']:.4f}",
                      flush=True)
        agg = summarise(rows, f"(conf={c})")
        payload[f"conf_{c}"] = {"runs": rows,
                                "summary": {k: {"mean": m, "std": s} for k, (m, s) in agg.items()}}

    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    if args.limit:
        print("NOTE: --limit was set, these are NOT full-split numbers.")


if __name__ == "__main__":
    main()
