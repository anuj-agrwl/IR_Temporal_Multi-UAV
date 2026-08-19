#!/usr/bin/env python3
"""Paired per-image significance test for the temporal-vs-single comparison (Paper 5).

WHY. The seed-level comparison in the paper uses n=3 per arm, at which a rank test bottoms out
at p=0.1 one-sided however large the effect -- not because the effect is weak but because three
ranks cannot produce a smaller statistic. That is stated honestly in the manuscript, but it
leaves the central claim without statistical support.

This tests the same claim on the unit that actually has samples: the 2000 held-out test frames.
For each frame we score both arms with the SAME metric under the SAME threshold, pair them by
frame, and apply a two-sided Wilcoxon signed-rank test over the paired differences.

WHAT IS BEING COMPARED, precisely -- this is a DIFFERENT metric from the mAP figures in the main
tables and the manuscript must say so. Per frame we compute the IoU of the predicted mask
against ground truth at the network's own binarisation (logit > 0, i.e. sigmoid > 0.5), which is
the convention DNANet's own metric code uses. Frames where both arms score zero carry no
information about the difference and are excluded from the ranking, which Wilcoxon does anyway;
the count of such ties is reported so the effective n is visible.

Seeds are paired within-seed (single_sN vs temporal_sN) and then pooled, so no cross-seed
comparison is made.

    python irstd/paired_test.py --model mshnet --arms single8f temporal8f
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "irstd"))


def per_image_iou(net, loader, dev="cuda"):
    """IoU per frame at the network's own threshold (logit > 0)."""
    out = []
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            r = net(x, False)
            pred = r[1] if isinstance(r, (list, tuple)) else r
            p = (pred > 0).float()
            t = (y > 0.5).float()
            inter = (p * t).sum(dim=(1, 2, 3))
            union = ((p + t) > 0).float().sum(dim=(1, 2, 3))
            iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
            out.extend(iou.cpu().numpy().tolist())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs=2, default=["single8f", "temporal8f"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda", help="cpu keeps this off a busy GPU")
    ap.add_argument("--out", default=str(ROOT / "irstd" / "paired_test_results.json"))
    args = ap.parse_args()

    import torch.utils.data as D
    sys.path.insert(0, str(ROOT / "irstd" / "mshnet"))
    from model.MSHNet import MSHNet
    from mirsdt_multiframe import MultiFrameSet

    idx = ROOT / "datasets" / "MIRSDT_dnanet" / "single" / "idx" / "test.txt"
    ids = [l.strip() for l in idx.read_text().split("\n") if l.strip()]
    print(f"test frames: {len(ids)}")

    per_arm = {}
    for arm in args.arms:
        rep, dif = arm.startswith("single"), arm.startswith("difference")
        ds = MultiFrameSet(ids, frames=args.frames, base_size=args.size, crop_size=args.size,
                           train=False, replicate=rep, difference=dif)
        dl = D.DataLoader(ds, batch_size=args.batch, num_workers=4, drop_last=False)
        runs = []
        for sd in args.seeds:
            ck = ROOT / "irstd" / "mshnet_runs" / f"{arm}_s{sd}" / "best.pth"
            if not ck.exists():
                print(f"  missing {ck}"); continue
            net = MSHNet(input_channels=args.frames).to(args.device)
            net.load_state_dict(torch.load(ck, map_location=args.device)["state_dict"])
            v = per_image_iou(net, dl, args.device)
            runs.append(v)
            print(f"  {arm}_s{sd}: mean per-image IoU {v.mean():.4f}")
            del net
            if args.device == "cuda": torch.cuda.empty_cache()
        per_arm[arm] = runs

    from scipy.stats import wilcoxon
    A, B = args.arms
    res = {"arms": args.arms, "n_frames": len(ids), "per_seed": [], "metric":
           "per-frame mask IoU at logit>0 (sigmoid>0.5); NOT the mAP of the main tables"}
    for k, sd in enumerate(args.seeds):
        if k >= len(per_arm[A]) or k >= len(per_arm[B]): continue
        a, b = per_arm[A][k], per_arm[B][k]
        d = b - a
        nz = int((d != 0).sum())
        st, pv = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        res["per_seed"].append({"seed": sd, "mean_A": float(a.mean()), "mean_B": float(b.mean()),
                                "mean_diff": float(d.mean()), "n_nonzero_diff": nz,
                                "W": float(st), "p": float(pv),
                                "frames_B_better": int((d > 0).sum()),
                                "frames_A_better": int((d < 0).sum())})
        print(f"  seed {sd}: mean diff {d.mean():+.4f}  W={st:.0f}  p={pv:.3e}  "
              f"(non-tied pairs {nz}; B better on {int((d>0).sum())}, A better on {int((d<0).sum())})")
    # pooled across seeds
    a = np.concatenate(per_arm[A]); b = np.concatenate(per_arm[B])
    st, pv = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    res["pooled"] = {"n_pairs": int(len(a)), "mean_A": float(a.mean()), "mean_B": float(b.mean()),
                     "mean_diff": float((b - a).mean()), "W": float(st), "p": float(pv)}
    print(f"\npooled ({len(a)} pairs): {A} {a.mean():.4f} vs {B} {b.mean():.4f}  "
          f"W={st:.0f}  p={pv:.3e}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
