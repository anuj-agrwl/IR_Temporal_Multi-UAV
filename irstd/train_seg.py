#!/usr/bin/env python3
"""Train one arm of the MIRSDT segmentation study (journal paper).

NOTE ON WEIGHTS: with `--pretrained` the model starts from `yolo11s-seg.pt` (COCO-pretrained) and
the run is tagged `segpre_*`; without the flag it is built from `yolo11s-seg.yaml` and trained from
a random initialisation, tagged `seg_*`. Both arms are always treated identically, so the
single-vs-temporal comparison is valid either way, but absolute IoU from a random initialisation
sits below published numbers that start from COCO weights. Use `--pretrained` for any comparison
against the literature.

    python irstd/train_seg.py --variant single
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "MIRSDT_seg"


def main():
    ap = argparse.ArgumentParser()
    # single/temporal = the main study. shuffled/wide = the two ablation controls built by
    # irstd/build_ablation.py: shuffled permutes the channel order of the same three frames
    # (does ORDER matter?), wide uses [I(t-4),I(t-2),I(t)] (does a longer baseline help?).
    # All four are three-channel, so the recipe below is identical for every arm.
    ap.add_argument("--variant", choices=["single", "temporal", "shuffled", "wide"],
                    required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--pretrained", action="store_true",
                    help="start from yolo11s-seg.pt instead of a random init")
    args = ap.parse_args()

    data = DS / args.variant / "data.yaml"
    if not data.exists():
        raise SystemExit(f"missing {data} -- run irstd/build_seg.py first")

    from ultralytics import YOLO
    W = ROOT / "anti-uav" / "yolo11s-seg.pt"
    if args.pretrained:
        if not W.exists():
            raise SystemExit(f"missing {W}")
        model = YOLO(str(W))
    else:
        model = YOLO("yolo11s-seg.yaml")      # from config: no download, offline-safe

    tag = "segpre" if args.pretrained else "seg"
    name = (f"mirsdt_{tag}_{args.variant}_s{args.seed}" if args.pretrained
            else f"mirsdt_seg_{args.variant}")
    kw = dict(data=str(data), imgsz=args.imgsz, epochs=(1 if args.sanity else args.epochs),
              batch=args.batch, device="0", seed=args.seed, workers=4,
              project=str(ROOT / "irstd" / "runs"),
              name=("sanity_seg_" + args.variant if args.sanity else name),
              patience=20, deterministic=True, val=True, plots=True, exist_ok=True,
              degrees=0.0, shear=0.0, perspective=0.0, mosaic=0.0, mixup=0.0)
    if args.sanity:
        kw["fraction"] = 0.02
    model.train(**kw)

    if not args.sanity:
        m = model.val(data=str(data), split="test", imgsz=args.imgsz,
                      project=str(ROOT / "irstd" / "runs"), name=name + "_test", exist_ok=True)
        print(f"TESTSEG {args.variant} seed={args.seed}  boxmAP50={m.box.map50:.4f}  maskmAP50={m.seg.map50:.4f} "
              f"maskmAP50-95={m.seg.map:.4f}  maskP={m.seg.mp:.4f} maskR={m.seg.mr:.4f}")


if __name__ == "__main__":
    main()
