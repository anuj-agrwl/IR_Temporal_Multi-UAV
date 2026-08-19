#!/usr/bin/env python3
"""Train MSHNet on NUDT-MIRSDT with a given input-channel construction (Paper 5).

WHY MSHNet AS A SECOND HOST. DNANet (2021) is the only backbone with a published
"+temporal module" number on this benchmark, which makes DNANet+DTUM the one genuinely
like-for-like comparison available -- but it is an old network and it caps the absolute
figure. MSHNet (CVPR 2024) is a modern IRSTD architecture with public code, so it raises the
ceiling while the DNANet result keeps the controlled anchor.

WHY IT IS A CLEAN HOST FOR THIS TEST. Exactly as with DNANet, MSHNet's own default input is
already our control condition:

    model/MSHNet.py : MSHNet(input_channels, ...) ; conv_init = Conv2d(input_channels, 16, 1, 1)
    utils/data.py   : img = Image.open(img_path).convert('RGB')      <- one frame, replicated

so substituting consecutive frames changes only the tensor contents. The init layer is 1x1, so
widening the input from 3 to 8 channels costs 8*16 - 3*16 = 80 weights.

TWO DEFECTS IN THE RELEASED CODE THAT THIS WRAPPER DOES NOT REPRODUCE -- both identical in kind
to DNANet's, which is worth noting in the paper as a pattern in this literature:

  1. MODEL SELECTION ON THE TEST SET. utils/data.py maps mode='val' to 'test.txt', and main.py
     keeps the best epoch by IoU on that loader. Every reported test number would then be
     optimistically biased. Here: train on train.txt, select on a held-out val.txt (the 20
     sequences in neither official list), evaluate test exactly once at the end.
  2. NO IDEMPOTENT COMPLETION MARKER. This writes a fixed run directory plus a done/ directory
     created only after the test evaluation succeeds, so the queue runner can test completion
     without the best.pt trap.

DELIBERATELY UNCHANGED from the authors' recipe: architecture, SLSIoULoss with deep supervision
on the four auxiliary masks, Adagrad, warm-up epochs, their own mIoU / PD_FA metric code, and
256x256 evaluation. Only the epoch budget differs (their default is 400).

    python irstd/train_mshnet.py --variant temporal --frames 8 --seed 0 --epochs 50
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as D

ROOT = Path(__file__).resolve().parent.parent
MSH = ROOT / "irstd" / "mshnet"
sys.path.insert(0, str(MSH))
sys.path.insert(0, str(ROOT / "irstd"))

from model.MSHNet import MSHNet                                   # noqa: E402
from model.loss import SLSIoULoss                                 # noqa: E402
from utils.metric import mIoU, PD_FA, ROCMetric                   # noqa: E402
from mirsdt_multiframe import MultiFrameSet                       # noqa: E402


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def ids(p):
    return [l.strip() for l in Path(p).read_text().split("\n") if l.strip()]


@torch.no_grad()
def evaluate(net, loader, n_imgs, size, bins=10):
    """IoU / Pd / Fa using the AUTHORS' metric code so the numbers stay comparable.

    NOTE ON THRESHOLDS: PD_FA bins at iBin*(255/bins) while the network emits logits, so only
    bin 0 -- threshold 0, i.e. sigmoid > 0.5 -- is meaningful. That is the same convention the
    IoU metric uses internally (predict = output > 0), so bin 0 is the self-consistent choice.
    """
    net.eval()
    iou = mIoU(1); pdfa = PD_FA(1, bins, size); roc = ROCMetric(1, bins)
    iou.reset(); pdfa.reset(); roc.reset()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        _, pred = net(x, False)                     # warm_flag off -> single output head
        iou.update(pred, y)
        roc.update(pred, y)
        for i in range(pred.shape[0]):
            pdfa.update(pred[i, 0], y[i, 0])
    _, mean_iou = iou.get()
    fa, pd = pdfa.get(n_imgs)
    return {"IoU": float(mean_iou),
            "Pd": [float(v) for v in pd],
            "Fa": [float(v) for v in fa]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["single", "temporal", "difference"], required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--tmax", type=int, default=0)
    ap.add_argument("--warm-epoch", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)      # 31 GB RAM, no swap
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seed_all(args.seed)
    idx = ROOT / "datasets" / "MIRSDT_dnanet" / "single" / "idx"   # id lists only; shared
    out = ROOT / "irstd" / "mshnet_runs" / f"{args.variant}{args.frames}f_s{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "results.json"
    if done.exists():
        print(f"already complete: {done}")
        return

    tr_ids, va_ids, te_ids = (ids(idx / f"{s}.txt") for s in ("train", "val", "test"))
    rep = (args.variant == "single")
    dif = (args.variant == "difference")
    mk = lambda i, tr: MultiFrameSet(i, frames=args.frames, stride=args.stride,
                                     base_size=args.size, crop_size=args.size,
                                     train=tr, replicate=rep, difference=dif)
    tl = D.DataLoader(mk(tr_ids, True), batch_size=args.batch, shuffle=True,
                      num_workers=args.workers, drop_last=True)
    vl = D.DataLoader(mk(va_ids, False), batch_size=args.batch,
                      num_workers=args.workers, drop_last=False)
    sl = D.DataLoader(mk(te_ids, False), batch_size=args.batch,
                      num_workers=args.workers, drop_last=False)
    print(f"MSHNet {args.variant} {args.frames}f s{args.seed}: "
          f"train {len(tr_ids)} | val {len(va_ids)} | test {len(te_ids)} | rep={rep} diff={dif}")

    net = MSHNet(input_channels=args.frames).cuda()
    print(f"  params = {sum(p.numel() for p in net.parameters()):,}")
    lossf = SLSIoULoss()
    down = nn.MaxPool2d(2, 2)
    opt = torch.optim.Adagrad(net.parameters(), lr=args.lr)
    tmax = args.tmax if args.tmax > 0 else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=1e-5)

    hist, best_val, best_ep, t0 = [], -1.0, -1, time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        tot, nb = 0.0, 0
        warm = ep > args.warm_epoch                      # authors' warm-up schedule
        for x, y in tl:
            x, y = x.cuda(), y.cuda()
            masks, pred = net(x, warm)
            loss = lossf(pred, y, args.warm_epoch, ep)
            lab = y
            for j, m in enumerate(masks):
                if j > 0:
                    lab = down(lab)
                loss = loss + lossf(m, lab, args.warm_epoch, ep)
            loss = loss / (len(masks) + 1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        v = evaluate(net, vl, len(va_ids), args.size)
        hist.append({"epoch": ep, "train_loss": tot / max(1, nb), "val_IoU": v["IoU"]})
        if v["IoU"] > best_val:                          # SELECTION ON VAL, never on test
            best_val, best_ep = v["IoU"], ep
            torch.save({"epoch": ep, "state_dict": net.state_dict(),
                        "val_IoU": v["IoU"]}, out / "best.pth")
        print(f"  ep {ep:3d}/{args.epochs} loss {tot/max(1,nb):.4f} "
              f"val IoU {v['IoU']:.4f} best {best_val:.4f}(ep{best_ep}) "
              f"{(time.time()-t0)/60:.1f}m", flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=2))

    net.load_state_dict(torch.load(out / "best.pth", map_location="cuda")["state_dict"])
    test = evaluate(net, sl, len(te_ids), args.size)
    done.write_text(json.dumps({
        "model": "MSHNet", "variant": args.variant, "frames": args.frames,
        "stride": args.stride, "seed": args.seed, "epochs": args.epochs, "tmax": tmax,
        "selected_epoch": best_ep, "val_IoU_at_selection": best_val, "test": test,
        "minutes": round((time.time() - t0) / 60, 1),
        "n_train": len(tr_ids), "n_val": len(va_ids), "n_test": len(te_ids),
        "note": "Selected on held-out val sequences, not test. Report Pd/Fa at bin 0.",
    }, indent=2))
    (out / "done").mkdir(exist_ok=True)
    (out / "done" / "ok.txt").write_text(f"test IoU {test['IoU']:.4f}\n")
    print(f"\nTEST MSHNet {args.variant}{args.frames}f s{args.seed}: "
          f"IoU={test['IoU']:.4f} Pd={test['Pd'][0]:.4f} Fa={test['Fa'][0]:.3e}")


if __name__ == "__main__":
    main()
