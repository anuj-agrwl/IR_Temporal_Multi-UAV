#!/usr/bin/env python3
"""Train DNANet on NUDT-MIRSDT with a given input-channel construction (Paper 5).

WHY THIS WRAPPER EXISTS rather than calling dnanet/train.py directly.

  1. THEIR SCRIPT SELECTS THE CHECKPOINT ON THE TEST SET. In dnanet/train.py the loader is
     built as `testset = TestSetLoader(..., img_id=val_img_ids)` where val_img_ids comes from
     test.txt, and `save_model` keeps the best epoch by IoU on exactly that set. That is
     test-set model selection. Reproducing it would make every number we report on the test
     split optimistically biased, so this wrapper trains on train.txt, selects on a held-out
     val.txt (the 20 sequences in neither official list), and touches test.txt ONCE at the end.
     The split is by sequence, matching the YOLO study exactly.

  2. THEIR OUTPUT DIRECTORY IS TIMESTAMPED (result/<dataset>_DNANet_<ts>_wDS), so a queue
     runner cannot test idempotently whether a run already finished. This writes a fixed
     directory and a results.json that appears only after the test evaluation has succeeded --
     the same completion-marker discipline the segmentation runs use.

  3. Seeding is made deterministic so the three seeds are genuinely different runs of the same
     recipe rather than three draws from an unseeded RNG.

WHAT IS DELIBERATELY UNCHANGED from the authors' recipe: model (DNANet, ResNet-18 backbone,
deep supervision), loss (SoftIoULoss on every supervised output), optimiser (Adagrad),
metrics (their own mIoU / PD_FA implementations), and the 256x256 evaluation. Only the epoch
budget differs -- their default of 1500 epochs is tuned for ~1300 images and is impractical on
our 8000. Both arms get the identical budget, so the single-vs-temporal delta stays valid; the
absolute figure will sit below their published number and the paper must say so.

    python irstd/train_dnanet.py --variant temporal --seed 0 --epochs 50
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as D
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
DNA = ROOT / "irstd" / "dnanet"
sys.path.insert(0, str(DNA))

from model.load_param_data import load_param                      # noqa: E402
from model.loss import SoftIoULoss                                # noqa: E402
from model.metric import mIoU, PD_FA, ROCMetric                   # noqa: E402
from model.model_DNANet import DNANet, Res_CBAM_block             # noqa: E402
from model.utils import TrainSetLoader, TestSetLoader             # noqa: E402


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def ids(p):
    return [l.strip() for l in Path(p).read_text().split("\n") if l.strip()]


@torch.no_grad()
def evaluate(net, loader, n_imgs, bins=10):
    """IoU / nIoU / Pd / Fa using the AUTHORS' metric code, so numbers are comparable."""
    net.eval()
    iou = mIoU(1); pdfa = PD_FA(1, bins); roc = ROCMetric(1, bins)
    iou.reset(); pdfa.reset(); roc.reset()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        out = net(x)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        iou.update(pred, y)
        roc.update(pred, y)
        pdfa.update(pred[0, 0, :, :], y[0, 0, :, :]) if pred.shape[0] == 1 else [
            pdfa.update(pred[i, 0, :, :], y[i, 0, :, :]) for i in range(pred.shape[0])]
    _, mean_iou = iou.get()
    fa, pd = pdfa.get(n_imgs)
    tpr, fpr, recall, precision = roc.get()
    return {
        "IoU": float(mean_iou),
        "Pd": [float(v) for v in pd],
        "Fa": [float(v) for v in fa],
        "recall": [float(v) for v in recall],
        "precision": [float(v) for v in precision],
    }


def main():
    ap = argparse.ArgumentParser()
    # 3-channel arms come from the pre-built PNG dataset (build_dnanet.py). Arms with
    # --frames != 3 are loaded on the fly from Sequence*/images/ by mirsdt_multiframe.py,
    # because PNG cannot hold more than 4 channels and materialising an 8-channel corpus
    # would cost ~7 GB for no benefit.
    ap.add_argument("--variant", choices=["single", "temporal", "wide"], required=True)
    ap.add_argument("--frames", type=int, default=3,
                    help="input frames = input channels. 3 uses the PNG build; >3 streams "
                         "from images/. The single arm replicates I(t) across all channels so "
                         "both arms have an identical first layer.")
    ap.add_argument("--stride", type=int, default=1,
                    help="frame spacing within the window (1 = consecutive)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    # Early stopping on VAL IoU. The epoch budget is generous so the arms reach convergence
    # rather than being cut off mid-improvement, but a plateaued run should not burn hours:
    # patience ends it. Both arms use the identical budget and identical patience, so the
    # comparison stays fair however many epochs each happens to use.
    ap.add_argument("--patience", type=int, default=15)
    # COSINE SCHEDULE LENGTH. The first 8-frame runs used T_max=100 while early stopping fired
    # at epochs 35-51, so the learning rate never annealed -- training ended while the LR was
    # still high, which is the worst point to stop a segmentation network. Setting tmax equal
    # to the epoch budget (and patience >= budget to disable early stopping) guarantees the
    # schedule completes. Leave tmax=0 to keep the old behaviour of tmax=epochs.
    ap.add_argument("--tmax", type=int, default=0,
                    help="cosine T_max; 0 means use --epochs (i.e. a complete schedule)")
    ap.add_argument("--tag", default="",
                    help="suffix for the run directory, so a re-run under a changed recipe "
                         "does not overwrite or falsely satisfy an earlier run's marker")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)     # 31 GB RAM, no swap -- keep at 4
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    seed_all(args.seed)
    data = ROOT / "datasets" / "MIRSDT_dnanet" / args.variant
    tag = f"{args.variant}_s{args.seed}" if args.frames == 3 else \
          f"{args.variant}{args.frames}f_s{args.seed}"
    tag += args.tag
    out = ROOT / "irstd" / "dnanet_runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    done_marker = out / "results.json"
    if done_marker.exists():
        print(f"already complete: {done_marker}")
        return

    tf = T.Compose([T.ToTensor(),
                    T.Normalize([.485, .456, .406], [.229, .224, .225])])
    tr_ids, va_ids, te_ids = (ids(data / "idx" / f"{s}.txt") for s in ("train", "val", "test"))
    print(f"{args.variant} s{args.seed}: train {len(tr_ids)} | val {len(va_ids)} | test {len(te_ids)}")

    if args.frames == 3:
        trainset = TrainSetLoader(str(data), img_id=tr_ids, base_size=args.size,
                                  crop_size=args.size, transform=tf, suffix=".png")
        valset = TestSetLoader(str(data), img_id=va_ids, base_size=args.size,
                               crop_size=args.size, transform=tf, suffix=".png")
        testset = TestSetLoader(str(data), img_id=te_ids, base_size=args.size,
                                crop_size=args.size, transform=tf, suffix=".png")
    else:
        # N-frame arms stream from Sequence*/images/. "single" replicates I(t) across all N
        # channels, so the control has an identical first layer to the temporal arm and the
        # comparison cannot be attributed to input width.
        from mirsdt_multiframe import MultiFrameSet
        rep = (args.variant == "single")
        mk = lambda i, tr: MultiFrameSet(i, frames=args.frames, stride=args.stride,
                                         base_size=args.size, crop_size=args.size,
                                         train=tr, replicate=rep)
        trainset, valset, testset = mk(tr_ids, True), mk(va_ids, False), mk(te_ids, False)
        print(f"  {args.frames}-frame stack, stride {args.stride}, replicate={rep}")
    tl = D.DataLoader(trainset, batch_size=args.batch, shuffle=True,
                      num_workers=args.workers, drop_last=True)
    vl = D.DataLoader(valset, batch_size=args.batch, num_workers=args.workers, drop_last=False)
    sl = D.DataLoader(testset, batch_size=args.batch, num_workers=args.workers, drop_last=False)

    nb_filter, num_blocks = load_param("three", "resnet_18")
    net = DNANet(num_classes=1, input_channels=args.frames, block=Res_CBAM_block,
                 num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True).cuda()
    nparam = sum(p.numel() for p in net.parameters())
    print(f"  DNANet input_channels={args.frames}  params={nparam:,}")
    opt = torch.optim.Adagrad(net.parameters(), lr=args.lr)
    tmax = args.tmax if args.tmax > 0 else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=1e-5)
    print(f"  cosine T_max={tmax}, epochs={args.epochs}, patience={args.patience}")

    hist, best_val, best_ep = [], -1.0, -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.cuda(), y.cuda()
            outs = net(x)
            loss = sum(SoftIoULoss(o, y) for o in outs) / len(outs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        v = evaluate(net, vl, len(va_ids))
        hist.append({"epoch": ep, "train_loss": tot / max(1, len(tl)), "val_IoU": v["IoU"]})
        if v["IoU"] > best_val:                       # SELECTION ON VAL, never on test
            best_val, best_ep = v["IoU"], ep
            torch.save({"epoch": ep, "state_dict": net.state_dict(), "val_IoU": v["IoU"]},
                       out / "best.pth")
        print(f"  ep {ep:3d}/{args.epochs}  loss {tot/max(1,len(tl)):.4f}  "
              f"val IoU {v['IoU']:.4f}  best {best_val:.4f} (ep {best_ep})  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        if ep - best_ep >= args.patience:
            print(f"  early stop: no val improvement for {args.patience} epochs "
                  f"(best {best_val:.4f} at epoch {best_ep})", flush=True)
            break

    # ---- test touched exactly once, on the val-selected checkpoint ------------------------
    ck = torch.load(out / "best.pth", map_location="cuda")
    net.load_state_dict(ck["state_dict"])
    test = evaluate(net, sl, len(te_ids))
    res = {
        "variant": args.variant, "seed": args.seed, "epochs": args.epochs, "tmax": tmax, "patience": args.patience,
        "batch": args.batch, "size": args.size, "lr": args.lr,
        "selected_epoch": best_ep, "val_IoU_at_selection": best_val,
        "test": test, "minutes": round((time.time() - t0) / 60, 1),
        "n_test": len(te_ids), "n_val": len(va_ids), "n_train": len(tr_ids),
        "note": ("Model selected on the held-out val sequences, NOT on test. Epoch budget "
                 "differs from the authors' 1500-epoch default; both arms share this budget."),
    }
    done_marker.write_text(json.dumps(res, indent=2))
    # Completion marker for run_all_jobs_v2.sh. Its is_done() understands a trailing-slash
    # DIRECTORY marker, so the flag has to be a directory -- and it must be created only here,
    # after the test evaluation has already succeeded. Keying on the run directory itself
    # would be the best.pt trap again: it becomes non-empty at epoch 1.
    (out / "done").mkdir(exist_ok=True)
    (out / "done" / "ok.txt").write_text(
        f"{args.variant} s{args.seed} completed, test IoU {test['IoU']:.4f}\n")
    print(f"\nTEST {args.variant} s{args.seed}: IoU={test['IoU']:.4f} "
          f"Pd@bin5={test['Pd'][5]:.4f} Fa@bin5={test['Fa'][5]:.3e}")
    print(f"wrote {done_marker}")


if __name__ == "__main__":
    main()
