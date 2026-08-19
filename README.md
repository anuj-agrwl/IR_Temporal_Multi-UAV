# Temporal channel stacking for infrared small-target detection

Code accompanying the paper:

> A. Agrawal and R. Rohilla, "Temporal channel stacking for infrared small-target detection:
> An order-independent difference cue."

The study asks what a conventional convolutional detector extracts from stacked temporal
frames. The network is held fixed and **only the contents of its input channels change**, so
parameter count, arithmetic cost and latency are identical across arms and any measured
difference is attributable to temporal information alone.

## Data

This repository contains **no image data**. NUDT-MIRSDT must be obtained from its original
authors, who distribute it from the download links in their own repository:

- repository (code and download links):
  https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM
- the 120 sequences themselves, Google Drive:
  https://drive.google.com/file/d/1mjFJmsaNoLhALaaL1_3N4qpAPCiBCqXy/view
  (the same README also offers a BaiduYun mirror)

Place the extracted `NUDT-MIRSDT/` tree under `datasets/`. Every build script reads from
`datasets/NUDT-MIRSDT/Sequence<N>/{images,masks}/` and writes only index files and
derived tensors.

If you use the dataset, cite its authors:

```
@article{li2023direction,
  title={Direction-Coded Temporal U-Shape Module for Multiframe Infrared Small Target Detection},
  author={Li, Ruojing and An, Wei and Xiao, Chao and Li, Boyang and Wang, Yingqian and Li, Miao and Guo, Yulan},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}
```

## Input constructions

All arms feed a network whose first layer is unchanged in width:

| Arm | Channels | Purpose |
|---|---|---|
| `single` | `[I_t, I_t, I_t]` | control — current frame replicated |
| `temporal` | `[I_{t-2}, I_{t-1}, I_t]` | consecutive frames |
| `shuffled` | permuted order | tests whether the cue is order-dependent |
| `wide` | `[I_{t-4}, I_{t-2}, I_t]` | longer temporal baseline, stride 2 |
| `difference` | `[I_t, I_t-I_{t-1}, ...]` | difference supplied explicitly |

The control **must** match the temporal arm in channel count. Feeding it a single channel
would make the first convolution differ in width between arms and confound capacity with
temporal content.

## Reproducing

```bash
# 1. Build index files and polygon masks from the official split
python irstd/build_official.py
python irstd/build_seg.py

# 2. YOLO11s-seg arms (3 seeds each)
for s in 0 1 2; do
  python irstd/train_seg.py --variant single   --seed $s --pretrained
  python irstd/train_seg.py --variant temporal --seed $s --pretrained
done

# 3. Purpose-built IRSTD hosts, 8 frames
python irstd/build_dnanet.py
for s in 0 1 2; do
  python irstd/train_dnanet.py --variant single   --frames 8 --seed $s
  python irstd/train_dnanet.py --variant temporal --frames 8 --seed $s
  python irstd/train_mshnet.py --variant temporal --frames 8 --seed $s
done

# 4. Field-standard IRSTD metrics and the paired test
python irstd/score_iou.py --official
python irstd/paired_test.py --device cpu
```

`train_dnanet.py` and `train_mshnet.py` expect the authors' released architectures on the
import path (`irstd/dnanet/`, `irstd/mshnet/`); neither is vendored here.

## Two deviations from the released reference implementations

Both wrappers **deliberately do not reproduce** two properties of the public code, and this
matters when comparing numbers:

1. **Model selection on the test set.** Both released codebases map their validation loader to
   the test list and keep the best epoch by test IoU. Here, training uses `train.txt`,
   selection uses a held-out `val.txt` (sequences in neither official list), and the test set
   is evaluated exactly once at the end.
2. **Completion markers.** A run is complete only when a `results.json` exists, written after
   test evaluation succeeds — never on the presence of a checkpoint.

Architecture, losses, optimiser, augmentation and the authors' own metric code are unchanged.

## Metric convention

`PD_FA` thresholds at `iBin * 255/bins` assuming 0-255 input, but the networks emit logits.
Only **bin 0** (threshold 0, i.e. sigmoid > 0.5) is meaningful, which is the same convention
`batch_intersection_union` uses internally (`predict = output > 0`). All reported Pd/Fa are
bin 0.

## Licence

Code: MIT (see `LICENSE`). The dataset is **not** covered by this licence and is not
redistributed here.
