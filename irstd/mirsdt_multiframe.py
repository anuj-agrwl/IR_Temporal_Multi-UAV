#!/usr/bin/env python3
"""N-frame dataset for NUDT-MIRSDT, for the DNANet arm of Paper 5.

WHY N FRAMES. The 3-channel study is capped at three frames because that is what a standard
RGB network expects -- a genuinely free modification, but an artificial ceiling on how much
temporal evidence the model can see. LVNet, the current best published method on this
benchmark, uses EIGHT frames. DNANet's first convolution is parameterised (`input_channels` is
a constructor argument), so widening the input costs only the first layer's extra weights:

    3 channels -> 3*16*9 =   432 weights
    8 channels -> 8*16*9 = 1,152 weights          (+720 params = +0.015% of ~4.7M)

That is not zero, and the paper must not call it zero -- but it is small enough that the
cost/benefit curve from 1 to 8 frames is the interesting result, with the truly free 3-frame
point marked on it.

WHERE THE FRAMES COME FROM. Each Sequence<N>/images/ holds every raw frame, and those frames
were verified byte-identical to the corresponding planes of Sequence<N>/Mix/*.mat:

    Mix[4] == images/<t>.png,  Mix[3] == images/<t-1>.png,  ...  Mix[0] == images/<t-4>.png

so reading from images/ is equivalent to the .mat route and is not limited to five frames.
Nothing is pre-materialised: an 8-channel copy of the corpus would be ~7 GB, and loading on
the fly keeps frame count and stride configurable at run time.

EDGE HANDLING. Near the start of a sequence the requested history does not exist. The earliest
available frame is REPEATED rather than wrapping to another sequence or dropping the sample --
repeating degrades gracefully towards the single-frame case, whereas wrapping would splice
unrelated scenes into the temporal window and manufacture motion that never happened. The
count of clamped samples is reported by build-time stats so it can be disclosed.

AUGMENTATION replicates DNANet's own `_sync_transform` exactly (random h-flip, random rescale
of the long edge in [0.5, 2.0] x base_size, pad-to-crop, random crop, 50% Gaussian blur), but
applied identically across all N channels so the temporal relationship survives the transform.
Using per-channel random augmentation would destroy the very cue being studied.
"""
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as D

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "NUDT-MIRSDT"

MEAN = np.array([0.485, 0.456, 0.406], np.float32).mean()   # scalar; IR frames are grayscale
STD = np.array([0.229, 0.224, 0.225], np.float32).mean()
_ID = re.compile(r"^(Sequence\d+)_(\d+)$")


def parse_id(img_id):
    m = _ID.match(img_id)
    if not m:
        raise ValueError(f"unparseable id: {img_id}")
    return m.group(1), int(m.group(2))


class MultiFrameSet(D.Dataset):
    """Stack `frames` frames ending at t, sampled every `stride`, as an N-channel tensor."""

    def __init__(self, img_ids, frames=8, stride=1, base_size=256, crop_size=256,
                 train=True, replicate=False, difference=False, diff_gain=8):
        """replicate=True fills every channel with I(t) -- the single-frame CONTROL.

        The control must have the same channel count as the temporal arm, not one channel.
        Otherwise the first convolution would differ in width between the arms and the
        comparison would be confounded at the very first layer: any gain could be attributed
        to capacity rather than to temporal content. With replicate=True both arms are
        identical networks fed identically shaped tensors, and only the CONTENT differs.

        difference=True builds [ I(t), I(t)-I(t-s), I(t)-I(t-2s), ... ] instead of raw frames.
        MOTIVATION, from this study's own controls: permuting the channel order of raw frames
        costs almost nothing in detection but wrecks strict-overlap localisation, which says
        the network is extracting an inter-frame DIFFERENCE and is order-sensitive only when
        deciding which instant to localise to. This mode supplies that difference explicitly
        and encodes the lag in the channel index, addressing both findings at once. Channel
        count, parameter count and cost are unchanged, so it stays comparable to the raw arm.

        Differences are computed in int16 and stored with a +128 offset in uint8, so negative
        values survive; a plain uint8 subtraction would clip every darkening edge to zero and
        silently discard half the signal.
        """
        self.ids = list(img_ids)
        self.frames, self.stride = frames, stride
        self.base_size, self.crop_size, self.train = base_size, crop_size, train
        self.replicate = replicate
        self.difference = difference
        self.diff_gain = diff_gain
        self.clamped = 0

    def __len__(self):
        return len(self.ids)

    def _load_stack(self, seq, t):
        """[I(t-(N-1)*s), ..., I(t)] with the earliest frame repeated where history is short."""
        idir = SRC / seq / "images"
        if self.replicate:                                  # single-frame control
            g = cv2.imread(str(idir / f"{t:05d}.png"), cv2.IMREAD_GRAYSCALE)
            return np.stack([g] * self.frames, axis=-1)
        planes = []
        for k in range(self.frames - 1, -1, -1):
            n = t - k * self.stride
            p = idir / f"{n:05d}.png"
            if not p.exists():                      # before the sequence starts
                n2 = t
                for cand in range(1, t + 1):        # earliest frame that does exist
                    if (idir / f"{cand:05d}.png").exists():
                        n2 = cand
                        break
                p = idir / f"{n2:05d}.png"
                self.clamped += 1
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            planes.append(g)
        stack = np.stack(planes, axis=-1)           # H x W x N, oldest first, current last
        if not self.difference:
            return stack
        # [ I(t), I(t)-I(t-s), I(t)-I(t-2s), ... ]. Channel 0 keeps the raw current frame so
        # appearance is not thrown away; channel k>0 is the lag-k difference, so the lag is
        # encoded by position. int16 then +128 offset: a uint8 subtraction would clip every
        # negative (darkening) difference to zero and discard half the motion signal.
        # GAIN. Measured on this corpus: inter-frame differences have mean|d| ~0.07 and std
        # ~0.4 grey levels (the background is static) but peak ~50 at the target. Encoded at
        # unit gain the channel is almost constant after normalisation -- std 0.015 against
        # 1.29 for a raw channel -- and the network would receive a near-dead input. A gain of
        # 8 maps the background noise to a visible few grey levels and deliberately SATURATES
        # the target, which is desirable here: the target becomes a maximally salient blob
        # rather than a faint one. Saturation loses the target's difference magnitude, which
        # this task does not need; it needs its position.
        cur = stack[..., -1].astype(np.int16)
        out = np.empty_like(stack)
        out[..., 0] = stack[..., -1]                      # keep raw appearance in channel 0
        for k in range(1, self.frames):
            d = cur - stack[..., self.frames - 1 - k].astype(np.int16)
            out[..., k] = np.clip(d * self.diff_gain + 128, 0, 255).astype(np.uint8)
        return out

    def _sync_transform(self, img, mask):
        """DNANet's transform, applied identically to every channel."""
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        cs = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        h, w = mask.shape
        if h > w:
            oh, ow = long_size, int(1.0 * w * long_size / h + 0.5)
        else:
            ow, oh = long_size, int(1.0 * h * long_size / w + 0.5)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        if min(oh, ow) < cs:                                    # pad, bottom/right, zeros
            ph, pw = max(0, cs - oh), max(0, cs - ow)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
            mask = np.pad(mask, ((0, ph), (0, pw)))
        h, w = mask.shape
        y1, x1 = random.randint(0, h - cs), random.randint(0, w - cs)
        img = img[y1:y1 + cs, x1:x1 + cs]
        mask = mask[y1:y1 + cs, x1:x1 + cs]
        if random.random() < 0.5:                               # blur, same kernel all channels
            r = random.random()
            if r > 0.05:
                k = max(3, int(r * 4) * 2 + 1)
                img = cv2.GaussianBlur(img, (k, k), r)
        return img, mask

    def _val_transform(self, img, mask):
        img = cv2.resize(img, (self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, i):
        img_id = self.ids[i]
        seq, t = parse_id(img_id)
        img = self._load_stack(seq, t)
        mask = cv2.imread(str(SRC / seq / "masks" / f"{t:05d}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        img, mask = (self._sync_transform(img, mask) if self.train
                     else self._val_transform(img, mask))
        x = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        y = np.ascontiguousarray(mask)[None].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
