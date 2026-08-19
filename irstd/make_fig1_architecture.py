#!/usr/bin/env python3
"""Figure 1 for the manuscript.

Design notes:
  * the two arms are drawn side by side, so the ONLY difference (channel contents) is
    visually the only difference
  * real frames from the held-out test set, plus magnified crops that show the actual
    inter-frame target displacement -- this is the paper's premise, shown rather than asserted
  * the target marker and the crop centre come from the ACTUAL polygon label file, never
    placed by eye
  * the PAN-FPN neck is drawn as two explicit passes (top-down semantics, bottom-up
    localisation) rather than one undifferentiated block
  * the mask-prototype branch is shown, since this paper is segmentation, not detection
  * an explicit "identical in both arms" cost bar, which is the paper's central claim

INTEGRITY: every pixel of image data here is read from disk. Nothing is drawn to look better
than it is. Contrast stretch is a fixed global percentile applied identically to all three
frames -- no per-frame adjustment that could exaggerate the motion.

Run from Object_Detection/:
    python ../Papers/5_IRTemporal_InfraredPhysics/make_fig1_architecture.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OD = ROOT.parent.parent / "Object_Detection"
OUT = ROOT / "figures" / "Figure_1.png"

# Sequence87_00050 has the largest measured 2-frame target displacement in the test split
# (6.66 px). It is shown BECAUSE it is legible, not because it is typical -- the test-set
# median is 1.90 px, and the caption in the manuscript states both. Selecting a clear example
# and disclosing that it is a clear example is honest; showing it as representative is not.
SAMPLE = "Sequence87_00050"
DISP_PX, MEDIAN_PX = 6.66, 1.90
IMDIR = OD / "datasets/MIRSDT_seg/temporal/images/test"
LABDIR = OD / "datasets/MIRSDT_seg/temporal/labels/test"
IMG = IMDIR / f"{SAMPLE}.png"

INK, MUTED, FAINT = "#1a1a1a", "#5c5c5c", "#9a9a9a"
C_SINGLE, C_TEMPORAL = "#b3541e", "#1f6f8b"
C_NET, C_COST = "#dce6ec", "#eef2f4"
C_TD, C_BU = "#7a5c8f", "#3f7d4f"
CROP = 34                             # half-width of the magnified crop, in source pixels


def centroid(stem, w, h):
    """Target centroid in pixels, read from that frame's OWN polygon label."""
    v = np.array([float(x) for x in (LABDIR / f"{stem}.txt").read_text().split()[1:]])
    v = v.reshape(-1, 2)
    return v[:, 0].mean() * w, v[:, 1].mean() * h


def load():
    im = cv2.imread(str(IMG))
    b, g, r = cv2.split(im)           # plane0=I(t-2), plane1=I(t-1), plane2=I(t)
    frames = [b, g, r]
    lo, hi = np.percentile(np.stack(frames), [1, 99.7])   # ONE stretch for all three frames
    frames = [np.clip((f.astype(float) - lo) / (hi - lo), 0, 1) for f in frames]
    h, w = im.shape[:2]

    # Each frame is marked at ITS OWN target position, not at frame t's. Marking all three at
    # the frame-t centroid would draw a moving target as a stationary one -- the exact claim
    # the figure exists to support, faked.
    seq, n = SAMPLE.rsplit("_", 1)
    cents = [centroid(f"{seq}_{int(n) - k:05d}", w, h) for k in (2, 1, 0)]
    return frames, (h, w), cents


def styled(ax, colour=MUTED, lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(colour); s.set_linewidth(lw)


def box(ax, x, y, w, h, text, fc, fs=7.4, ec=None, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.010",
                                fc=fc, ec=ec or MUTED, lw=0.7, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, colour=MUTED, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7.5, lw=lw,
                                 color=colour, connectionstyle=f"arc3,rad={rad}", zorder=5))


def stack_glyph(ax, x, y, w, h, planes, colour, title, caption):
    """Three offset planes. Labels sit in each plane's EXPOSED strip, so they never collide."""
    off, dy = 0.026, 0.022
    for k in (2, 1, 0):                                   # back plane first
        px, py = x + k * off, y - k * dy
        ax.add_patch(Rectangle((px, py), w, h, fc="white", ec=colour, lw=1.25, zorder=3 + k))
        # Front plane: centred. Back planes: label sits in the strip the front plane does not
        # cover, so the three never collide however the glyph is scaled.
        lx = px + w * 0.5 if k == 0 else px + w - off * 0.5
        ax.text(lx, py + h / 2, planes[k], fontsize=7.4 if k == 0 else 6.6, color=colour,
                ha="center", va="center", zorder=6 + k)
    ax.text(x + w / 2 + off, y + h + 0.030, title, fontsize=8.0, color=colour,
            ha="center", va="bottom", fontweight="bold")
    ax.text(x + w / 2 + off, y - 2 * dy - 0.034, caption, fontsize=6.6, color=MUTED,
            ha="center", va="top", linespacing=1.35)


def main():
    frames, (h, w), cents = load()
    cxr, cyr = cents[2]                       # crop is anchored on the frame-t centroid
    fig = plt.figure(figsize=(7.4, 8.0), dpi=600)
    fig.patch.set_facecolor("white")

    # ================= (a) three real frames + magnified crops ==========================
    fig.text(0.035, 0.988, "(a)", fontsize=10, fontweight="bold", color=INK, va="top")
    fig.text(0.082, 0.988,
             "Three consecutive frames from the held-out test set. In any single frame the "
             "target (circled) is one bright dot",
             fontsize=7.8, color=MUTED, va="top")
    fig.text(0.082, 0.9705,
             "among many. The magnified crops show what no single frame can: it is the one "
             "that moves.",
             fontsize=7.8, color=MUTED, va="top")

    labels = [r"$I_{t-2}$", r"$I_{t-1}$", r"$I_{t}$"]
    xs, wpan = [0.075, 0.375, 0.675], 0.250
    for i, (fr, lb, x0) in enumerate(zip(frames, labels, xs)):
        ring = C_TEMPORAL if i == 2 else FAINT
        cx, cy = cents[i]                     # each frame marked at ITS OWN target position
        ax = fig.add_axes([x0, 0.788, wpan, 0.130])
        ax.imshow(fr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.add_patch(Circle((cx, cy), radius=13, fill=False, ec=ring, lw=1.5))
        ax.set_title(lb, fontsize=9.5, color=INK, pad=3)
        styled(ax)

        axz = fig.add_axes([x0 + wpan / 2 - 0.055, 0.672, 0.110, 0.093])
        y0, x1c = int(cyr - CROP), int(cxr - CROP)
        axz.imshow(fr[y0:y0 + 2 * CROP, x1c:x1c + 2 * CROP], cmap="gray",
                   vmin=0, vmax=1, interpolation="nearest")
        axz.add_patch(Circle((cx - x1c, cy - y0), radius=8, fill=False, ec=ring, lw=1.4))
        axz.axhline(CROP, color=FAINT, lw=0.45, ls=":", alpha=0.8)
        axz.axvline(CROP, color=FAINT, lw=0.45, ls=":", alpha=0.8)
        axz.set_xlim(0, 2 * CROP); axz.set_ylim(2 * CROP, 0)
        styled(axz, ring, 0.9)
    fig.text(0.5, 0.655,
             rf"magnified $\times$4. Crosshair fixed at the frame-$t$ centroid; the marker "
             rf"tracks each frame's own label. Total shift {DISP_PX:.1f} px.",
             fontsize=7.0, color=MUTED, ha="center", va="top", style="italic")
    fig.text(0.5, 0.638,
             rf"This example was chosen for legibility, not typicality: the test-set median "
             rf"two-frame displacement is {MEDIAN_PX:.2f} px.",
             fontsize=6.6, color=FAINT, ha="center", va="top", style="italic")

    # ================= (b) the four input constructions ================================
    axb = fig.add_axes([0.0, 0.370, 1.0, 0.235]); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    axb.axis("off")
    axb.text(0.035, 0.99, "(b)", fontsize=10, fontweight="bold", color=INK, va="top")
    axb.text(0.082, 0.99,
             "The only thing that differs between arms. Every tensor is "
             r"$3\times640\times640$, so capacity, arithmetic",
             fontsize=7.8, color=MUTED, va="top")
    axb.text(0.082, 0.915,
             "cost and the first-layer weight count are identical; only the channel contents "
             "change.",
             fontsize=7.8, color=MUTED, va="top")

    # Four constructions: the baseline, the proposed arm, and the two controls that identify
    # the mechanism. Drawn together because the paper's claim rests on their comparison, not
    # on the first two alone.
    cfg = [
        (0.045, [r"$I_{t}$", r"$I_{t}$", r"$I_{t}$"], C_SINGLE,
         "Single-frame", "no motion\ninformation"),
        (0.285, [r"$I_{t}$", r"$I_{t-1}$", r"$I_{t-2}$"], C_TEMPORAL,
         "Temporal", "consecutive\nframes"),
        (0.525, [r"$I_{t-1}$", r"$I_{t}$", r"$I_{t-2}$"], "#7a5c8f",
         "Shuffled", "same frames,\norder permuted"),
        (0.765, [r"$I_{t}$", r"$I_{t-2}$", r"$I_{t-4}$"], "#3f7d4f",
         "Wide", "stride 2,\nspans 5 frames"),
    ]
    for x0, planes, col, title, cap in cfg:
        stack_glyph(axb, x0, 0.42, 0.135, 0.26, planes, col, title, cap)

    # ================= (c) the network ==================================================
    axc = fig.add_axes([0.0, 0.020, 1.0, 0.360]); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.axis("off")
    axc.text(0.035, 0.99, "(c)", fontsize=10, fontweight="bold", color=INK, va="top")
    axc.text(0.082, 0.99,
             "YOLO11-S-seg, unmodified. All four arms above feed this same network, with the "
             "same weights and the same graph.",
             fontsize=7.8, color=MUTED, va="top")

    yB, yN, yH, bh = 0.655, 0.395, 0.145, 0.115
    bx = [0.175, 0.310, 0.445, 0.580, 0.715]
    lblx = 0.030

    axc.text(lblx, yB + bh / 2, "Backbone", fontsize=7.6, color=INK, va="center",
             fontweight="bold")
    for x, lb, res in zip(bx, ["P1", "P2", "P3", "P4", "P5"],
                          ["320²", "160²", "80²", "40²", "20²"]):
        box(axc, x, yB, 0.100, bh, f"{lb}\n{res}", C_NET, fs=6.9)
    for i in range(4):
        arrow(axc, (bx[i] + 0.100, yB + bh / 2), (bx[i + 1], yB + bh / 2))
    arrow(axc, (0.128, yB + bh / 2), (bx[0], yB + bh / 2), C_TEMPORAL, lw=1.3)
    axc.text(0.128, yB + bh + 0.055, "input\ntensor", fontsize=6.9, color=C_TEMPORAL,
             ha="center", va="center", linespacing=1.3)

    axc.text(lblx, yN + bh / 2, "PAN-FPN\nneck", fontsize=7.6, color=INK, va="center",
             fontweight="bold", linespacing=1.3)
    for x in bx[2:]:
        box(axc, x, yN, 0.100, bh, "", C_NET)
    arrow(axc, (bx[4] + 0.050, yB), (bx[3] + 0.050, yN + bh), C_TD, 1.0, rad=-0.30)
    arrow(axc, (bx[3] + 0.030, yB), (bx[2] + 0.070, yN + bh), C_TD, 1.0, rad=-0.30)
    arrow(axc, (bx[2] + 0.050, yB), (bx[2] + 0.050, yN + bh), C_TD, 1.0)
    arrow(axc, (bx[2] + 0.100, yN + bh * 0.62), (bx[3], yN + bh * 0.62), C_BU, 1.0)
    arrow(axc, (bx[3] + 0.100, yN + bh * 0.62), (bx[4], yN + bh * 0.62), C_BU, 1.0)
    axc.text(0.828, yN + bh * 0.80, "top-down: semantics", fontsize=6.7, color=C_TD, va="center")
    axc.text(0.828, yN + bh * 0.30, "bottom-up: localisation", fontsize=6.7, color=C_BU,
             va="center")

    axc.text(lblx, yH + bh / 2, "Heads", fontsize=7.6, color=INK, va="center",
             fontweight="bold")
    for x, s in zip(bx[2:], ["stride 8\n80²", "stride 16\n40²", "stride 32\n20²"]):
        box(axc, x, yH, 0.100, bh, s, "white", fs=6.9)
    for x in bx[2:]:
        arrow(axc, (x + 0.050, yN), (x + 0.050, yH + bh))
    axc.text(bx[2] + 0.050, yH - 0.030,
             "carries nearly all the signal\nat a 7 px target size",
             fontsize=6.6, color=C_TEMPORAL, ha="center", va="top", linespacing=1.3)

    box(axc, 0.828, yH, 0.140, bh, "mask\nprototypes", "white", fs=6.9)
    box(axc, 0.828, yN, 0.140, bh, "per-target\nmask", C_NET, fs=6.9, bold=True)
    arrow(axc, (bx[4] + 0.100, yH + bh / 2), (0.828, yH + bh / 2))
    arrow(axc, (0.898, yH + bh), (0.898, yN))

    axc.add_patch(FancyBboxPatch((0.100, -0.045), 0.868, 0.080,
                                 boxstyle="round,pad=0.004,rounding_size=0.010",
                                 fc=C_COST, ec=MUTED, lw=0.6, zorder=1))
    axc.text(0.534, -0.005,
             "Identical for all four arms:   10,067,203 parameters   ·   32.8 GFLOPs   ·   "
             "20.5 MB   ·   0 added operators",
             fontsize=7.3, color=INK, ha="center", va="center", fontweight="bold", zorder=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600, facecolor="white")
    im = cv2.imread(str(OUT))
    print(f"wrote {OUT}  ->  {im.shape[1]}x{im.shape[0]} px "
          f"(journal full-width needs >=2244 px)")


if __name__ == "__main__":
    main()
