"""Timeline figure: human labels vs timer baseline vs model (out-of-fold)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CLASSES = ["est. attack", "def→att transition", "est. defense", "att→def transition"]
COLORS = ["#2a6fdb", "#8fd14f", "#d6453d", "#f2a93b"]


def main(run_dir: str):
    run = Path(run_dir)
    z = np.load(run / "oof.npz")
    hz = float(z["hz"])
    gt, probs = z["gt"], z["probs"]
    rows = [("human labels", gt), ("timer τ*", np.where(gt >= 0, z["timer_star"], -1)),
            ("model + HSMM", np.where(gt >= 0, z["hsmm"], -1))]
    t = np.arange(len(gt)) / hz / 60
    cmap = ListedColormap(["#eeeeee"] + COLORS)

    n_panels = 4
    minutes = t[-1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.2 * n_panels), sharex=False)
    chunk = minutes / n_panels
    for i, ax in enumerate(axes):
        lo, hi = int(i * chunk * 60 * hz), int(min((i + 1) * chunk * 60 * hz, len(gt)))
        img = np.stack([r[lo:hi] + 1 for _, r in rows])
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest",
                  extent=[t[lo], t[hi - 1], len(rows) + 0.2, 0.2])
        ax.plot(t[lo:hi], len(rows) + 1.1 - 0.9 * (probs[lo:hi, 1] + probs[lo:hi, 3]), color="k", lw=0.7)
        ax.set_ylim(len(rows) + 1.2, 0.2)
        ax.set_yticks(list(np.arange(len(rows)) + 0.7) + [len(rows) + 0.65])
        ax.set_yticklabels([r[0] for r in rows] + ["P(transition)"], fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    axes[-1].set_xlabel("match time (min)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS] + [plt.Rectangle((0, 0), 1, 1, color="#eeeeee")]
    fig.legend(handles, CLASSES + ["unlabeled"], loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run / "timeline.png", dpi=130)
    print(run / "timeline.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/run_full")
