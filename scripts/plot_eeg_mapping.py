"""Figures for the model->EEG mapping sweep (eeg_mapping_sweep.py outputs).

Two figures per target level (parcels | electrodes), across the model size ladder:
  A. layer-selection curves  — mean CV-on-train r vs normalized model depth, one line per model,
     the CV-chosen layer marked; held-out TEST r overlaid (dashed) as an honesty check.
  B. held-out test-r bars    — one panel per model, a bar per target at that model's CHOSEN layer
     (per-parcel for parcels; per-electrode, sorted by reliability, for electrodes). No whiskers.

  python scripts/plot_eeg_mapping.py --results_dir outputs/results/eeg_mapping --target_level parcels
"""

import json
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]
PARCEL_COLOR = {"frontal": "tab:blue", "central": "tab:orange", "temporal": "tab:green",
                "parietal": "tab:red", "occipital": "tab:purple"}


def load_runs(results_dir, level):
    runs = {}
    for f in glob.glob(f"{results_dir}/*.json"):
        d = json.load(open(f))
        if d.get("target_level") == level:
            runs[d["model_id"]] = d
    return [runs[m] for m in ORDER if m in runs] + \
           [runs[m] for m in runs if m not in ORDER]


def plot_layer_curves(runs, level, out):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    for k, d in enumerate(runs):
        col = cmap(k / max(len(runs) - 1, 1))
        x = d["positions"]
        cv = d["cv_score_by_layer"]
        test = [float(np.nanmean(t)) for t in d["test_r_by_layer"]]
        ci = d["chosen_idx"]
        ax.plot(x, cv, "-o", color=col, ms=4, lw=1.6, label=f"{d['model_id']} (CV)")
        ax.plot(x, test, "--", color=col, lw=1.0, alpha=0.7)
        ax.scatter([x[ci]], [cv[ci]], s=140, facecolors="none", edgecolors=col, linewidths=2, zorder=5)
    ax.set_xlabel("normalized model depth (0 = first block, 1 = last)")
    ax.set_ylabel("mean Pearson r over targets")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"Layer selection — {level} (solid = CV-on-train, dashed = held-out test;\n"
                 f"circle = CV-chosen layer)  D2", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_test_bars(runs, level, out):
    n = len(runs)
    ymax = max(max(d["test_r_chosen"]) for d in runs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.6), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], runs):
        names, vals = d["targets"], d["test_r_chosen"]
        if level == "parcels":
            cols = [PARCEL_COLOR.get(nm, "grey") for nm in names]
            ax.bar(range(len(names)), vals, color=cols)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.002, f"{v:+.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        else:  # electrodes: many bars, sorted by reliability (json order), thin ticks
            ax.bar(range(len(names)), vals, color="tab:blue")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylim(min(0, min(vals) * 1.1), ymax * 1.18)
        ax.set_title(f"{d['model_id']}\nchosen {d['chosen_layer']}  "
                     f"(mean r={np.mean(vals):+.3f})", fontsize=9)
    axes[0][0].set_ylabel("held-out test Pearson r")
    fig.suptitle(f"Held-out test fit quality at the CV-chosen layer — {level}  (D2)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="outputs/results/eeg_mapping")
    p.add_argument("--target_level", choices=["parcels", "electrodes"], required=True)
    p.add_argument("--out_dir", default="outputs/figures/eeg_mapping")
    args = p.parse_args()

    runs = load_runs(args.results_dir, args.target_level)
    assert runs, f"no {args.target_level} JSONs in {args.results_dir}"
    od = Path(args.out_dir)
    plot_layer_curves(runs, args.target_level, od / f"layer_selection__{args.target_level}__D2.png")
    plot_test_bars(runs, args.target_level, od / f"test_fit_quality__{args.target_level}__D2.png")


if __name__ == "__main__":
    main()
