"""Held-out fit-quality BAR figure: per-parcel out-of-sample r, one panel per model (D2).

Companion to plot_fit_quality.py. Where that script overlays recorded vs predicted EEG time
courses for ONE model on ONE test window, this reduces the whole held-out TEST split to a single
number per parcel — the out-of-sample Pearson r — and shows it as a bar, so the size ladder
(tiny -> base -> small -> medium) can be compared at a glance.

Same mTRF (Workstream A) as plot_fit_quality / insilico_mmn: parcels are built from the D2 neural
file (surprisal_30s.h5, NC floor r>0.2 -> 5 parcels incl. central), the mapping is fit on TRAIN,
and r is scored on the built-in held-out TEST split. Each model uses its own best layer (selected
elsewhere by mean held-out r over parcels; see outputs/results/<model>-mtrf-parcels-d2/).

  python scripts/plot_fit_quality_bars.py \
    --neural outputs/neural_data/surprisal_30s.h5 \
    --out outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png
"""

import os
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import the sibling script
from insilico_mmn import build_parcels, fit_mapping, TIME_STEP_MS  # noqa: E402
from mbs.evaluation.evaluate_features_mtrf import lags_in_bins  # noqa: E402

# (model, best layer on D2, features dir).  Best layers from <model>-mtrf-parcels-d2 summaries.
SIGF = "/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features"
MODELS = [
    ("whisper-tiny",   "blocks.1",  f"{SIGF}/whisper-tiny-delta-t-surprisal/merged"),
    ("whisper-base",   "blocks.5",  "outputs/features/whisper-base-delta-t-surprisal/merged"),
    ("whisper-small",  "blocks.10", f"{SIGF}/whisper-small-delta-t-surprisal/merged"),
    ("whisper-medium", "blocks.22", f"{SIGF}/whisper-medium-delta-t-surprisal/merged"),
]

# fixed parcel order + colours so bars line up across panels
PARCEL_ORDER = ["frontal", "central", "temporal", "parietal", "occipital"]
PARCEL_COLOR = {
    "frontal": "tab:blue", "central": "tab:orange", "temporal": "tab:green",
    "parietal": "tab:red", "occipital": "tab:purple",
}


def heldout_r(neural, features_dir, layer, args):
    """Fit the mTRF for one model/layer and return {parcel: held-out r}."""
    lags = lags_in_bins(0.0, args.lag_max_ms, TIME_STEP_MS, TIME_STEP_MS)
    parcels = build_parcels(Path(neural), args.nc_r_threshold)
    fit_args = SimpleNamespace(
        layer=layer, broderick_features=features_dir, broderick_neural=neural,
        highpass_hz=args.highpass_hz, n_train_time_samples=args.n_train_time_samples,
        alpha_log_min=args.alpha_log_min, alpha_log_max=args.alpha_log_max, alpha_n=args.alpha_n,
        eval_heldout=True, n_eval_time_samples=args.n_eval_time_samples)
    _, _, _, ev = fit_mapping(fit_args, lags, parcels)
    assert ev, "held-out TEST split empty"
    return {n: float(r) for n, r in zip(ev["parcels"], ev["r"])}, ev["n_test_windows"]


def main():
    p = argparse.ArgumentParser(description="Per-parcel held-out r bars, one panel per model.")
    p.add_argument("--neural", default="outputs/neural_data/surprisal_30s.h5")
    p.add_argument("--nc_r_threshold", type=float, default=0.2)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--n_train_time_samples", type=int, default=200)
    p.add_argument("--n_eval_time_samples", type=int, default=400)
    p.add_argument("--alpha_log_min", type=float, default=1.0)
    p.add_argument("--alpha_log_max", type=float, default=7.0)
    p.add_argument("--alpha_n", type=int, default=25)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results = []  # (model, layer, {parcel:r}, n_test)
    for model, layer, fdir in MODELS:
        print(f"\n=== {model} [{layer}] ===")
        if not Path(fdir).exists():
            print(f"  SKIP — features dir missing: {fdir}")
            continue
        r_by_parcel, n_test = heldout_r(args.neural, fdir, layer, args)
        results.append((model, layer, r_by_parcel, n_test))

    assert results, "no models produced results"
    n = len(results)
    ymax = max(max(rbp.values()) for _, _, rbp, _ in results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2), sharey=True, squeeze=False)
    for ax, (model, layer, rbp, n_test) in zip(axes[0], results):
        names = [pn for pn in PARCEL_ORDER if pn in rbp]
        vals = [rbp[pn] for pn in names]
        cols = [PARCEL_COLOR[pn] for pn in names]
        bars = ax.bar(range(len(names)), vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:+.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{model}\nbest layer {layer}  ({n_test} test win.)", fontsize=9)
        ax.set_ylim(0, ymax * 1.18)
    axes[0][0].set_ylabel("held-out test Pearson r")
    fig.suptitle("Fit quality — mTRF held-out test r per parcel (D2 = Cortical Surprisal)\n"
                 f"5 parcels (NC floor r>{args.nc_r_threshold}), {args.highpass_hz} Hz HP, "
                 f"lags 0–{args.lag_max_ms:.0f} ms, best layer per model", fontsize=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
