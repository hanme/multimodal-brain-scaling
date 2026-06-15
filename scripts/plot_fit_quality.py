"""Fit-quality figure: actual vs predicted parcel EEG on a held-out TEST window.

Companion to the in-silico MMN plots. Where the MMN figure shows the mapping APPLIED to tones,
this shows how faithful the mapping is on real held-out speech EEG: for one test window we overlay
the recorded parcel EEG and what the encoder predicts, one panel per parcel. No MMN here — this is
the "is the fit any good?" sanity figure Sophie asked for.

It reuses insilico_mmn.fit_mapping / load_split_parcels / predict_timecourse, so the mapping is the
SAME mTRF (Workstream A) as the MMN figure when given the same --features_dir/--neural/--layer.
(Workstream B / attention encoder: separate driver, fast-follow once it can save+predict.)

Example (whisper-small, D2 = Cortical Surprisal, best layer blocks.10):
  python scripts/plot_fit_quality.py \
    --features_dir /work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features/whisper-small-delta-t-surprisal/merged \
    --neural outputs/neural_data/surprisal_30s.h5 --layer blocks.10 \
    --lag_max_ms 800 --n_train_time_samples 200 \
    --out outputs/figures/fit_quality/fit_quality__whisper-small__d2__blocks.10__mTRF.png
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
from insilico_mmn import (  # noqa: E402
    build_parcels, fit_mapping, load_split_parcels, predict_timecourse, FS, TIME_STEP_MS,
)
from mbs.evaluation.utils.evaluation_helpers import load_layer_features  # noqa: E402
from mbs.evaluation.evaluate_features_mtrf import (  # noqa: E402
    lags_in_bins, highpass_along_time, pearson_along_time,
)


def main():
    p = argparse.ArgumentParser(description="Actual vs predicted parcel EEG on a held-out test window.")
    p.add_argument("--features_dir", required=True, help="mapping (train) features for the layer")
    p.add_argument("--neural", required=True, help="neural HDF5 with train/test splits")
    p.add_argument("--layer", default="blocks.10")
    p.add_argument("--nc_r_threshold", type=float, default=0.2)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--n_train_time_samples", type=int, default=200)
    p.add_argument("--alpha_log_min", type=float, default=1.0)
    p.add_argument("--alpha_log_max", type=float, default=7.0)
    p.add_argument("--alpha_n", type=int, default=25)
    p.add_argument("--window_idx", type=int, default=-1,
                   help="which test window to plot; -1 = the median-correlation (representative) one")
    p.add_argument("--max_seconds", type=float, default=0.0,
                   help="if >0, crop the plotted window to its first N seconds for legibility")
    p.add_argument("--label", default="mTRF", help="method label for the title (e.g. mTRF)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lags = lags_in_bins(0.0, args.lag_max_ms, TIME_STEP_MS, TIME_STEP_MS)

    print(f"Building parcels (NC floor r>{args.nc_r_threshold}) from {args.neural}:")
    parcels = build_parcels(Path(args.neural), args.nc_r_threshold)
    assert parcels, "no parcels survived the NC threshold"
    names = [pp[0] for pp in parcels]

    # Fit the mapping on TRAIN (same call the MMN figure uses); get held-out r for the titles.
    fit_args = SimpleNamespace(
        layer=args.layer, broderick_features=args.features_dir, broderick_neural=args.neural,
        highpass_hz=args.highpass_hz, n_train_time_samples=args.n_train_time_samples,
        alpha_log_min=args.alpha_log_min, alpha_log_max=args.alpha_log_max, alpha_n=args.alpha_n,
        eval_heldout=True, n_eval_time_samples=400)
    model, mu, sd, ev = fit_mapping(fit_args, lags, parcels)
    r_by_parcel = {n: float(r) for n, r in zip(ev["parcels"], ev["r"])} if ev else {}

    # Load held-out TEST windows (recorded parcel EEG + aligned features).
    feats_all, id_map = load_layer_features(args.layer, features_folder=Path(args.features_dir))
    eeg, feats = load_split_parcels(args.neural, feats_all.astype(np.float32), id_map, parcels, "test")
    n_win = eeg.shape[0]
    assert n_win > 0, "no test windows"

    # Predict every window once; pick which to show.
    hp_eeg = highpass_along_time(eeg, FS, args.highpass_hz)              # actual, fit-space [n,T,P]
    preds, t_idx = [], None
    for w in range(n_win):
        t_idx, pred = predict_timecourse(feats[w], model, mu, sd, lags, args.highpass_hz)
        preds.append(pred)
    preds = np.stack(preds, 0)                                           # [n, n_t, P]
    actual = hp_eeg[:, t_idx, :]                                         # [n, n_t, P]

    if args.window_idx >= 0:
        w = args.window_idx
    else:  # median-correlation window = representative, not cherry-picked
        per_win = np.array([np.nanmean(pearson_along_time(actual[i], preds[i])) for i in range(n_win)])
        w = int(np.argsort(per_win)[len(per_win) // 2])
    print(f"Plotting test window {w}/{n_win}  (mean-parcel r this window = "
          f"{np.nanmean(pearson_along_time(actual[w], preds[w])):+.3f})")

    t_s = t_idx * TIME_STEP_MS / 1000.0
    keep = t_s <= (t_s[0] + args.max_seconds) if args.max_seconds > 0 else np.ones_like(t_s, bool)
    x, A, P_ = t_s[keep], actual[w][keep], preds[w][keep]

    n = len(parcels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True, squeeze=False)
    for i, (pname, members, pr) in enumerate(parcels):
        ax = axes[i][0]
        ax.plot(x, A[:, i], color="k", lw=1.4, label="recorded EEG")
        ax.plot(x, P_[:, i], color="tab:red", lw=1.4, alpha=0.85, label="predicted")
        ax.axhline(0, color="grey", lw=0.5)
        rr = r_by_parcel.get(pname, float("nan"))
        ax.set_ylabel(f"{pname}\nNC r={pr:.2f}  ({'+'.join(members)})", fontsize=9)
        ax.set_title(f"{pname}: held-out r = {rr:+.3f}  (over all {n_win} test windows)", fontsize=9)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("time within held-out test window (s)")
    fig.suptitle(
        f"Fit quality — {args.label}: recorded vs predicted parcel EEG, held-out test window {w}\n"
        f"layer {args.layer}, {args.highpass_hz} Hz HP, lags 0–{args.lag_max_ms:.0f} ms  |  "
        f"{Path(args.neural).name}  (each panel its own y-scale)", fontsize=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
