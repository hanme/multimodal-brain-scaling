"""Re-score the mTRF (Workstream A) fit quality with B-IDENTICAL methodology, for a clean
head-to-head against the attention encoder.

The number `insilico_mmn.py` prints comes from a light eval (n_eval_time_samples random points).
The attention encoder instead scores EVERY valid time on EVERY held-out window, concatenated,
correlated along time (`engine_temporal.score_heldout`). This script makes the mTRF do exactly
that, so A and B fit-quality r are apples-to-apples on the same parcels / splits.
"""

import os
import sys
import json
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insilico_mmn import (  # noqa: E402
    build_parcels, fit_mapping, predict_timecourse, FS, TIME_STEP_MS,
)
from eeg_targets import load_split_targets  # noqa: E402
from mbs.evaluation.utils.evaluation_helpers import load_layer_features  # noqa: E402
from mbs.evaluation.evaluate_features_mtrf import (  # noqa: E402
    lags_in_bins, highpass_along_time, pearson_along_time,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True)
    p.add_argument("--neural", required=True)
    p.add_argument("--layer", default="blocks.10")
    p.add_argument("--nc_r_threshold", type=float, default=0.2)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--n_train_time_samples", type=int, default=200)
    p.add_argument("--alpha_log_min", type=float, default=1.0)
    p.add_argument("--alpha_log_max", type=float, default=7.0)
    p.add_argument("--alpha_n", type=int, default=25)
    p.add_argument("--out", default="")
    args = p.parse_args()

    lags = lags_in_bins(0.0, args.lag_max_ms, TIME_STEP_MS, TIME_STEP_MS)
    parcels = build_parcels(Path(args.neural), args.nc_r_threshold)
    names = [pp[0] for pp in parcels]
    nc = np.array([pp[2] for pp in parcels], np.float32)

    fit_args = SimpleNamespace(
        layer=args.layer, broderick_features=args.features_dir, broderick_neural=args.neural,
        highpass_hz=args.highpass_hz, n_train_time_samples=args.n_train_time_samples,
        alpha_log_min=args.alpha_log_min, alpha_log_max=args.alpha_log_max, alpha_n=args.alpha_n,
        eval_heldout=False, n_eval_time_samples=400)
    model, mu, sd, _ = fit_mapping(fit_args, lags, parcels)

    feats_all, id_map = load_layer_features(args.layer, features_folder=Path(args.features_dir))
    eeg, feats = load_split_targets(args.neural, feats_all.astype(np.float32), id_map, parcels, "test")
    hp_eeg = highpass_along_time(eeg, FS, args.highpass_hz)

    preds, t_idx = [], None
    for w in range(eeg.shape[0]):
        t_idx, pred = predict_timecourse(feats[w], model, mu, sd, lags, args.highpass_hz)
        preds.append(pred)
    Yhat = np.concatenate(preds, 0)                              # [n_win*(T-Lmax), P]
    Y = hp_eeg[:, t_idx, :].reshape(-1, hp_eeg.shape[2])         # aligned actual
    r = pearson_along_time(Y, Yhat)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_nc = np.where(nc > 0, r / nc, np.nan)

    print(f"\nmTRF fit quality (B-identical scoring) — {Path(args.neural).name}, layer {args.layer}, "
          f"{eeg.shape[0]} test windows, {Yhat.shape[0]} along-time samples")
    print(f"{'parcel':10s} {'NC r':>5s} {'r':>7s} {'r/NC':>7s}")
    for nm, rr, rn, cc in zip(names, r, r_nc, nc):
        print(f"{nm:10s} {cc:5.2f} {rr:7.3f} {rn:7.3f}")
    print(f"{'MEAN':10s} {nc.mean():5.2f} {np.nanmean(r):7.3f} {np.nanmean(r_nc):7.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"parcels": names, "nc_r": nc.tolist(), "r": r.tolist(), "r_nc": r_nc.tolist(),
                   "n_test_windows": int(eeg.shape[0]), "n_samples": int(Yhat.shape[0]),
                   "layer": args.layer, "scoring": "along_time_all_windows"},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
