"""In-silico MMN at the ELECTRODE level: feed MMN stimuli to a model, predict per-electrode EEG,
plot the MMN topographically, and auto-flag whether an MMN is present.

Electrode version of insilico_mmn.py. Every helper there is reused as-is by treating each electrode
as a single-member "parcel" (build_electrodes returns (channel, [channel], r)); the mapping, the
held-out eval, and the time-locking are identical. Only the plotting/scoring differs:

  * one figure per method: a 10-20 montage grid of the predicted MMN (deviant - standard) trace per
    electrode (shared y-scale so the topography is readable; 100-240 ms MMN band shaded).
  * a simple per-method MMN verdict: mean MMN amplitude in 100-240 ms over a fronto-central ROI;
    negative beyond --mmn_thresh => "MMN present". Stored + printed alongside the figure.

Sophie's loop: generate a stimulus pair into outputs/mmn_stimuli/<name> + features into
outputs/features/mmn-<name>-delta-t, then add <name> to --methods. Pick ~10 pairs from recent EEG
studies where the last/eliciting tone is physically identical in standard and deviant (see
aux/mmn_screening_plan.md). Run AFTER the per-pair delta_T features exist.

  # default trains on D2 (Cortical Surprisal, human-speech audiobook EEG, healthy fronto-central);
  # --train_neural/--train_features override the dataset (old --broderick_* names still work).
  python scripts/insilico_mmn_electrodes.py --layer blocks.10 --methods method_09,method_12
"""

from pathlib import Path
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mbs.evaluation.evaluate_features_mtrf import lags_in_bins
# electrode/montage builders live in the shared module; the mTRF fit + time-locking are reused
# unchanged from the parcel driver (electrodes are just singleton-member targets).
from eeg_targets import FS, TIME_STEP_MS, build_electrodes, montage_pos
from insilico_mmn import METHODS, fit_mapping, analyze_method

# Fronto-central ROI for the automatic MMN criterion (Umbricht & Krljes 2005: MMN max fronto-central).
FC_ROI = ["Fz", "FCz", "Cz", "FC1", "FC2", "F1", "F2"]


def mmn_metric(res, electrodes, lo_ms, hi_ms, roi):
    """Mean MMN (deviant-standard) amplitude in [lo,hi] ms over the fronto-central ROI electrodes."""
    rel, diff = res["rel_ms"], res["diff_b"]
    win = (rel >= lo_ms) & (rel <= hi_ms)
    idx = [i for i, (ch, _, _) in enumerate(electrodes) if ch in roi]
    if not idx or not win.any():
        return float("nan"), []
    used = [electrodes[i][0] for i in idx]
    return float(diff[np.ix_(win, idx)].mean()), used


def plot_topo(method, label, direction, res, electrodes, args, amp, roi_used, present, out_path):
    rel, diff = res["rel_ms"], res["diff_b"]
    win = (rel >= -args.win_pre_ms) & (rel <= args.win_post_ms)
    x = rel[win]
    ymax = float(np.nanmax(np.abs(diff[win]))) or 1.0

    fig = plt.figure(figsize=(11, 11))
    for i, (ch, _, r) in enumerate(electrodes):
        px, py = montage_pos(ch)
        ax = fig.add_axes([0.5 + 0.42 * px - 0.045, 0.5 + 0.42 * py - 0.03, 0.09, 0.06])
        in_roi = ch in roi_used
        ax.plot(x, diff[win, i], color="tab:red" if in_roi else "tab:blue", lw=1.1)
        ax.axvspan(args.mmn_lo_ms, args.mmn_hi_ms, color="orange", alpha=0.12)
        ax.axvline(0, color="k", ls=":", lw=0.5)
        ax.axhline(0, color="grey", lw=0.4)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        ax.set_title(ch, fontsize=7, pad=1, color="firebrick" if in_roi else "black")
    verdict = "MMN PRESENT" if present else "no MMN"
    fig.suptitle(
        f"In-silico MMN (electrodes) — {method} ({label}, {direction})  |  layer {args.layer}\n"
        f"predicted deviant - standard per electrode (red = fronto-central ROI); "
        f"shaded = {args.mmn_lo_ms:.0f}-{args.mmn_hi_ms:.0f} ms band\n"
        f"ROI mean amp = {amp:+.3g}  ->  {verdict}  (thresh {-args.mmn_thresh:+.3g})",
        fontsize=11, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}   [{verdict}, ROI amp {amp:+.3g}]")


def main():
    p = argparse.ArgumentParser()
    # The EEG dataset to FIT the mapping on. Default = D2 (Cortical Surprisal): a human-speech
    # audiobook EEG with healthy fronto-central channels (Broderick/D1 had ~zero central NC).
    # fit_mapping/analyze_method (imported from insilico_mmn.py) read args.train_*; --broderick_*
    # kept as back-compat aliases.
    p.add_argument("--train_features", "--broderick_features", dest="train_features",
                   default="/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features/"
                           "whisper-small-delta-t-surprisal/merged",
                   help="features dir of the EEG dataset to fit on (default: D2/surprisal, whisper-small)")
    p.add_argument("--train_neural", "--broderick_neural", dest="train_neural",
                   default="outputs/neural_data/surprisal_30s.h5",
                   help="EEG HDF5 to fit + held-out-eval the mapping on (default: D2 = Cortical Surprisal)")
    p.add_argument("--mmn_features_root", default="outputs/features")
    p.add_argument("--stimuli_root", default="outputs/mmn_stimuli")
    p.add_argument("--layer", default="blocks.10")
    p.add_argument("--methods", default="all", help="comma-sep stim-dir names, or 'all'")
    p.add_argument("--nc_r_threshold", type=float, default=0.2)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=500.0)
    p.add_argument("--n_train_time_samples", type=int, default=120)
    p.add_argument("--eval_heldout", type=lambda s: s.lower() not in ("0", "false", "no"), default=True)
    p.add_argument("--n_eval_time_samples", type=int, default=400)
    p.add_argument("--alpha_log_min", type=float, default=1.0)
    p.add_argument("--alpha_log_max", type=float, default=7.0)
    p.add_argument("--alpha_n", type=int, default=25)
    p.add_argument("--win_pre_ms", type=float, default=150.0)
    p.add_argument("--win_post_ms", type=float, default=400.0)
    p.add_argument("--mmn_lo_ms", type=float, default=100.0, help="MMN scoring window start")
    p.add_argument("--mmn_hi_ms", type=float, default=240.0, help="MMN scoring window end")
    p.add_argument("--mmn_thresh", type=float, default=0.0,
                   help="ROI mean amp must be < -thresh to count as an MMN (0 = any negativity)")
    p.add_argument("--out_dir", default="outputs/figures/insilico_mmn_electrodes")
    p.add_argument("--data_dir", default="outputs/insilico_mmn_predictions")
    args = p.parse_args()

    lags = lags_in_bins(0.0, args.lag_max_ms, TIME_STEP_MS, TIME_STEP_MS)

    print(f"Building electrodes (NC floor r>{args.nc_r_threshold}):")
    electrodes = build_electrodes(Path(args.train_neural), args.nc_r_threshold)
    roi = [c for c in FC_ROI if any(c == e[0] for e in electrodes)]
    print(f"  fronto-central ROI used for the auto criterion: {roi}")

    # fit the model->EEG mapping ONCE for this layer (electrodes as targets), apply to every method
    model, mu, sd, _ = fit_mapping(args, lags, electrodes)

    if args.methods == "all":
        run = METHODS
    else:
        reg = {m[0]: m for m in METHODS}
        run = [reg.get(w.strip(), (w.strip(), w.strip(), "")) for w in args.methods.split(",")]

    out_dir = Path(args.out_dir)
    data_path = Path(args.data_dir) / f"electrode_predictions__{args.layer}.h5"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(data_path, "w")
    h5.attrs.update(dict(layer=args.layer, highpass_hz=args.highpass_hz, fs=FS,
                         time_step_ms=TIME_STEP_MS, nc_r_threshold=args.nc_r_threshold,
                         mmn_lo_ms=args.mmn_lo_ms, mmn_hi_ms=args.mmn_hi_ms, fc_roi=",".join(roi),
                         note=("Per-ELECTRODE RAW predicted EEG. time_ms=0 = final/critical-tone onset. "
                               "MMN = deviant_mean - standard. roi_mmn_amp = mean diff in the MMN "
                               "window over the fronto-central ROI; negative => MMN.")))
    h5.create_dataset("electrodes", data=np.array([e[0] for e in electrodes], dtype="S8"))
    h5.create_dataset("electrode_nc_r", data=np.array([e[2] for e in electrodes], np.float32))

    summary = []
    for method, label, direction in run:
        feat_dir = Path(args.mmn_features_root) / f"mmn-{method}-delta-t"
        stim_dir = Path(args.stimuli_root) / method
        if not feat_dir.exists():
            print(f"  {method}: feature dir {feat_dir} missing -> skipped")
            continue
        res = analyze_method(method, feat_dir, stim_dir, model, mu, sd, lags, electrodes, args)
        if res is None:
            continue
        amp, roi_used = mmn_metric(res, electrodes, args.mmn_lo_ms, args.mmn_hi_ms, roi)
        present = bool(amp < -args.mmn_thresh)
        out_path = out_dir / f"insilico_mmn_electrodes__{method}__{args.layer}.png"
        plot_topo(method, label, direction, res, electrodes, args, amp, roi_used, present, out_path)

        g = h5.create_group(method)
        g.attrs.update(dict(context_final=label, direction=direction,
                            final_tone_onset_s=res["final_s"], n_deviants=len(res["dev_ids"]),
                            roi_mmn_amp=amp, mmn_present=present))
        g.create_dataset("time_ms", data=res["rel_ms"])
        g.create_dataset("standard", data=res["std_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_mean", data=res["dev_raw"], compression="gzip", compression_opts=4)
        summary.append((method, label, amp, present))
    h5.close()

    print("\n=== MMN screen summary (layer "
          f"{args.layer}, ROI {roi}, {args.mmn_lo_ms:.0f}-{args.mmn_hi_ms:.0f} ms) ===")
    for method, label, amp, present in summary:
        print(f"  {method:<18} {label:<14} ROI amp {amp:+.3g}   {'MMN' if present else '-'}")
    if summary:
        n_mmn = sum(p for *_, p in summary)
        print(f"  -> {n_mmn}/{len(summary)} pairs show an MMN")
    print(f"Wrote electrode predictions -> {data_path}")


if __name__ == "__main__":
    main()
