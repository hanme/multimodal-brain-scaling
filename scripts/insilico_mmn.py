"""In-silico MMN: predict parcel-level EEG for MMN tone stimuli via the mTRF mapping.

Pipeline:
  1. Fit a FIR mTRF on the training EEG (default D2; EEG[t] = sum_lag W[lag] . feature[t-lag]) for ONE model
     layer, with electrodes aggregated into coarse 10-20 parcels (Kadir-style clusters).
     Each parcel = raw average over its constituent channels that pass an NC floor
     (Gokce/Kadir precedent: drop channels with reliability r <= --nc_r_threshold before
     fitting; see evaluate_features_committed_layers.py `noise_ceiling > 0.1`).
  2. Apply that single mapping to whisper-base delta_T features of EACH MMN method (classic
     oddball / Definition-1 design: the deviant train's LAST tone differs in frequency from
     the standard's repeating tone -- see METHODS below and
     data/metadata/literature_frequency_intensity_duration_metadata.csv).
  3. Per method: average all 15 deviant trials (N in {3,5,7} x var in {1..5}), take the
     standard, time-lock to the final-tone onset, and z-score dev/std within a baseline window
     (sized to 3x the method's SOA) to get z_dev/z_std; baseline_normalized_peak = min(z_dev -
     z_std) in the 100-240 ms band (see finalize_method()).
  4. Plot a grid: rows = frontal/central/temporal parcels (each its OWN y-scale, annotated
     with NC + member channels), columns = deviant / standard / (deviant - standard), all
     z-scored; the third column is annotated with baseline_normalized_peak, the most-negative
     point of that same plotted line in the shaded band.

The mapping depends only on (layer, parcels), NOT on the MMN method, so it is fit ONCE
per layer and applied to all methods in the loop.

Run AFTER the MMN delta_T features exist (scripts/slurm_mmn_extract.sh) for every method.
"""

from pathlib import Path
import argparse
import csv
import glob
import numpy as np
import h5py
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

from mbs.evaluation.utils.evaluation_helpers import load_layer_features
from mbs.evaluation.evaluate_features_mtrf import (
    lags_in_bins, build_lagged_design, highpass_along_time, sample_time_indices,
    pearson_along_time,
)
# parcels/electrodes + dataset I/O live in the shared, dataset-agnostic module
from eeg_targets import FS, TIME_STEP_MS, build_parcels, load_split_targets, montage_pos
from analyze_mmn_criteria_s5_s6 import load_duration_map
from mmn_criteria_table import compute_criteria_table, CRITERIA_COLUMNS

DURATION_CSV = "data/metadata/literature_frequency_intensity_duration_metadata.csv"

# MMN methods: literature classic-oddball (Definition 1) set, 10 pairs sourced from
# data/metadata/literature_frequency_intensity_duration_metadata.csv. Each method has 1
# standard file (repeating tone) + 15 deviant files (N in {3,5,7} x var in {1..5}); the
# deviant train's LAST tone differs in frequency from the standard's repeating tone (unlike
# the older identity-MMN design this registry used to describe, where the final tone was
# physically identical in std & dev). Tuple = (stimulus-dir name, "standard->deviant" label,
# source citation).
METHODS = [
    # Regular (standard → deviant)
    ("method_75", "1000→1200 Hz", "Karger_2014"),
    ("method_74", "1000→1500 Hz", "Domjan_2012"),
    ("method_72", "1000→1200 Hz", "Bodatsch_2011"),
    ("method_60", "1000→1500 Hz", "Umbricht_2003a"),
    ("method_53", "1000→1200 Hz", "Salisbury_2002a"),
    ("method_55", "1000→2000 Hz", "Shinozaki_2002a"),
    ("method_37", "1000→1050 Hz", "Javitt_2000a"),
    ("method_43", "633→700 Hz",   "Michie_2000b"),
    ("method_44", "633→1000 Hz",  "Michie_2000c"),
    ("method_27", "1000→1064 Hz", "Schall_1999a"),
    # Counterbalanced (standard/deviant frequencies swapped)
    ("method_75_counter", "1200→1000 Hz", "Karger_2014"),
    ("method_74_counter", "1500→1000 Hz", "Domjan_2012"),
    ("method_72_counter", "1200→1000 Hz", "Bodatsch_2011"),
    ("method_60_counter", "1500→1000 Hz", "Umbricht_2003a"),
    ("method_53_counter", "1200→1000 Hz", "Salisbury_2002a"),
    ("method_55_counter", "2000→1000 Hz", "Shinozaki_2002a"),
    ("method_37_counter", "1050→1000 Hz", "Javitt_2000a"),
    ("method_43_counter", "700→633 Hz",   "Michie_2000b"),
    ("method_44_counter", "1000→633 Hz",  "Michie_2000c"),
    ("method_27_counter", "1064→1000 Hz", "Schall_1999a"),
]

DEFAULT_SOA_CSV = "data/metadata/literature_frequency_intensity_duration_metadata.csv"


def load_soa_table(csv_path=DEFAULT_SOA_CSV):
    """method_id (int) -> standard_soa (ms), from the literature stimulus metadata CSV."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            table[int(row["method_id"])] = float(row["standard_soa"])
    return table


def soa_for_method(method, soa_table):
    """'method_37' -> soa_table[37] (ms)."""
    return soa_table[int(method.split("_")[1])]


def detect_final_tone_onset_s(wav_path):
    """Onset time (s) of the LAST tone in the clip (the MMN-critical tone)."""
    x, sr = sf.read(wav_path)
    if x.ndim > 1:
        x = x.mean(1)
    w = int(0.005 * sr)
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    on = env > 0.05 * env.max()
    edges = np.diff(on.astype(int))
    starts = np.where(edges == 1)[0]
    return float(starts[-1] / sr) if len(starts) else None


def fit_mapping(args, lags, parcels):
    """Fit FIR mTRF on the training EEG dataset for one layer -> (model, feat_mu, feat_sd, eval).

    Target columns are the parcels (raw average over NC-surviving member channels). If
    --eval_heldout, also predicts the built-in held-out TEST split (separate audiobook runs)
    and returns per-parcel out-of-sample Pearson r (raw and NC-normalized); else eval=None.
    """
    feats_all, id_map = load_layer_features(args.layer, features_folder=Path(args.train_features))
    feats_all = feats_all.astype(np.float32)  # [n_stim, T, d]

    eeg, feats = load_split_targets(args.train_neural, feats_all, id_map, parcels, "train")

    # high-pass both, standardize features (save stats for the MMN + eval sides)
    feats = highpass_along_time(feats, FS, args.highpass_hz)
    eeg = highpass_along_time(eeg, FS, args.highpass_hz)
    mu = feats.reshape(-1, feats.shape[-1]).mean(0)
    sd = feats.reshape(-1, feats.shape[-1]).std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    feats = (feats - mu) / sd

    rng = np.random.default_rng(0)
    t_idx = sample_time_indices(feats.shape[1], int(lags.max()), args.n_train_time_samples, rng)
    X, Y = build_lagged_design(feats, eeg, lags, t_idx)
    alphas = np.logspace(args.alpha_log_min, args.alpha_log_max, args.alpha_n)
    model = RidgeCV(alphas=alphas, alpha_per_target=True)
    model.fit(X.astype(np.float32), Y.astype(np.float32))
    chosen = np.atleast_1d(model.alpha_)
    print(f"Fitted FIR mapping [{args.layer}]: X{X.shape} -> Y{Y.shape}  "
          f"alpha grid [{alphas.min():.0f}, {alphas.max():.0f}] x{args.alpha_n}  "
          f"chosen={np.array2string(chosen, formatter={'float_kind': lambda v: f'{v:.0f}'})}")
    if np.any(chosen >= alphas.max() * 0.999) or np.any(chosen <= alphas.min() * 1.001):
        print("  WARNING: a chosen alpha is at a grid edge -> widen --alpha_log_min/max.")

    eval_metrics = None
    if args.eval_heldout:
        eval_metrics = evaluate_heldout(args, lags, parcels, model, mu, sd, feats_all, id_map)
    return model, mu, sd, eval_metrics


def evaluate_heldout(args, lags, parcels, model, mu, sd, feats_all, id_map):
    """Score the fitted mapping on the built-in held-out TEST split (separate runs).

    No leakage: features are standardized with TRAIN mu/sd; test windows come from audiobook
    runs absent from train. Returns dict with per-parcel out-of-sample Pearson r (raw + NC-norm).
    """
    eeg, feats = load_split_targets(args.train_neural, feats_all, id_map, parcels, "test")
    if eeg.shape[0] == 0:
        print(f"  held-out eval [{args.layer}]: TEST split empty -> skipped")
        return None
    feats = highpass_along_time(feats, FS, args.highpass_hz)
    eeg = highpass_along_time(eeg, FS, args.highpass_hz)
    feats = (feats - mu) / sd

    rng = np.random.default_rng(1)
    t_idx = sample_time_indices(feats.shape[1], int(lags.max()), args.n_eval_time_samples, rng)
    X, Y = build_lagged_design(feats, eeg, lags, t_idx)
    Yhat = model.predict(X.astype(np.float32))
    r = pearson_along_time(Y, Yhat)                              # [n_parcel]
    nc_r = np.array([p[2] for p in parcels], np.float32)         # parcel reliability (r-scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_nc = np.where(nc_r > 0, r / nc_r, np.nan).astype(np.float32)
    names = [p[0] for p in parcels]
    n_test = eeg.shape[0]
    print(f"  held-out eval [{args.layer}] on {n_test} TEST windows ({X.shape[0]} samples):")
    for nm, rr, rn in zip(names, r, r_nc):
        print(f"    {nm:<10} r={rr:+.3f}   r/NC={rn:+.3f}")
    return dict(parcels=names, r=r, r_nc=r_nc, nc_r=nc_r,
                n_test_windows=int(n_test), n_samples=int(X.shape[0]))


def predict_timecourse(feat_1stim, model, mu, sd, lags, highpass_hz):
    """feat_1stim [T, d] -> (t_idx, predicted EEG [n_t, n_parcel]) over valid output bins."""
    f = highpass_along_time(feat_1stim[None], FS, highpass_hz)[0]
    f = (f - mu) / sd
    T = f.shape[0]
    t_idx = np.arange(int(lags.max()), T)
    X, _ = build_lagged_design(f[None], np.zeros((1, T, 1), np.float32), lags, t_idx)
    return t_idx, model.predict(X.astype(np.float32))   # [n_t, n_parcel]


def finalize_method(method, t_idx, std_raw, dev_preds, dev_ids, stim_dir, soa_ms,
                     baseline_start_mult=-3.0, baseline_end_mult=0.0):
    """Time-lock + baseline one method (A and B sides share this). Returns a dict with both
    the baseline-corrected arrays (for plotting) and the RAW full time courses (for downstream
    MMN metrics), or None.

    All arrays span the full valid window (the entire ~29 s pre-final-tone baseline + post-onset),
    columns = parcels (same order as `parcels`). rel_ms = 0 is the final/critical-tone onset.

    dev_b/std_b/diff_b use mean-only baseline correction, exactly like the old `bc()`, just with
    a baseline window sized by [baseline_start_mult, baseline_end_mult) x the method's SOA instead
    of a fixed --win_pre_ms (default [-3, 0) x SOA); kept around for callers/tests that want the
    un-normalized traces, but no longer what gets plotted.
    z_dev/z_std/z_diff additionally divide by the baseline std (full z-score) -- these are what
    plot_method()/plot_topo() now draw, so the plotted curve's units match the printed peak.
    peak/n7v1_peak are the most-negative point of z_diff (or the N7/var1 deviant's own z-diff)
    in [100, 240] ms.
    """
    dev_stack = np.stack(dev_preds, 0)              # [n_dev, n_t, n_parcel], RAW
    dev_raw = dev_stack.mean(0)                     # [n_t, n_parcel], RAW

    std_wav = sorted(glob.glob(f"{stim_dir}/*standard*.wav"))[0]
    final_s = detect_final_tone_onset_s(std_wav)
    rel_ms = t_idx * TIME_STEP_MS - final_s * 1000.0          # 0 = final-tone onset
    base = (rel_ms >= baseline_start_mult * soa_ms) & (rel_ms < baseline_end_mult * soa_ms)

    def bc(sig):
        return sig - sig[base].mean(0, keepdims=True)
    dev_b, std_b = bc(dev_raw), bc(std_raw)
    diff_b = dev_b - std_b

    def z(sig):
        sdv = sig[base].std(0, keepdims=True)
        sdv = np.where(sdv > 1e-8, sdv, 1.0)
        return bc(sig) / sdv                        # full z-score (now also the plotted MMN trace)

    z_dev, z_std = z(dev_raw), z(std_raw)
    z_diff = z_dev - z_std

    mmn_win = (rel_ms >= 100.0) & (rel_ms <= 240.0)
    peak = z_diff[mmn_win].min(0) if mmn_win.any() else np.full(z_diff.shape[1], np.nan, np.float32)

    n7v1_idx = next((i for i, sid in enumerate(dev_ids)
                      if "n7" in sid.lower() and "var1" in sid.lower()), None)
    if n7v1_idx is not None and mmn_win.any():
        n7v1_peak = (z(dev_stack[n7v1_idx]) - z(std_raw))[mmn_win].min(0)
    else:
        n7v1_peak = np.full_like(peak, np.nan)

    print(f"  {method}: {len(dev_preds)} deviants avg; final tone ~{final_s:.2f}s")
    return dict(rel_ms=rel_ms.astype(np.float32), dev_b=dev_b, std_b=std_b, diff_b=diff_b,
                z_dev=z_dev.astype(np.float32), z_std=z_std.astype(np.float32),
                z_diff=z_diff.astype(np.float32),
                peak=peak.astype(np.float32), n7v1_peak=n7v1_peak.astype(np.float32),
                std_raw=std_raw.astype(np.float32), dev_raw=dev_raw.astype(np.float32),
                dev_stack=dev_stack.astype(np.float32), dev_ids=dev_ids, final_s=final_s)


def analyze_method(method, feat_dir, stim_dir, model, mu, sd, lags, parcels, args, soa_ms):
    """Predict + time-lock one method via the mTRF, then hand off to finalize_method()."""
    mfeats, mid_map = load_layer_features(args.layer, features_folder=Path(feat_dir))
    mfeats = mfeats.astype(np.float32)
    id_by_row = {v: k for k, v in mid_map.items()}

    std_raw, dev_preds, dev_ids, t_idx = None, [], [], None
    for row in range(mfeats.shape[0]):
        sid = str(id_by_row[row])
        t_idx, pred = predict_timecourse(mfeats[row], model, mu, sd, lags, args.highpass_hz)
        if "standard" in sid.lower():
            std_raw = pred
        elif "deviant" in sid.lower():
            dev_preds.append(pred); dev_ids.append(sid)
    if std_raw is None or not dev_preds:
        print(f"  {method}: missing standard or deviants -> skipped")
        return None
    return finalize_method(method, t_idx, std_raw, dev_preds, dev_ids, stim_dir, soa_ms,
                            baseline_start_mult=args.baseline_start_mult,
                            baseline_end_mult=args.baseline_end_mult)


# Parcel-level MMN figures show only these 3 rows (deliverable spec); the mapping itself is
# still fit on all NC-surviving parcels, this only restricts what gets plotted.
PLOT_ROWS = ("frontal", "central", "temporal")


def _draw_criteria_table(fig, criteria_table, row_names, rect):
    """Render the C0-S6 1/0 table for `row_names` (electrode or parcel names) into a dedicated
    axes at `rect` (figure-fraction [left, bottom, width, height]). `criteria_table` is the dict
    returned by mmn_criteria_table.compute_criteria_table."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    cell_text = [[str(criteria_table[name][c]) for c in CRITERIA_COLUMNS] for name in row_names]
    tbl = ax.table(cellText=cell_text, rowLabels=row_names, colLabels=CRITERIA_COLUMNS,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax.text(0.5, -0.05, "1 = TRUE, 0 = FALSE", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, style="italic")


def plot_method(method, label, source, res, parcels, args, out_path,
                 row_filter=PLOT_ROWS, criteria_table=None):
    """row_filter: keep only targets whose name is in this collection (None = keep all of
    `parcels` as passed -- callers filter upstream, e.g. to ["Fz", "FCz"]). criteria_table:
    optional dict from mmn_criteria_table.compute_criteria_table, rendered as an extra panel
    below the grid (keyed by the SAME names left after row_filter)."""
    rel_ms, z_dev, z_std, z_diff, peak = (
        res["rel_ms"], res["z_dev"], res["z_std"], res["z_diff"], res["peak"])
    win = (rel_ms >= -args.win_pre_ms) & (rel_ms <= args.win_post_ms)
    x = rel_ms[win]

    keep = ([i for i, p in enumerate(parcels) if p[0] in row_filter] if row_filter is not None
            else list(range(len(parcels))))
    parcels = [parcels[i] for i in keep]
    z_dev, z_std, z_diff, peak = z_dev[:, keep], z_std[:, keep], z_diff[:, keep], peak[keep]

    n = len(parcels)
    table_h = (0.9 + 0.45 * len(parcels)) if criteria_table else 0.0
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n + table_h), sharex=True, squeeze=False)
    col_titles = ["deviant (z-scored)", "standard (z-scored)", "deviant - standard (MMN, z-scored)"]
    sigs = [z_dev, z_std, z_diff]
    for i, (pname, members, pr) in enumerate(parcels):
        for j, (ct, sig) in enumerate(zip(col_titles, sigs)):
            ax = axes[i][j]
            ax.plot(x, sig[win, i], color="tab:blue", lw=1.8)
            ax.axvspan(100, 240, color="orange", alpha=0.10)   # MMN scoring band
            ax.axvline(0, color="k", ls=":", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5)
            if j == 2:
                ax.set_title((f"{ct}\nbaseline_normalized_peak={peak[i]:+.2f}" if i == 0
                              else f"peak={peak[i]:+.2f}"), fontsize=10 if i == 0 else 9)
            elif i == 0:
                ax.set_title(ct, fontsize=11)
            if i == n - 1:
                ax.set_xlabel("time from final-tone onset (ms)")
        axes[i][0].set_ylabel(f"{pname}\nr={pr:.2f}  ({'+'.join(members)})", fontsize=9)
    level = getattr(args, "level", "parcels")
    fig.suptitle(
        f"In-silico MMN — {method} ({label}, {source})  |  layer {args.layer}, "
        f"{args.highpass_hz} Hz HP, {level} NC r>{args.nc_r_threshold} (raw avg)\n"
        f"classic oddball design: deviant train's final tone differs from the standard's; "
        f"shaded = 100–240 ms MMN band. Columns are full baseline z-scores (mean and std of the "
        f"pre-onset window); 3rd column annotated with its own most-negative point in that band "
        f"(baseline_normalized_peak). Each row its own y-scale.",
        fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if criteria_table:
        bottom_frac = table_h / (2.6 * n + table_h)
        fig.tight_layout(rect=[0, bottom_frac, 1, 0.97])
        row_names = [p[0] for p in parcels]
        _draw_criteria_table(fig, criteria_table, row_names,
                             rect=[0.12, 0.02, 0.76, bottom_frac - 0.04])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# Fronto-central ROI for the automatic electrode-level MMN criterion (Umbricht & Krljes 2005:
# MMN max fronto-central). Shared by insilico_mmn_electrodes.py (mTRF) and insilico_mmn_attn.py
# (encoder, electrode-level checkpoints) -- relocated here so both can import it identically.
FC_ROI = ["Fz", "FCz", "Cz", "FC1", "FC2", "F1", "F2"]


def mmn_metric(res, electrodes, roi):
    """ROI-averaged z-scored baseline_normalized_peak (already computed per-electrode by
    analyze_method/finalize_method) -- the canonical verdict metric, never re-derived here."""
    idx = [i for i, (ch, _, _) in enumerate(electrodes) if ch in roi]
    if not idx:
        return float("nan"), []
    used = [electrodes[i][0] for i in idx]
    return float(np.nanmean(res["peak"][idx])), used


def plot_topo(method, label, source, res, electrodes, args, amp, roi_used, present, out_path):
    """Row B: topographic montage, ONE trace per electrode (deviant - standard, z-scored only --
    no separate deviant/standard panels). Shared by insilico_mmn_electrodes.py (mTRF) and
    insilico_mmn_attn.py (encoder, electrode-level checkpoints)."""
    rel, diff = res["rel_ms"], res["z_diff"]      # z_diff is the full baseline z-score, same units as peak
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
        f"In-silico MMN (electrodes) — {method} ({label}, {source})  |  layer {args.layer}\n"
        f"z-scored deviant - standard per electrode (red = fronto-central ROI); "
        f"shaded = {args.mmn_lo_ms:.0f}-{args.mmn_hi_ms:.0f} ms band\n"
        f"ROI mean baseline_normalized_peak = {amp:+.3g}  ->  {verdict}  (thresh {-args.mmn_thresh:+.3g})",
        fontsize=11, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}   [{verdict}, ROI peak {amp:+.3g}]")


def main():
    p = argparse.ArgumentParser()
    # EEG dataset to FIT on. Default = D2 (Cortical Surprisal, human-speech audiobook EEG with healthy
    # fronto-central channels). --broderick_* kept as back-compat aliases. See project_plan §19/§20.
    p.add_argument("--train_features", "--broderick_features", dest="train_features",
                   default="/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features/"
                           "whisper-small-delta-t-surprisal/merged",
                   help="features dir of the EEG dataset to fit on (default: D2/surprisal, whisper-small)")
    p.add_argument("--train_neural", "--broderick_neural", dest="train_neural",
                   default="outputs/neural_data/surprisal_30s.h5",
                   help="EEG HDF5 to fit + held-out-eval on (default: D2 = Cortical Surprisal)")
    p.add_argument("--mmn_features_root", default="outputs/features")
    p.add_argument("--stimuli_root", default="outputs/mmn_stimuli")
    p.add_argument("--layer", default="blocks.3")
    p.add_argument("--methods", default="all",
                   help="comma-sep stim-dir names, or 'all' for the registry")
    p.add_argument("--nc_r_threshold", type=float, default=0.2,
                   help="drop channels with reliability r <= this before averaging into a parcel")
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--metadata_csv", default=DEFAULT_SOA_CSV,
                   help="per-method standard_soa lookup, for the verdict baseline window")
    p.add_argument("--baseline_start_mult", type=float, default=-3.0,
                   help="pre-onset baseline window start, in units of soa_ms (negative = before onset)")
    p.add_argument("--baseline_end_mult", type=float, default=0.0,
                   help="pre-onset baseline window end, in units of soa_ms")
    p.add_argument("--n_train_time_samples", type=int, default=120)
    p.add_argument("--eval_heldout", type=lambda s: s.lower() not in ("0", "false", "no"),
                   default=True, help="score the mapping on the built-in held-out TEST runs")
    p.add_argument("--n_eval_time_samples", type=int, default=400,
                   help="output time bins sampled per held-out window for the eval correlation")
    p.add_argument("--alpha_log_min", type=float, default=1.0)
    p.add_argument("--alpha_log_max", type=float, default=7.0)
    p.add_argument("--alpha_n", type=int, default=25)
    p.add_argument("--win_pre_ms", type=float, default=150.0)
    p.add_argument("--win_post_ms", type=float, default=500.0)
    p.add_argument("--out_dir", default="outputs/figures/insilico_mmn")
    p.add_argument("--data_dir", default="outputs/insilico_mmn_predictions",
                   help="where to write the parcel-level raw prediction HDF5 (one per layer)")
    args = p.parse_args()

    lags = lags_in_bins(0.0, args.lag_max_ms, TIME_STEP_MS, TIME_STEP_MS)

    print(f"Building parcels (NC floor r>{args.nc_r_threshold}):")
    parcels = build_parcels(Path(args.train_neural), args.nc_r_threshold)
    assert parcels, "no parcels survived the NC threshold"

    # fit the model->EEG mapping ONCE for this layer, apply to every method
    model, mu, sd, eval_metrics = fit_mapping(args, lags, parcels)
    soa_table = load_soa_table(args.metadata_csv)
    duration_map = load_duration_map(DURATION_CSV)

    if args.methods == "all":
        run = METHODS
    else:
        want = [m.strip() for m in args.methods.split(",")]
        reg = {m[0]: m for m in METHODS}
        run = [reg.get(w, (w, w, "")) for w in want]

    out_dir = Path(args.out_dir)
    data_path = Path(args.data_dir) / f"predictions__{args.layer}.h5"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(data_path, "w")
    h5.attrs.update(dict(
        layer=args.layer, highpass_hz=args.highpass_hz, lag_max_ms=args.lag_max_ms,
        fs=FS, time_step_ms=TIME_STEP_MS, nc_r_threshold=args.nc_r_threshold,
        note=("Parcel-level RAW (NOT baseline-corrected) predicted EEG. time_ms=0 is the "
              "final/critical-tone onset (negatives = before onset); classic oddball design: the "
              "deviant train's final tone differs in frequency from the standard's. Parcels = raw "
              "average over channels passing NC r>threshold. Compute MMN as deviant_mean - standard; "
              "see each method group's 'peak' attr for the z-scored baseline_normalized_peak.")))
    h5.create_dataset("parcels", data=np.array([p[0] for p in parcels], dtype="S12"))
    h5.create_dataset("parcel_members", data=np.array(["+".join(p[1]) for p in parcels], dtype="S40"))
    h5.create_dataset("parcel_nc_r", data=np.array([p[2] for p in parcels], np.float32))
    if eval_metrics is not None:
        # out-of-sample mapping quality on the built-in held-out TEST runs (separate audiobook
        # runs, never in train); same parcel order. heldout_r = raw Pearson r, heldout_r_nc = NC-normalized.
        h5.create_dataset("heldout_r", data=eval_metrics["r"])
        h5.create_dataset("heldout_r_nc", data=eval_metrics["r_nc"])
        h5.attrs["heldout_test_windows"] = eval_metrics["n_test_windows"]
        h5.attrs["heldout_n_samples"] = eval_metrics["n_samples"]

    for method, label, source in run:
        feat_dir = Path(args.mmn_features_root) / f"mmn-{method}-delta-t"
        stim_dir = Path(args.stimuli_root) / method
        if not feat_dir.exists():
            print(f"  {method}: feature dir {feat_dir} missing -> skipped")
            continue
        soa_ms = soa_for_method(method, soa_table)
        res = analyze_method(method, feat_dir, stim_dir, model, mu, sd, lags, parcels, args, soa_ms)
        if res is None:
            continue
        # Row C criteria table: C0-S6 per individual parcel (frontal/central/temporal), not the
        # ROI-mean -- reuses mmn_criteria_table.compute_criteria_table (Task 3).
        row_idx = [i for i, p in enumerate(parcels) if p[0] in PLOT_ROWS]
        criteria_table = compute_criteria_table(
            res["rel_ms"], res["z_diff"][:, row_idx], [parcels[i][0] for i in row_idx],
            method, duration_map)

        out_path = out_dir / f"insilico_mmn__{method}__{args.layer}.png"
        plot_method(method, label, source, res, parcels, args, out_path,
                   criteria_table=criteria_table)

        g = h5.create_group(method)
        g.attrs.update(dict(context_final=label, source=source, soa_ms=soa_ms,
                            final_tone_onset_s=res["final_s"], n_deviants=len(res["dev_ids"])))
        g.create_dataset("time_ms", data=res["rel_ms"])
        g.create_dataset("standard", data=res["std_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_mean", data=res["dev_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviants", data=res["dev_stack"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_ids", data=np.array(res["dev_ids"], dtype="S40"))
        g.create_dataset("peak", data=res["peak"])
        g.create_dataset("n7v1_peak", data=res["n7v1_peak"])
    h5.close()
    print(f"Wrote parcel predictions -> {data_path}")


if __name__ == "__main__":
    main()
