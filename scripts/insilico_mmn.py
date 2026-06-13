"""In-silico MMN: predict parcel-level EEG for MMN tone stimuli via the mTRF mapping.

Pipeline:
  1. Fit a FIR mTRF on Broderick (EEG[t] = sum_lag W[lag] . feature[t-lag]) for ONE model
     layer, with electrodes aggregated into coarse 10-20 parcels (Kadir-style clusters).
     Each parcel = raw average over its constituent channels that pass an NC floor
     (Gokce/Kadir precedent: drop channels with reliability r <= --nc_r_threshold before
     fitting; see evaluate_features_committed_layers.py `noise_ceiling > 0.1`).
  2. Apply that single mapping to whisper-base delta_T features of EACH MMN method
     (identity-MMN design: the final/critical tone is physically identical in standard and
     deviant; the deviance lives in the preceding context frequency).
  3. Per method: average deviant trials, take the standard, time-lock to the final-tone
     onset, baseline-correct, compute deviant - standard (= MMN).
  4. Plot a grid: rows = parcels (each its OWN y-scale, annotated with NC + member channels),
     columns = deviant / standard / (deviant - standard).

The mapping depends only on (layer, parcels), NOT on the MMN method, so it is fit ONCE
per layer and applied to all methods in the loop.

Run AFTER the MMN delta_T features exist (scripts/slurm_mmn_extract.sh) for every method.
"""

from pathlib import Path
import argparse
import glob
import numpy as np
import h5py
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

from mbs.evaluation.utils.evaluation_helpers import (
    load_layer_features, load_neural_data,
)
from mbs.evaluation.evaluate_features_mtrf import (
    lags_in_bins, build_lagged_design, highpass_along_time, sample_time_indices,
    pearson_along_time,
)

FS = 50.0          # EEG / feature grid (Hz)
TIME_STEP_MS = 20.0

# Coarse 10-20 parcels (must match src/mbs/data_prep/format_eeg_hdf5.py cluster definitions).
CLUSTERS = {
    "frontal":   ["Fz", "F3", "F4", "FCz"],
    "central":   ["Cz", "C3", "C4"],
    "temporal":  ["T7", "T8"],
    "parietal":  ["Pz", "P3", "P4", "P7", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}

# MMN methods: (stimulus-dir name, "context->final" label, deviance direction). All are
# identity-MMN designs (final tone identical in std & dev). m12/44/55 appear in both
# directions (regular = downward deviant, _counter = upward), giving matched mirror pairs.
METHODS = [
    ("method_37",         "1050→1000 Hz", "DOWN ~4%"),
    ("method_12",         "1200→1000 Hz", "DOWN ~17%"),
    ("method_44",         "1000→633 Hz",  "DOWN ~37%"),
    ("method_09",         "1000→600 Hz",  "DOWN ~40%"),
    ("method_55",         "2000→1000 Hz", "DOWN ~50%"),
    ("method_12_counter", "1000→1200 Hz", "UP ~20%"),
    ("method_44_counter", "633→1000 Hz",  "UP ~58%"),
    ("method_55_counter", "1000→2000 Hz", "UP octave"),
]


def decode(xs):
    return [x.decode() if hasattr(x, "decode") else x for x in xs]


def channel_r(neural_h5, ch):
    """Reliability (correlation scale) of one channel: sqrt(mean stored var% / 100)."""
    try:
        with h5py.File(neural_h5, "r") as h:
            v = float(np.nanmean(h["noise_ceilings"]["group"][ch][:]))
        return float(np.sqrt(max(v, 0.0) / 100.0))
    except Exception:
        return float("nan")


def build_parcels(neural_h5, threshold):
    """Return ordered list of (parcel, members_kept, parcel_r) for parcels with >=1 survivor.

    parcel_r = mean reliability of the kept members (correlation scale).
    """
    out = []
    for name, members in CLUSTERS.items():
        rs = {c: channel_r(neural_h5, c) for c in members}
        kept = [c for c in members if rs[c] > threshold]
        if not kept:
            print(f"  parcel '{name}': no channel passes r>{threshold} "
                  f"({', '.join(f'{c}={rs[c]:.2f}' for c in members)}) -> DROPPED")
            continue
        pr = float(np.mean([rs[c] for c in kept]))
        out.append((name, kept, pr))
        print(f"  parcel '{name}': keep {kept} (r={pr:.2f}); "
              f"drop {[c for c in members if c not in kept]}")
    return out


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


def load_split_parcels(neural_h5, feats_all, id_map, parcels, split):
    """Build (parcel-target EEG, aligned raw features) for one Broderick split.

    Returns (eeg [n, T, n_parcel], feats [n, T, d]) restricted to stimuli whose ids exist
    in id_map. Same id alignment for every channel; parcels are raw averages over members.
    """
    fi = keep = None
    parcel_cols = []
    for _, members, _ in parcels:
        chan_stack = []
        for ch in members:
            ids, eeg_ch, _ = load_neural_data(Path(neural_h5), "group", ch, split)
            if fi is None:
                ids = decode(ids)
                raw = [id_map.get(s) for s in ids]
                keep = [i for i, v in enumerate(raw) if v is not None]
                fi = [v for v in raw if v is not None]
            chan_stack.append(eeg_ch[keep][:, :, 0])             # [n, T]
        parcel_cols.append(np.mean(chan_stack, axis=0)[:, :, None])  # raw avg -> [n, T, 1]
    eeg = np.concatenate(parcel_cols, axis=2).astype(np.float32)     # [n, T, n_parcel]
    return eeg, feats_all[fi]


def fit_mapping(args, lags, parcels):
    """Fit FIR mTRF on Broderick TRAIN for one layer -> (model, feat_mu, feat_sd, eval_metrics).

    Target columns are the parcels (raw average over NC-surviving member channels). If
    --eval_heldout, also predicts the built-in held-out TEST split (separate audiobook runs)
    and returns per-parcel out-of-sample Pearson r (raw and NC-normalized); else eval=None.
    """
    feats_all, id_map = load_layer_features(args.layer, features_folder=Path(args.broderick_features))
    feats_all = feats_all.astype(np.float32)  # [n_stim, T, d]

    eeg, feats = load_split_parcels(args.broderick_neural, feats_all, id_map, parcels, "train")

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
    eeg, feats = load_split_parcels(args.broderick_neural, feats_all, id_map, parcels, "test")
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


def analyze_method(method, feat_dir, stim_dir, model, mu, sd, lags, parcels, args):
    """Predict + time-lock one method. Returns a dict with both the baseline-corrected
    arrays (for plotting) and the RAW full time courses (for downstream MMN metrics), or None.

    All arrays span the full valid window (the entire ~29 s pre-final-tone baseline + post-onset),
    columns = parcels (same order as `parcels`). rel_ms = 0 is the final/critical-tone onset.
    """
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
    dev_stack = np.stack(dev_preds, 0)              # [n_dev, n_t, n_parcel], RAW
    dev_raw = dev_stack.mean(0)                     # [n_t, n_parcel], RAW

    std_wav = sorted(glob.glob(f"{stim_dir}/*standard*.wav"))[0]
    final_s = detect_final_tone_onset_s(std_wav)
    rel_ms = t_idx * TIME_STEP_MS - final_s * 1000.0          # 0 = final-tone onset
    base = (rel_ms >= -args.win_pre_ms) & (rel_ms < 0)        # pre-onset baseline

    def bc(sig):
        return sig - sig[base].mean(0, keepdims=True)
    dev_b, std_b = bc(dev_raw), bc(std_raw)
    print(f"  {method}: {len(dev_preds)} deviants avg; final tone ~{final_s:.2f}s")
    return dict(rel_ms=rel_ms.astype(np.float32), dev_b=dev_b, std_b=std_b, diff_b=(dev_b - std_b),
                std_raw=std_raw.astype(np.float32), dev_raw=dev_raw.astype(np.float32),
                dev_stack=dev_stack.astype(np.float32), dev_ids=dev_ids, final_s=final_s)


def plot_method(method, label, direction, res, parcels, args, out_path):
    rel_ms, dev_b, std_b, diff_b = res["rel_ms"], res["dev_b"], res["std_b"], res["diff_b"]
    win = (rel_ms >= -args.win_pre_ms) & (rel_ms <= args.win_post_ms)
    x = rel_ms[win]
    n = len(parcels)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n), sharex=True, squeeze=False)
    col_titles = ["deviant", "standard", "deviant - standard (MMN)"]
    sigs = [dev_b, std_b, diff_b]
    for i, (pname, members, pr) in enumerate(parcels):
        for j, (ct, sig) in enumerate(zip(col_titles, sigs)):
            ax = axes[i][j]
            ax.plot(x, sig[win, i], color="tab:blue", lw=1.8)
            ax.axvspan(100, 250, color="orange", alpha=0.10)   # typical MMN latency band
            ax.axvline(0, color="k", ls=":", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5)
            if i == 0:
                ax.set_title(ct, fontsize=11)
            if i == n - 1:
                ax.set_xlabel("time from final-tone onset (ms)")
        axes[i][0].set_ylabel(f"{pname}\nr={pr:.2f}  ({'+'.join(members)})", fontsize=9)
    fig.suptitle(
        f"In-silico MMN — {method} ({label}, {direction})  |  layer {args.layer}, "
        f"{args.highpass_hz} Hz HP, parcels NC r>{args.nc_r_threshold} (raw avg)\n"
        f"identity design: final tone physically identical in std & dev; "
        f"shaded = 100–250 ms MMN band. Each row has its own y-scale.",
        fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--broderick_features", default="outputs/features/whisper-base-delta-t/merged/")
    p.add_argument("--broderick_neural", default="outputs/neural_data/broderick2018_30s.h5")
    p.add_argument("--mmn_features_root", default="outputs/features")
    p.add_argument("--stimuli_root", default="outputs/mmn_stimuli")
    p.add_argument("--layer", default="blocks.3")
    p.add_argument("--methods", default="all",
                   help="comma-sep stim-dir names, or 'all' for the registry")
    p.add_argument("--nc_r_threshold", type=float, default=0.2,
                   help="drop channels with reliability r <= this before averaging into a parcel")
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=500.0)
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
    parcels = build_parcels(Path(args.broderick_neural), args.nc_r_threshold)
    assert parcels, "no parcels survived the NC threshold"

    # fit the Broderick->EEG mapping ONCE for this layer, apply to every method
    model, mu, sd, eval_metrics = fit_mapping(args, lags, parcels)

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
              "final/critical-tone onset (negatives = before onset); identity-MMN design: the "
              "final tone is physically identical in standard and deviant. Parcels = raw average "
              "over channels passing NC r>threshold. Compute MMN as deviant_mean - standard.")))
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

    for method, label, direction in run:
        feat_dir = Path(args.mmn_features_root) / f"mmn-{method}-delta-t"
        stim_dir = Path(args.stimuli_root) / method
        if not feat_dir.exists():
            print(f"  {method}: feature dir {feat_dir} missing -> skipped")
            continue
        res = analyze_method(method, feat_dir, stim_dir, model, mu, sd, lags, parcels, args)
        if res is None:
            continue
        out_path = out_dir / f"insilico_mmn__{method}__{args.layer}.png"
        plot_method(method, label, direction, res, parcels, args, out_path)

        g = h5.create_group(method)
        g.attrs.update(dict(context_final=label, direction=direction,
                            final_tone_onset_s=res["final_s"], n_deviants=len(res["dev_ids"])))
        g.create_dataset("time_ms", data=res["rel_ms"])
        g.create_dataset("standard", data=res["std_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_mean", data=res["dev_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviants", data=res["dev_stack"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_ids", data=np.array(res["dev_ids"], dtype="S40"))
    h5.close()
    print(f"Wrote parcel predictions -> {data_path}")


if __name__ == "__main__":
    main()
