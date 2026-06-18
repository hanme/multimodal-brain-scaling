"""Workstream-B counterpart of insilico_mmn.py: drive the in-silico MMN + fit-quality figures
with the TRAINED attention encoder instead of the mTRF.

Where insilico_mmn.py re-fits a ridge on the spot, here we LOAD a checkpoint saved by
``evaluate_features_attn_probe_temporal.py --save_model`` (the reusable blocks.X -> parcel-EEG
network) and apply it. The time-locking, baseline-correction and plotting are reused verbatim
from insilico_mmn.py (a swapped predictor is the only difference), so the A and B figures are
visually identical in layout and directly comparable.

Produces, for one method + the checkpoint's layer:
  <out_dir>/insilico_mmn__<method>__<layer>__attn.png     (deviant / standard / dev-std, per parcel)
  <out_dir>/fit_quality__<tag>__<layer>__attn.png         (recorded vs predicted, held-out window)
  <data_dir>/predictions__<layer>__attn.h5                (raw arrays for downstream MMN metrics)

Example (whisper-small, D2, blocks.10 checkpoint):
  python scripts/insilico_mmn_attn.py \
    --checkpoint outputs/results/whisper-small-probe-group-d2-mmn/model__blocks.10.pt \
    --mmn_features_root outputs/features/whisper-small-mmn --method method_09 \
    --features_dir /work/.../whisper-small-delta-t-surprisal/merged \
    --neural outputs/neural_data/surprisal_30s.h5 \
    --out_dir outputs/figures/insilico_mmn_small --data_dir outputs/insilico_mmn_predictions_small
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling insilico_mmn.py
from insilico_mmn import (  # noqa: E402
    FS, TIME_STEP_MS, METHODS, load_split_parcels, detect_final_tone_onset_s, plot_method,
)
from mbs.evaluation.attn_probe.checkpoint import load_probe_checkpoint, predict_timecourse  # noqa: E402
from mbs.evaluation.utils.evaluation_helpers import load_layer_features  # noqa: E402
from mbs.evaluation.evaluate_features_mtrf import highpass_along_time, pearson_along_time  # noqa: E402


def parcels_from_ckpt(ck):
    """Reconstruct insilico-style [(name, [members], nc_r)] from a saved checkpoint."""
    names = ck["parcels"]["names"]
    members = [m.split("+") for m in ck["parcels"]["members"]]
    nc = list(ck["parcels"]["nc_r"])
    return [(n, mem, float(r)) for n, mem, r in zip(names, members, nc)]


def analyze_method_attn(method, feat_dir, stim_dir, model, ck, layer, highpass_hz,
                        win_pre_ms, device):
    """B-side of insilico_mmn.analyze_method: predict every MMN clip with the network, time-lock
    to the final-tone onset, baseline-correct. Returns the same dict shape plot_method expects."""
    mfeats, mid_map = load_layer_features(layer, features_folder=Path(feat_dir))
    mfeats = mfeats.astype(np.float32)
    id_by_row = {v: k for k, v in mid_map.items()}

    std_raw, dev_preds, dev_ids, t_idx = None, [], [], None
    for row in range(mfeats.shape[0]):
        sid = str(id_by_row[row])
        t_idx, pred = predict_timecourse(model, ck, mfeats[row], "group", device)
        if "standard" in sid.lower():
            std_raw = pred
        elif "deviant" in sid.lower():
            dev_preds.append(pred); dev_ids.append(sid)
    if std_raw is None or not dev_preds:
        print(f"  {method}: missing standard or deviants -> skipped")
        return None

    dev_stack = np.stack(dev_preds, 0)              # [n_dev, n_t, P] RAW
    dev_raw = dev_stack.mean(0)                     # [n_t, P] RAW
    std_wav = sorted(glob.glob(f"{stim_dir}/*standard*.wav"))[0]
    final_s = detect_final_tone_onset_s(std_wav)
    rel_ms = t_idx * TIME_STEP_MS - final_s * 1000.0
    base = (rel_ms >= -win_pre_ms) & (rel_ms < 0)

    def bc(sig):
        return sig - sig[base].mean(0, keepdims=True)
    dev_b, std_b = bc(dev_raw), bc(std_raw)
    print(f"  {method}: {len(dev_preds)} deviants avg; final tone ~{final_s:.2f}s")
    return dict(rel_ms=rel_ms.astype(np.float32), dev_b=dev_b, std_b=std_b, diff_b=(dev_b - std_b),
                std_raw=std_raw.astype(np.float32), dev_raw=dev_raw.astype(np.float32),
                dev_stack=dev_stack.astype(np.float32), dev_ids=dev_ids, final_s=final_s)


def fit_quality_figure(model, ck, parcels, neural, features_dir, layer, highpass_hz,
                       device, out_path, window_idx=-1):
    """Recorded vs network-predicted parcel EEG on a held-out TEST window (B-side of
    plot_fit_quality.py)."""
    feats_all, id_map = load_layer_features(layer, features_folder=Path(features_dir))
    eeg, feats = load_split_parcels(neural, feats_all.astype(np.float32), id_map, parcels, "test")
    n_win = eeg.shape[0]
    if n_win == 0:
        print("  fit-quality: no test windows -> skipped"); return None

    hp_eeg = highpass_along_time(eeg, FS, highpass_hz)
    preds, t_idx = [], None
    for w in range(n_win):
        t_idx, pred = predict_timecourse(model, ck, feats[w], "group", device)
        preds.append(pred)
    preds = np.stack(preds, 0)
    actual = hp_eeg[:, t_idx, :]

    if window_idx >= 0:
        w = window_idx
    else:
        per_win = np.array([np.nanmean(pearson_along_time(actual[i], preds[i])) for i in range(n_win)])
        w = int(np.argsort(per_win)[len(per_win) // 2])
    # per-parcel held-out r over ALL windows (for panel titles)
    r_all = np.nanmean(np.stack([pearson_along_time(actual[i], preds[i]) for i in range(n_win)], 0), 0)
    print(f"  fit-quality: window {w}/{n_win}, mean-parcel r this window = "
          f"{np.nanmean(pearson_along_time(actual[w], preds[w])):+.3f}")

    t_s = t_idx * TIME_STEP_MS / 1000.0
    n = len(parcels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True, squeeze=False)
    for i, (pname, members, pr) in enumerate(parcels):
        ax = axes[i][0]
        ax.plot(t_s, actual[w][:, i], color="k", lw=1.4, label="recorded EEG")
        ax.plot(t_s, preds[w][:, i], color="tab:green", lw=1.4, alpha=0.85, label="predicted (attn encoder)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(f"{pname}\nNC r={pr:.2f}  ({'+'.join(members)})", fontsize=9)
        ax.set_title(f"{pname}: held-out r = {r_all[i]:+.3f}  (over all {n_win} test windows)", fontsize=9)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("time within held-out test window (s)")
    fig.suptitle(
        f"Fit quality — attention encoder: recorded vs predicted parcel EEG, held-out window {w}\n"
        f"layer {layer}, {highpass_hz} Hz HP, lookback {ck['lookback']} bins  |  {Path(neural).name}",
        fontsize=10)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return r_all


def main():
    p = argparse.ArgumentParser(description="In-silico MMN + fit-quality figures from a trained attention encoder.")
    p.add_argument("--checkpoint", required=True, help="model__<layer>.pt from --save_model")
    p.add_argument("--mmn_features_root", required=True, help="dir containing mmn-<method>-delta-t")
    p.add_argument("--method", default="method_09")
    p.add_argument("--stimuli_root", default="outputs/mmn_stimuli")
    p.add_argument("--features_dir", required=True, help="mapping (train/test) features for fit-quality")
    p.add_argument("--neural", required=True, help="neural HDF5 with the test split (fit-quality)")
    p.add_argument("--win_pre_ms", type=float, default=150.0)
    p.add_argument("--win_post_ms", type=float, default=500.0)
    p.add_argument("--window_idx", type=int, default=-1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="outputs/figures/insilico_mmn_small")
    p.add_argument("--data_dir", default="outputs/insilico_mmn_predictions_small")
    p.add_argument("--tag", default="attn", help="filename tag for the fit-quality figure")
    args = p.parse_args()

    model, ck = load_probe_checkpoint(args.checkpoint, device=args.device)
    layer = ck["layer"]
    parcels = parcels_from_ckpt(ck)
    print(f"Loaded checkpoint: layer={layer}  parcels={[n for n,_,_ in parcels]}  "
          f"lookback={ck['lookback']}  hp={ck['highpass_hz']}Hz")

    label, direction = "attn-encoder", "identity"
    reg = {m[0]: m for m in METHODS}
    if args.method in reg:
        _, label, direction = reg[args.method]

    out_dir = Path(args.out_dir)
    feat_dir = Path(args.mmn_features_root) / f"mmn-{args.method}-delta-t"
    stim_dir = Path(args.stimuli_root) / args.method
    assert feat_dir.exists(), f"MMN feature dir missing: {feat_dir}"

    # ---- fit-quality (held-out speech) ----
    print("Fit-quality figure:")
    fq_path = out_dir / f"fit_quality__{args.tag}__{layer}__attn.png"
    fit_quality_figure(model, ck, parcels, args.neural, args.features_dir, layer,
                       ck["highpass_hz"], args.device, fq_path, args.window_idx)

    # ---- in-silico MMN ----
    print("In-silico MMN figure:")
    res = analyze_method_attn(args.method, feat_dir, stim_dir, model, ck, layer,
                              ck["highpass_hz"], args.win_pre_ms, args.device)
    if res is None:
        return
    plot_args = SimpleNamespace(win_pre_ms=args.win_pre_ms, win_post_ms=args.win_post_ms,
                                layer=layer, highpass_hz=ck["highpass_hz"], nc_r_threshold=0.2)
    mmn_path = out_dir / f"insilico_mmn__{args.method}__{layer}__attn.png"
    plot_method(args.method, label, direction, res, parcels, plot_args, mmn_path)

    # ---- raw arrays for downstream MMN metrics ----
    data_path = Path(args.data_dir) / f"predictions__{layer}__attn.h5"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "w") as h5:
        h5.attrs.update(dict(layer=layer, highpass_hz=ck["highpass_hz"], fs=FS,
                             time_step_ms=TIME_STEP_MS, method="attn_encoder",
                             note=("Parcel-level RAW predicted EEG from the trained attention "
                                   "encoder. time_ms=0 = final-tone onset; MMN = deviant - standard.")))
        h5.create_dataset("parcels", data=np.array([n for n, _, _ in parcels], dtype="S12"))
        h5.create_dataset("parcel_members", data=np.array(["+".join(m) for _, m, _ in parcels], dtype="S40"))
        h5.create_dataset("parcel_nc_r", data=np.array([r for _, _, r in parcels], np.float32))
        g = h5.create_group(args.method)
        g.attrs.update(dict(context_final=label, direction=direction,
                            final_tone_onset_s=res["final_s"], n_deviants=len(res["dev_ids"])))
        g.create_dataset("time_ms", data=res["rel_ms"])
        g.create_dataset("standard", data=res["std_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_mean", data=res["dev_raw"], compression="gzip", compression_opts=4)
        g.create_dataset("deviants", data=res["dev_stack"], compression="gzip", compression_opts=4)
        g.create_dataset("deviant_ids", data=np.array(res["dev_ids"], dtype="S40"))
    print(f"Wrote parcel predictions -> {data_path}")


if __name__ == "__main__":
    main()
