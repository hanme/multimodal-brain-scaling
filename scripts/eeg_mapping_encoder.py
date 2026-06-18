"""Collect a trained attention-encoder layer sweep into the SAME JSON schema as
`eeg_mapping_sweep.py`, so `plot_eeg_mapping.py` produces the layer-selection + fit-quality
figures for the encoder exactly as it does for the mTRF (apples-to-apples, one plotter).

The encoder driver (`evaluate_features_attn_probe_temporal.py`) writes per-layer held-out TEST r
into `attn_probe_temporal_scores.h5` but NO per-layer validation score and does no layer
selection. Here we supply the missing selection signal **honestly**: for each layer we reload its
checkpoint (`model__<layer>.pt`) and score the **validation split carved from TRAIN** — the same
split the driver used for MIRAGE-style checkpoint selection, reproduced from the checkpoint's
seed/val_frac. The test split is never touched during selection, mirroring the mTRF CV-on-train
rule. (`cv_score_by_layer` here is a single train-internal validation split, not k-fold.)

One JSON per (model x level), schema-compatible with the mTRF JSONs:
  python scripts/eeg_mapping_encoder.py \
    --model_id whisper-small --target_level parcels \
    --results_dir outputs/results/whisper-small-probe-group-d2-parcels \
    --features_dir <whisper-small D2 features>/merged \
    --out outputs/results/eeg_mapping_encoder/whisper-small__parcels__D2.json
Then:
  python scripts/plot_eeg_mapping.py --results_dir outputs/results/eeg_mapping_encoder --target_level parcels
"""

import json
import argparse
from pathlib import Path

import numpy as np
import h5py

from mbs.evaluation.utils.evaluation_helpers import load_layer_features
from mbs.evaluation.attn_probe.checkpoint import load_probe_checkpoint
from mbs.evaluation.attn_probe.engine_temporal import score_heldout
from mbs.evaluation.evaluate_features_attn_probe_temporal import _aligned_feats


def layers_for(model_id, layers_config):
    cfg = layers_config or f"configs/extraction/audio/{model_id.replace('-', '_')}_layers.json"
    spec = json.load(open(cfg))
    names = [e["name"] if isinstance(e, dict) else e for e in spec]
    pos = [float(e["position"]) if isinstance(e, dict) else i / max(len(spec) - 1, 1)
           for i, e in enumerate(spec)]
    return names, pos


def val_indices(n_tr, seed, val_frac):
    """Reproduce the driver's train/val carve EXACTLY (engine selection split)."""
    perm = np.random.default_rng(seed).permutation(n_tr)
    n_val = int(round(val_frac * n_tr)) if n_tr > 1 else 0
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    if tr_idx.size == 0:
        tr_idx, val_idx = np.arange(n_tr), np.array([], dtype=int)
    return val_idx


def parcels_from_ckpt(ckpt):
    """Rebuild the (name, members, nc) list stored in the checkpoint."""
    p = ckpt["parcels"]
    names = list(p["names"])
    members = [m.split("+") for m in p["members"]]
    nc = [float(x) for x in p["nc_r"]]
    return list(zip(names, members, nc))


def val_r_for_layer(layer, ckpt_path, args):
    """Mean validation Pearson r over targets for one layer's checkpoint (selection signal)."""
    model, ckpt = load_probe_checkpoint(ckpt_path, device=args.device)
    parcels = parcels_from_ckpt(ckpt)
    layer_feats, id_map = load_layer_features(layer, features_folder=Path(args.features_dir))
    layer_feats = layer_feats.astype(np.float32)
    feats_tr, eeg_tr = _aligned_feats(args.neural, "group", parcels, "train",
                                      layer_feats, id_map, ckpt["highpass_hz"])
    feats_tr = (feats_tr - ckpt["mu"]) / ckpt["sd"]                 # checkpoint's TRAIN-portion stats
    vidx = val_indices(feats_tr.shape[0], args.seed, args.val_frac)
    if vidx.size == 0:
        return float("nan")
    # Pearson is scale-invariant, so the raw (high-passed) val EEG is fine as the target.
    r = score_heldout(model, feats_tr[vidx], eeg_tr[vidx], ckpt["lookback"], "group", device=args.device)
    return float(np.nanmean(r))


def main():
    p = argparse.ArgumentParser(description="Encoder layer sweep -> eeg_mapping-schema JSON.")
    p.add_argument("--model_id", required=True)
    p.add_argument("--target_level", choices=["parcels", "electrodes"], required=True)
    p.add_argument("--results_dir", required=True,
                   help="<model>-probe-group-d2-<level>/ (has attn_probe_temporal_scores.h5 + model__*.pt)")
    p.add_argument("--features_dir", required=True, help="D2 mapping features for this model")
    p.add_argument("--neural", default="outputs/neural_data/surprisal_30s.h5")
    p.add_argument("--layers_config", default="")
    p.add_argument("--test_split", default="test", help="held-out split key in the scores h5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    scores_h5 = results_dir / "attn_probe_temporal_scores.h5"
    assert scores_h5.exists(), f"no scores h5 in {results_dir}"
    all_layers, all_pos = layers_for(args.model_id, args.layers_config)

    layers, positions, cv_by, test_by, names = [], [], [], [], None
    with h5py.File(scores_h5, "r") as h:
        for layer, pos in zip(all_layers, all_pos):
            key = layer.replace(".", "-")
            ckpt = results_dir / f"model__{layer}.pt"
            if key not in h or not ckpt.exists():
                continue                                            # layer not finished yet -> skip
            g = h[key]
            tr = g[f"heldout_r__{args.test_split}"][:].astype(float)
            if names is None:
                names = [s.decode() if hasattr(s, "decode") else str(s) for s in g["parcels"][:]]
            vr = val_r_for_layer(layer, ckpt, args)
            layers.append(layer); positions.append(pos)
            test_by.append(tr.tolist()); cv_by.append(vr)
            print(f"  {layer:10s} VAL r={vr:+.3f}  TEST r={np.nanmean(tr):+.3f}")

    assert layers, f"no finished layers (h5 group + model__<layer>.pt) in {results_dir}"
    chosen_idx = int(np.nanargmax(cv_by))
    chosen = layers[chosen_idx]
    print(f"\nCHOSEN layer (max VAL) = {chosen}  "
          f"VAL r={cv_by[chosen_idx]:+.3f}  TEST r={np.nanmean(test_by[chosen_idx]):+.3f}")

    out = dict(
        model_id=args.model_id, target_level=args.target_level, neural=args.neural,
        features_dir=args.features_dir, method="attn_encoder",
        seed=args.seed, val_frac=args.val_frac, selection="train-carved validation split",
        layers=layers, positions=positions, targets=names,
        cv_r_by_layer=[[c] for c in cv_by],          # parity w/ mTRF (per-target list); single val r
        test_r_by_layer=test_by, cv_score_by_layer=cv_by,
        chosen_idx=chosen_idx, chosen_layer=chosen,
        test_r_chosen=test_by[chosen_idx], cv_score_chosen=cv_by[chosen_idx])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
