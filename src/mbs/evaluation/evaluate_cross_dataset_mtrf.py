"""Cross-dataset (out-of-domain) mTRF transfer — train on one dataset, test on another.

Complements the D3 pooling experiment (plan §13): instead of pooling, this measures pure
transfer. Fits the parcel-mTRF on the SOURCE train split, then scores held-out r BOTH on the
source's own test (in-domain reference) AND on the TARGET dataset's test (out-of-domain), from
the same fit. Source feature standardisation is applied to the target (no peeking).

  python -m mbs.evaluation.evaluate_cross_dataset_mtrf \
    --source_tag d1 --source_data_hdf5 outputs/neural_data/broderick2018_30s.h5 \
    --source_features_dir outputs/features/whisper-base-delta-t/merged/ \
    --target_tag d2 --target_data_hdf5 outputs/neural_data/surprisal_30s.h5 \
    --target_features_dir outputs/features/whisper-base-delta-t-surprisal/merged/ \
    --output_dir outputs/results/whisper-base-mtrf-xfer-d1-to-d2/

Output: cross_mtrf_scores.h5
  <layer>/parcels, parcel_nc_r,
          heldout_r__<source_tag>_<split>   (in-domain),
          heldout_r__<target_tag>_<split>   (transfer)
"""

from pathlib import Path
import argparse
import json

import h5py
import numpy as np

from mbs.core import str2bool
from mbs.evaluation.utils.evaluation_helpers import load_layer_features, ALPHA_LIST_SHORT, ALPHA_LIST
from mbs.evaluation.evaluate_features_mtrf import lags_in_bins, pearson_along_time
from mbs.evaluation.evaluate_features_mtrf_parcels import (
    fit_parcel_mtrf, score_parcel_mtrf, cross_score_dataset, _aligned, FS, TIME_STEP_MS,
)
from mbs.evaluation.attn_probe.dataset_temporal import (
    build_parcels, recompute_parcel_nc, list_test_splits, parcel_nc_vector,
)


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset mTRF transfer (train source, test target).")
    p.add_argument("--model_id", type=str, default="whisper-base")
    p.add_argument("--target_feature_layers", type=str, required=True)
    p.add_argument("--source_tag", type=str, required=True)
    p.add_argument("--source_data_hdf5", type=str, required=True)
    p.add_argument("--source_features_dir", type=str, required=True)
    p.add_argument("--target_tag", type=str, required=True)
    p.add_argument("--target_data_hdf5", type=str, required=True)
    p.add_argument("--target_features_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--parcels_from", type=str, default="outputs/neural_data/broderick2018_30s.h5")
    p.add_argument("--lag_min_ms", type=float, default=0.0)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--lag_step_ms", type=float, default=20.0)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--nc_threshold", type=float, default=0.2)
    p.add_argument("--n_train_time_samples", type=int, default=200)
    p.add_argument("--pca_var", type=float, default=None,
                   help="if set (e.g. 0.95), PCA features to this variance fraction before lagging "
                        "(see evaluate_features_mtrf_parcels). Source PCA is re-applied to target.")
    p.add_argument("--use_wide_range_alphas", type=str2bool, default=False)
    p.add_argument("--layer_id", type=int, default=None)
    p.add_argument("--overwrite", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    alphas = ALPHA_LIST if args.use_wide_range_alphas else ALPHA_LIST_SHORT
    with open(args.target_feature_layers) as f:
        layer_list = [e["name"] if isinstance(e, dict) else e for e in json.load(f)]
    lags = lags_in_bins(args.lag_min_ms, args.lag_max_ms, args.lag_step_ms, TIME_STEP_MS)

    parcels = recompute_parcel_nc(build_parcels(args.parcels_from, args.nc_threshold),
                                  args.source_data_hdf5)
    names = [p[0] for p in parcels]
    nc_r = parcel_nc_vector(parcels)
    src_splits = list_test_splits(args.source_data_hdf5)
    print(f"SOURCE={args.source_tag} (in-domain splits {src_splits}) -> "
          f"TARGET={args.target_tag} (transfer); parcels {names}")

    scores_path = out_dir / "cross_mtrf_scores.h5"
    summary = {"source": args.source_tag, "target": args.target_tag, "config": vars(args), "entries": []}
    file_mode = "w" if args.overwrite else "a"
    with h5py.File(scores_path, file_mode) as out_h5:
        out_h5.attrs["source"] = args.source_tag
        out_h5.attrs["target"] = args.target_tag
        for layer_idx, layer_name in enumerate(layer_list):
            if args.layer_id is not None and layer_idx != args.layer_id:
                continue
            key = layer_name.replace(".", "-")
            if key in out_h5 and not args.overwrite:
                print(f"Layer {layer_name} present — skip."); continue

            src_feats, src_idmap = load_layer_features(layer_name, features_folder=Path(args.source_features_dir))
            if src_feats.ndim != 3:
                print(f"Layer {layer_name}: {src_feats.ndim}D features — skip."); continue
            src_feats = src_feats.astype(np.float32)
            feats_tr, eeg_tr = _aligned(args.source_data_hdf5, parcels, "train", src_feats, src_idmap)
            fit = fit_parcel_mtrf(feats_tr, eeg_tr, lags, highpass_hz=args.highpass_hz,
                                  n_train_time_samples=args.n_train_time_samples, fs=FS, rng=rng,
                                  alphas=alphas, pca_var=args.pca_var)
            if fit["pca"] is not None:
                print(f"  [{layer_name}] PCA: {fit['pca'].n_components_} comps "
                      f"({fit['pca'].explained_variance_ratio_.sum():.3f} var) from d={src_feats.shape[-1]}")

            if key in out_h5:
                del out_h5[key]
            g = out_h5.create_group(key)
            g.create_dataset("parcels", data=np.array(names, dtype="S"))
            g.create_dataset("parcel_nc_r", data=nc_r)
            entry = {"layer": layer_name, "in_domain": {}, "transfer": {}}

            # in-domain (source -> source test)
            for split in src_splits:
                feats_te, eeg_te = _aligned(args.source_data_hdf5, parcels, split, src_feats, src_idmap)
                r = score_parcel_mtrf(fit, feats_te, eeg_te, lags, highpass_hz=args.highpass_hz, fs=FS)
                g.create_dataset(f"heldout_r__{args.source_tag}_{split}", data=r.astype(np.float32))
                entry["in_domain"][f"{args.source_tag}_{split}"] = r.tolist()
                print(f"  [{layer_name}] in-domain  {args.source_tag}_{split:<8} " +
                      "  ".join(f"{n}={v:+.3f}" for n, v in zip(names, r)))

            # transfer (source model -> target test)
            cross = cross_score_dataset(fit, args.target_data_hdf5, args.target_features_dir,
                                        parcels, layer_name, lags, highpass_hz=args.highpass_hz, fs=FS)
            for split, r in cross.items():
                g.create_dataset(f"heldout_r__{args.target_tag}_{split}", data=r.astype(np.float32))
                entry["transfer"][f"{args.target_tag}_{split}"] = r.tolist()
                print(f"  [{layer_name}] TRANSFER   {args.target_tag}_{split:<8} " +
                      "  ".join(f"{n}={v:+.3f}" for n, v in zip(names, r)))

            summary["entries"].append(entry)
            out_h5.flush()
            with open(out_dir / "cross_mtrf_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

    print(f"Scores -> {scores_path}")


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
