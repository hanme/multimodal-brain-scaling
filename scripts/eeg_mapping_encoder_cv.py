"""Aggregate the attention-encoder group-by-part CV folds into one eeg_mapping-schema JSON per
(model x level), so plot_eeg_mapping.py renders the encoder figures exactly like the mTRF.

Reads the 4 fold score files written by kuma_probe_d2_cv.sh:
  outputs/results/<model>-probe-group-d2-<level>-cv/fold<F>/attn_probe_temporal_scores.h5
each carrying, per layer: heldout_r__val (the held-out audiobook-part fold = non-overlapping
selection signal) and heldout_r__test (the clean test split). Per (model, level, layer) we average
val r and test r over the 4 folds; the chosen layer maximises mean val r (test never used to select).

  python scripts/eeg_mapping_encoder_cv.py --model_id whisper-small --target_level parcels \
    --cv_dir outputs/results/whisper-small-probe-group-d2-parcels-cv \
    --out outputs/results/eeg_mapping_encoder/whisper-small__parcels__D2.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import h5py


def layers_for(model_id, layers_config):
    cfg = layers_config or f"configs/extraction/audio/{model_id.replace('-', '_')}_layers.json"
    spec = json.load(open(cfg))
    names = [e["name"] if isinstance(e, dict) else e for e in spec]
    pos = [float(e["position"]) if isinstance(e, dict) else i / max(len(spec) - 1, 1)
           for i, e in enumerate(spec)]
    return names, pos


def main():
    p = argparse.ArgumentParser(description="Aggregate encoder CV folds -> eeg_mapping JSON.")
    p.add_argument("--model_id", required=True)
    p.add_argument("--target_level", choices=["parcels", "electrodes"], required=True)
    p.add_argument("--cv_dir", required=True, help="<model>-probe-group-d2-<level>-cv/ (has fold*/)")
    p.add_argument("--layers_config", default="")
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cv_dir = Path(args.cv_dir)
    fold_h5 = [cv_dir / f"fold{f}" / "attn_probe_temporal_scores.h5" for f in range(args.n_folds)]
    present = [f for f in fold_h5 if f.exists()]
    assert present, f"no fold score files under {cv_dir}"
    if len(present) < args.n_folds:
        print(f"  WARNING: only {len(present)}/{args.n_folds} folds present — averaging those.")

    all_layers, all_pos = layers_for(args.model_id, args.layers_config)
    layers, positions, cv_by, test_by, names = [], [], [], [], None
    for layer, pos in zip(all_layers, all_pos):
        key = layer.replace(".", "-")
        vals, tests, nm = [], [], None
        for f in present:
            with h5py.File(f, "r") as h:
                if key not in h or "heldout_r__val" not in h[key]:
                    continue
                vals.append(h[key]["heldout_r__val"][:].astype(float))
                tests.append(h[key]["heldout_r__test"][:].astype(float))
                nm = [s.decode() if hasattr(s, "decode") else str(s) for s in h[key]["parcels"][:]]
        if not vals:                                   # layer not finished in any fold yet
            continue
        names = names or nm
        layers.append(layer); positions.append(pos)
        cv_by.append(float(np.nanmean([np.nanmean(v) for v in vals])))      # mean-over-targets, mean-over-folds
        test_by.append(np.nanmean(np.stack(tests, 0), axis=0).tolist())     # per-target, mean-over-folds
        print(f"  {layer:10s} VAL r={cv_by[-1]:+.3f}  TEST r={np.nanmean(test_by[-1]):+.3f}  "
              f"({len(vals)} folds)")

    assert layers, f"no finished layers in {cv_dir}"
    chosen_idx = int(np.nanargmax(cv_by))
    chosen = layers[chosen_idx]
    print(f"\nCHOSEN layer (max mean-fold VAL) = {chosen}  VAL r={cv_by[chosen_idx]:+.3f}  "
          f"TEST r={np.nanmean(test_by[chosen_idx]):+.3f}")

    out = dict(
        model_id=args.model_id, target_level=args.target_level, method="attn_encoder_cv",
        selection="group-by-part k-fold (non-overlapping)", n_folds=args.n_folds,
        layers=layers, positions=positions, targets=names,
        cv_r_by_layer=[[c] for c in cv_by],            # parity w/ mTRF schema (per-target list)
        test_r_by_layer=test_by, cv_score_by_layer=cv_by,
        chosen_idx=chosen_idx, chosen_layer=chosen,
        test_r_chosen=test_by[chosen_idx], cv_score_chosen=cv_by[chosen_idx])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
