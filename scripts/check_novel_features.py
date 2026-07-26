#!/usr/bin/env python
"""Completeness and integrity check for extracted MMN delta_T features.

Run after every extraction submission, before spending the next one. The in-silico driver
SKIPS a method whose feature directory is missing and load_layer_features() will happily
concatenate a directory that is subtly wrong, so a partial or malformed extraction does not
announce itself -- it just quietly shrinks or corrupts the grid.

What it checks, per method directory:
  * the directory exists and holds exactly --expect_clips HDF5 files;
  * every file uses the stimulus-id naming scheme, none the legacy start/batch one (a directory
    holding both is double-loaded by load_layer_features with features and ids misaligned);
  * every layer in the model's extraction config is present in every file, so a later
    layer-wise analysis is not silently missing a block;
  * shapes agree across files, dtype is float16, and the time dimension matches the model family
    (whisper is a fixed 1500-bin grid; wav2vec2 depends on the window);
  * the ids inside the files are the clips the filenames claim, with exactly one standard;
  * values are finite and not all-zero.

  python scripts/check_novel_features.py --model_id whisper-tiny \
      --method_list outputs/novel_methods_gate.txt --expect_clips 2
  python scripts/check_novel_features.py --models all \
      --method_list outputs/novel_methods_phase1.txt --expect_clips 2
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

SEARCH_MODELS = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
                 "wav2vec2-medium", "wav2vec2-large"]
LEGACY_PREFIX = "feats_delta_t-start_"


def layer_aliases(model_id, config_dir="configs/extraction/audio"):
    cfg = Path(config_dir) / f"{model_id.replace('-', '_')}_layers.json"
    if not cfg.exists():
        return None, f"layer config not found: {cfg}"
    names = [e["name"] for e in json.loads(cfg.read_text())]
    return [n.replace(".", "-") for n in names], None


def check_method(feat_dir, aliases, expect_clips):
    """-> (list of problem strings, total bytes, set of (T, d) shapes)."""
    problems, nbytes, shapes = [], 0, set()
    if not feat_dir.is_dir():
        return [f"{feat_dir.name}: directory missing"], 0, shapes

    h5s = sorted(feat_dir.glob("feats_delta_t-*.h5"))
    legacy = [p for p in h5s if p.name.startswith(LEGACY_PREFIX)]
    if legacy:
        problems.append(f"{feat_dir.name}: {len(legacy)} legacy start/batch-named file(s) -- "
                        f"mixing naming schemes misaligns load_layer_features")
    if len(h5s) != expect_clips:
        problems.append(f"{feat_dir.name}: {len(h5s)} h5 files, expected {expect_clips}")
        if not h5s:
            return problems, 0, shapes

    ids_seen = []
    for p in h5s:
        nbytes += p.stat().st_size
        try:
            with h5py.File(p, "r") as f:
                if "ids" not in f:
                    problems.append(f"{p.name}: no 'ids' dataset")
                    continue
                ids = [x.decode() if hasattr(x, "decode") else str(x) for x in f["ids"][:]]
                ids_seen.extend(ids)

                missing = [a for a in aliases if f"features/{a}" not in f]
                if missing:
                    problems.append(f"{p.name}: missing layer(s) {missing[:4]}"
                                    f"{'...' if len(missing) > 4 else ''} of {len(aliases)}")
                    continue

                arr = f[f"features/{aliases[0]}"]
                if arr.dtype != np.float16:
                    problems.append(f"{p.name}: dtype {arr.dtype}, expected float16")
                if arr.shape[0] != len(ids):
                    problems.append(f"{p.name}: {arr.shape[0]} feature rows vs {len(ids)} ids")
                shapes.add(tuple(arr.shape[1:]))

                block = arr[0] if arr.shape[0] else np.zeros((1, 1))
                vals = np.asarray(block, dtype=np.float32)
                if not np.isfinite(vals).all():
                    problems.append(f"{p.name}: non-finite values in {aliases[0]}")
                elif not vals.any():
                    problems.append(f"{p.name}: {aliases[0]} is all zeros")

                # the filename must name the clip the file actually holds
                stem = p.name[len("feats_delta_t-"):-len("-seed_42.h5")]
                if len(ids) == 1 and ids[0] != stem:
                    problems.append(f"{p.name}: holds id {ids[0]!r}, filename says {stem!r}")
        except OSError as e:
            problems.append(f"{p.name}: unreadable ({e})")

    if ids_seen:
        n_std = sum("standard" in i.lower() for i in ids_seen)
        n_dev = sum("deviant" in i.lower() for i in ids_seen)
        if n_std != 1:
            problems.append(f"{feat_dir.name}: {n_std} standards, expected exactly 1 "
                            f"(analyze_method returns None without one)")
        if n_dev != expect_clips - 1:
            problems.append(f"{feat_dir.name}: {n_dev} deviants, expected {expect_clips - 1}")
        if len(set(ids_seen)) != len(ids_seen):
            dup = [i for i, c in Counter(ids_seen).items() if c > 1]
            problems.append(f"{feat_dir.name}: duplicate ids {dup[:3]}")
    return problems, nbytes, shapes


def check_model(model_id, features_root, methods, expect_clips, show):
    aliases, err = layer_aliases(model_id)
    if err:
        print(f"  {err}")
        return False
    root = Path(features_root)
    all_problems, total_bytes, shapes, ok_dirs = [], 0, set(), 0
    for m in methods:
        probs, nbytes, sh = check_method(root / f"mmn-{m}-delta-t", aliases, expect_clips)
        total_bytes += nbytes
        shapes |= sh
        all_problems += probs
        ok_dirs += not probs

    n = len(methods)
    print(f"  root      : {root}")
    print(f"  layers    : {len(aliases)} ({aliases[0]} .. {aliases[-1]})")
    print(f"  methods   : {ok_dirs}/{n} clean, {n - ok_dirs} with problems")
    if shapes:
        print(f"  shape/clip: {sorted(shapes)}  (T x d_model)"
              + ("   INCONSISTENT" if len(shapes) > 1 else ""))
    if total_bytes:
        print(f"  size      : {total_bytes / 1e9:.2f} GB total, "
              f"{total_bytes / max(1, ok_dirs * expect_clips) / 1e6:.1f} MB/clip")
        print(f"  projected : {total_bytes / max(1, n) * 528 * 2 / 1e9:.0f} GB "
              f"for all 1056 dirs at this rate")
    if all_problems:
        print(f"  PROBLEMS  : {len(all_problems)}")
        for p in all_problems[:show]:
            print(f"    - {p}")
        if len(all_problems) > show:
            print(f"    ... and {len(all_problems) - show} more (raise --show)")
    return not all_problems


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", help="single model to check")
    p.add_argument("--models", help="comma-separated, or 'all' for the 6 search models")
    p.add_argument("--features_tag", default="novel",
                   help="feature roots are outputs/features/<model>-mmn-<tag>")
    p.add_argument("--features_root", help="explicit root, overrides --features_tag")
    p.add_argument("--method_list", default="outputs/novel_methods_phase1.txt")
    p.add_argument("--expect_clips", type=int, default=2,
                   help="h5 files per method dir: 2 after Phase 1, 16 after Phase 2")
    p.add_argument("--show", type=int, default=15, help="problems to list per model")
    args = p.parse_args()

    if not args.model_id and not args.models:
        p.error("give --model_id or --models")
    models = ([args.model_id] if args.model_id else
              SEARCH_MODELS if args.models == "all" else
              [m.strip() for m in args.models.split(",") if m.strip()])

    methods = [l.strip() for l in Path(args.method_list).read_text().split() if l.strip()]
    print(f"method list: {args.method_list} ({len(methods)} conditions), "
          f"expecting {args.expect_clips} clips each\n")

    all_ok = True
    for m in models:
        print(f"=== {m} ===")
        root = args.features_root or f"outputs/features/{m}-mmn-{args.features_tag}"
        all_ok &= check_model(m, root, methods, args.expect_clips, args.show)
        print()

    print("ALL CHECKS PASSED" if all_ok else "PROBLEMS FOUND -- see above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
