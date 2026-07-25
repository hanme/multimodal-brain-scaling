#!/usr/bin/env python
"""Subset the novel grid CSV to the pairs Phase 1 selected, for the full-deviant regeneration.

Phase 2 regenerates audio for the selected pairs with the complete deviant grid
(--trial_levels 3,5,7 --num_variations 5) into the SAME method directories, so each grows from
2 wavs to 16. Rows are copied verbatim from the grid CSV -- same method_id, same frequencies,
same tone parameters -- because the standard and the N7/var1 deviant must regenerate
byte-identically to the Phase-1 files for the already-extracted features to stay valid.

  python scripts/build_novel_phase2_csv.py
  python scripts/00aa_generate_audio_stimuli.py \
      --metadata_csv data/metadata/novel_grid_phase2_subset.csv \
      --output_dir outputs/stim_gen_novel_phase2 \
      --trial_levels 3,5,7 --num_variations 5 --models whisper,wav2vec2 --n_workers 16
"""

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_csv", default="data/metadata/novel_grid_frequency_metadata.csv")
    p.add_argument("--selected_pairs",
                   default="outputs/results_novel_search/phase2_selected_pairs.csv")
    p.add_argument("--out", default="data/metadata/novel_grid_phase2_subset.csv")
    args = p.parse_args()

    with open(args.selected_pairs, newline="") as f:
        wanted = [int(r["method_id"]) for r in csv.DictReader(f)]
    wanted_set = set(wanted)
    assert len(wanted_set) == len(wanted), "duplicate method_id in the selected-pairs CSV"

    with open(args.grid_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if int(r["method_id"]) in wanted_set]

    found = {int(r["method_id"]) for r in rows}
    missing = sorted(wanted_set - found)
    assert not missing, f"selected pairs absent from {args.grid_csv}: {missing}"

    # Preserve the selection order so the CSV reads in rank order rather than method_id order.
    order = {m: i for i, m in enumerate(wanted)}
    rows.sort(key=lambda r: order[int(r["method_id"])])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out}: {len(rows)} pairs -> {2 * len(rows)} method dirs")
    print(f"  each dir goes from 2 wavs (standard + N7/var1) to 16; "
          f"{2 * len(rows) * 14} new clips to extract")
    print(f"  method_id range {min(found)}-{max(found)}")


if __name__ == "__main__":
    main()
