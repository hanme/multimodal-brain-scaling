# Encoder vs. mTRF plot/output organization

Grounded in the actual on-disk paths in this checkout (confirmed via `find`) and the row-block
restructure done in this round (see `aux/results_analysis.md` for the underlying metrics, and the
code in `scripts/insilico_mmn.py` / `scripts/insilico_mmn_electrodes.py` /
`scripts/insilico_mmn_attn.py`). Not a description of an ideal layout — a description of what the
code actually writes, where.

## Two mapping methods, two scripts each

| Mapping | Parcels driver | Electrodes driver |
|---|---|---|
| **mTRF (Method A)** | `scripts/insilico_mmn.py` — re-fits a ridge mapping on the spot, loops over **all 10 methods in one invocation** (`--methods all`) | `scripts/insilico_mmn_electrodes.py` — same mapping machinery, electrodes as singleton targets, also loops all 10 methods in one invocation |
| **Encoder (Method B)** | `scripts/insilico_mmn_attn.py` — loads a trained checkpoint (`model__<layer>.pt`); level (parcels vs electrodes) is inferred from the checkpoint's own target list, not a flag | same script, same inference — **one `--method` per invocation**, no `--methods all`; the SLURM array script (`slurm_insilico_mmn_attn.sh`) submits one job per method to cover all 10 |

`plot_method()`, `plot_topo()`, `mmn_metric()`, `FC_ROI`, and `PLOT_ROWS` all live in
`scripts/insilico_mmn.py` and are imported unchanged by the other two scripts — this is what
keeps the encoder and mTRF figures visually identical row-for-row.

## The 3 row-blocks, and where each one is written

Per (model, layer, method), each row-block is its own PNG (per the chosen design — not merged):

| Row-block | mTRF file (driver) | Encoder file (driver) |
|---|---|---|
| **A** — Electrodes: Fz, FCz only (+ C0-S6 table) | `outputs/figures/insilico_mmn_electrodes/<model>/insilico_mmn_electrodes_fz_fcz__<method>__<layer>.png` (`insilico_mmn_electrodes.py`) | `outputs/figures/insilico_mmn/<model>-electrodes/insilico_mmn_fz_fcz__<method>__<layer>__attn.png` (`insilico_mmn_attn.py`, electrode-level checkpoint) |
| **B** — Electrodes: all electrodes, topo, diff-only | `outputs/figures/insilico_mmn_electrodes/<model>/insilico_mmn_electrodes__<method>__<layer>.png` (`insilico_mmn_electrodes.py`) | `outputs/figures/insilico_mmn/<model>-electrodes/insilico_mmn_electrodes__<method>__<layer>__attn.png` (`insilico_mmn_attn.py`, electrode-level checkpoint) |
| **C** — Parcels: frontal/central/temporal (+ C0-S6 table) | `outputs/figures/insilico_mmn/<model>/insilico_mmn__<method>__<layer>.png` (`insilico_mmn.py`) | `outputs/figures/insilico_mmn/<model>-parcels/insilico_mmn__<method>__<layer>__attn.png` (`insilico_mmn_attn.py`, parcel-level checkpoint) |

Plus one diagnostic that is **not** a row-block: fit-quality (recorded vs. predicted EEG on a
held-out speech window). mTRF: 1 file per model/layer via the separate
`scripts/plot_fit_quality.py`. Encoder: `outputs/figures/insilico_mmn/<model>-<level>/fit_quality__attn__<layer>__attn.png`,
written once per `insilico_mmn_attn.py` invocation — since that script runs once per method, this
file is rewritten (identically) 10 times per model/level if all 10 methods are run.

## Directory-naming asymmetry (pre-existing, not changed this round)

- **mTRF**: bare `<model>/` directory for both figures and predictions — level is implicit in
  *which script* wrote the file (`insilico_mmn/` = parcels, `insilico_mmn_electrodes/` = electrodes).
- **Encoder**: `<model>-<level>/` directory (e.g. `whisper-tiny-parcels`, `whisper-tiny-electrodes`)
  under the *same* `outputs/figures/insilico_mmn/` root — level is explicit in the directory name
  because both levels share one script and one parent folder.

This means an mTRF/encoder comparison for the same model+method needs to know to look in
`insilico_mmn/<model>/` + `insilico_mmn_electrodes/<model>/` (mTRF) vs.
`insilico_mmn/<model>-parcels/` + `insilico_mmn/<model>-electrodes/` (encoder) — different
directory shapes, same eventual file-per-row-block count (see Output symmetry below).

## Prediction HDF5s (raw arrays backing every figure + table)

| Mapping | Level | Path |
|---|---|---|
| mTRF | parcels | `outputs/insilico_mmn_predictions/<model>/predictions__<layer>.h5` (all 10 methods as groups in one file) |
| mTRF | electrodes | `outputs/insilico_mmn_predictions/<model>/electrode_predictions__<layer>.h5` (all 10 methods, one file) |
| Encoder | parcels | `outputs/insilico_mmn_predictions/<model>-parcels/<method>/predictions__<layer>__attn.h5` (one file per method) |
| Encoder | electrodes | `outputs/insilico_mmn_predictions/<model>-electrodes/<method>/predictions__<layer>__attn.h5` (one file per method) |

Each group/file stores `time_ms`, `standard`, `deviant_mean`, `peak` (z-scored
`baseline_normalized_peak`), `n7v1_peak`, and (mTRF parcels / encoder) `deviants`/`deviant_ids` —
enough to recompute `z_dev`/`z_std`/`z_diff` and any criterion without re-running the model.

## Criteria tables

- **Embedded per-figure tables** (Task 3, under row-blocks A and C): computed at plot time from
  the in-memory `z_diff` array via `scripts/mmn_criteria_table.py:compute_criteria_table`, rendered
  directly into the PNG — not written to a separate file.
- **Cross-model summary tables** (Task 4): `outputs/results/mmn_criteria_summary__{mtrf,encoder}_{electrodes,parcels}.csv`
  + one combined `outputs/results/mmn_criteria_summary_tables.md`, built by
  `scripts/build_mmn_criteria_summary_tables.py` from the pre-existing
  `outputs/results/mmn_criteria_s5_s6_fz_central.csv` (reproduces `aux/results_analysis.md`
  Tables 25-28 exactly — verified cell-for-cell when this round shipped).

## Output symmetry between mTRF and encoder

The **30 row-block files per model/layer** (10×A + 10×B + 10×C) match 1:1 in type and count
between mTRF and encoder, by construction — both mappings render through the same
`plot_method`/`plot_topo` functions with the same row selections. Two asymmetries remain,
intentionally out of scope for this round:

1. **Invocation shape**: mTRF produces all 10 methods' worth of row-blocks from 1 script call;
   encoder needs 10 calls (one per method) per checkpoint level, since `insilico_mmn_attn.py` has
   no `--methods all`. Final file sets match; the path to get there doesn't.
2. **Fit-quality duplication**: encoder's `fit_quality__attn__<layer>__attn.png` is rewritten
   identically on every one of those 10 calls (it doesn't depend on the MMN method); mTRF's
   equivalent is 1 file via a separate script, run once.
