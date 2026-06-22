# MMN deliverable: registry fix, verdict-criterion rewrite, import fixes, cluster runbook

## Context

The pasted handover (`aux/XX_handover_for_Sophie.md` + `aux/mmn_screening_plan.md` + `aux/project_plan_20260611.md` §17/§20/§21 + the new `aux/sophies_repository_overview.md` §13) lays out a correction round on top of earlier work: a prior stimulus round (old `method_37/44/55`, a physically-controlled "identity-MMN" design) was superseded by a new literature-replication oddball round that happens to reuse the same `method_id` numbers for unrelated stimuli. All claims in the handover were independently re-verified against the live repo this session (three parallel Explore passes — docs, code, data/state) and **all checked out exactly**: the METHODS registry is still the old 8-entry design, the `load_split_parcels`/`load_split_targets` import bug is real in exactly the 3 files claimed, the 10 new stimuli (160 wav files) are present and match the CSV, the committed-layer table matches the JSONs byte-for-byte, and no `model__*.pt` checkpoints exist yet.

One thing to flag: the handover's claim that `outputs/neural_data/surprisal_30s.h5` and the per-model feature `merged/` dirs are "confirmed present" does **not** hold on this local Mac checkout — they don't exist here. This local repo is a thin checkout for code editing; the actual data lives on the SCITAS `jed` cluster (`/work/upschrimpf1/sigfstea/...`, matching the `~/.ssh/config` alias `jed` → `sigfstea@jed.hpc.epfl.ch`). I tested SSH connectivity and it requires interactive auth I can't supply non-interactively, and `sbatch`/`squeue` aren't installed locally. **Confirmed with you:** I will make and statically verify all local code changes, then hand you a precise runbook for the cluster-side steps (stimulus verification, feature extraction, checkpoint training, running the analysis scripts, generating figures) — I cannot execute those myself from here.

The write-up deliverable goes into a new section appended to `aux/sophies_repository_overview.md` (per your choice), documenting the stimulus-design clarification, the phase_00a non-interoperability finding, and the CV-done-vs-newly-trained-checkpoint status.

## Part A — Local code changes (I do these, statically verified, no cluster needed)

### A1. `scripts/insilico_mmn.py` — METHODS registry + design captions + verdict criterion
- Replace the 8-entry old-design `METHODS` list (lines 45–54) with the 10 new methods, derived from `data/metadata/literature_frequency_intensity_duration_metadata.csv`:
  `method_75` (Karger_2014, 1000→1200), `method_74` (Domjan_2012, 1000→1500), `method_72` (Bodatsch_2011, 1000→1200), `method_60` (Umbricht_2003a, 1000→1500), `method_53` (Salisbury_2002a, 1000→1200), `method_55` (Shinozaki_2002a, 1000→2000), `method_37` (Javitt_2000a, 1000→1050), `method_43` (Michie_2000b, 633→700), `method_44` (Michie_2000c, 633→1000), `method_27` (Schall_1999a, 1000→1064).
- Update the module docstring (lines 1–21) and the `METHODS` comment (lines 42–44): this is now a **classic oddball (Definition 1)** design — final tone differs between standard and deviant — not the old identity-MMN design. Same fix for the `plot_method()` suptitle (lines 209–213) and the h5 `note` attr (lines ~280–283): drop "final tone physically identical in std & dev," replace with a description of the z-scored SOA-window verdict (A1 below).
- Add `load_soa_table(csv_path)` reading `standard_soa` per `method_id` from the metadata CSV (default path `data/metadata/literature_frequency_intensity_duration_metadata.csv`), so each method's baseline window is data-driven rather than hardcoded.
- Rewrite the post-processing in `analyze_method()` (currently the mean-only `bc()` at lines 180–182, fixed `--win_pre_ms` baseline) into a shared helper `finalize_method(...)`, splitting **what gets plotted** from **what gets z-scored for the verdict** — z-scoring must never touch the plotted traces, only the summary statistic:
  - Shared baseline window for both branches below: `base = (rel_ms >= -3*soa_ms) & (rel_ms < 0)`, SOA looked up per method from the metadata CSV via `load_soa_table()`.
  - **Plotted traces (`dev_b`, `std_b`, `diff_b`)** — mean-only baseline correction, same as the existing `bc()`, just with the SOA-derived window instead of the fixed 150 ms: `dev_b = dev_raw - mean(dev_raw[base])`, `std_b = std_raw - mean(std_raw[base])`, `diff_b = dev_b - std_b`. These stay in the original prediction units and are exactly what `plot_method()` draws — unchanged in kind from today, just a different baseline window.
  - **Verdict computation (not plotted)** — full z-score (mean *and* std) within the same `base` window: `z_dev(t) = (dev_raw(t) - dev_mu) / dev_sd`, `z_std(t) = (std_raw(t) - std_mu) / std_sd`, `z_diff = z_dev - z_std`, `baseline_normalized_peak = z_diff[100..240ms].min(0)` per target (parcel/electrode). This number is computed purely as a scalar annotation; the `z_dev`/`z_std`/`z_diff` arrays themselves are intermediate and never stored as a plotted field.
  - Also identify the single deviant whose stimulus id matches `N7`+`var1` (already inside `dev_preds`/`dev_ids` — just locate it) and run the same z-score-only verdict computation against it as a diagnostic (`n7v1_peak`), reusing the standard trace's baseline stats for an apples-to-apples comparison. No plotted counterpart for this either.
  - To avoid duplicating this block between `insilico_mmn.py` and `insilico_mmn_attn.py`, factor the **time-locking + mean-correction + z-score-peak + N7/var1** logic out of `analyze_method` into this shared `finalize_method(...)` helper that both `analyze_method` and `analyze_method_attn` call — the only thing that differs between A/B is how `pred` is produced (`predict_timecourse` mTRF vs. checkpoint), not this part.
  - Returned dict keeps `dev_b`/`std_b`/`diff_b` exactly as today (mean-corrected, plotted, minimal call-site churn) and adds two new scalar-per-target fields: `peak`, `n7v1_peak`.
- `plot_method()`: restrict the plotted rows to **frontal/central/temporal only** (3 rows, not all 5 parcels) by filtering `parcels` and the matching columns of `dev_b/std_b/diff_b` before calling; the third column still plots the `diff_b` line (mean-corrected, original units) but its subplot title is annotated with the separately-computed `baseline_normalized_peak` scalar from `peak`.
- Propagate `peak`/`n7v1_peak` into the per-method h5 group already written in `main()` (alongside `standard`/`deviant_mean`/`deviants`).
- Default `--lag_max_ms` is currently `500.0` (line 241) but §13.7 confirms `score_mtrf_fitquality.py`/`plot_fit_quality.py` already default to `800.0` and the committed layers were chosen at 800 ms — change the default here to `800.0` to remove the footgun (still overridable via flag).

### A2. `scripts/insilico_mmn_attn.py`
- Fix the broken import (line 38–40): `load_split_parcels` does not exist in `insilico_mmn.py`. Change to import `load_split_targets` from `eeg_targets` directly, and update the one call site (line 95) from `load_split_parcels(...)` to `load_split_targets(...)`.
- Reuse the new shared `finalize_method` helper from A1 instead of `analyze_method_attn`'s own duplicated `bc()` block (lines 79–87).
- Fix the h5 `note` attr (lines 199–200): same "identity design" wording removed.
- Update the module docstring's example command (lines 15–21) to the corrected `--lag_max_ms 800` / model-namespaced output dirs convention used in the Part B runbook.

### A3. `scripts/insilico_mmn_electrodes.py`
- `mmn_metric()` (lines 41–49) currently computes a raw mean amplitude over the ROI in a fixed window. Replace with: take `res["peak"]` (already computed per-electrode by `analyze_method`/`finalize_method`) and average over the ROI electrode indices — drop the separate mean-amplitude computation entirely since the z-scored peak now is the canonical verdict metric.
- Update `plot_topo()`'s verdict text/labels to describe the metric as "ROI mean baseline-normalized peak" instead of "ROI mean amp."
- Update the module docstring (lines 13–16): it currently describes an open-ended "Sophie's loop ... pick ~10 pairs" — replace with a short note that the 10-pair literature set is now fixed (§A1) and this script just screens it.

### A4. `scripts/score_mtrf_fitquality.py` and `scripts/plot_fit_quality.py`
- Same import fix in both: drop `load_split_parcels` from the `from insilico_mmn import (...)` line (line 21 / line 33) and add `from eeg_targets import load_split_targets`; rename the call site (line 57 / line 79) to `load_split_targets(...)`. Signature is identical, no other changes.

### A5. New script `scripts/build_mmn_results_table.py`
- Globs the per-(model, method) prediction HDF5s written by `insilico_mmn.py` (mTRF) and `insilico_mmn_attn.py` (encoder) — see Part B's directory convention — and assembles one CSV: rows = (stimulus pair × model × mapping method), columns = `baseline_normalized_peak` per parcel (frontal/central/temporal) plus the `n7v1_peak` diagnostic per parcel. Reuses `h5py` reads only, no new analysis logic — this is the "Combined results table" deliverable.

## Part A.6 — Commit and push the local changes

Repo remote is `origin` → `https://github.com/hanme/multimodal-brain-scaling.git`. Once Parts A1–A5 are done, commit in logical groups (not one giant commit) and push to `main`:

```bash
git add scripts/insilico_mmn.py
git commit -m "$(cat <<'EOF'
Replace MMN METHODS registry with the 10-method literature oddball set

Old registry described the superseded identity-MMN design (final tone
physically identical in std/dev); the shipped stimuli are now a classic
oddball (Definition 1) set sourced from literature_frequency_intensity_
duration_metadata.csv. Also rewrites the mean-only baseline correction
into the z-scored, SOA-derived-window verdict (baseline_normalized_peak)
and restricts MMN figures to the frontal/central/temporal rows.
EOF
)"

git add scripts/insilico_mmn_attn.py scripts/score_mtrf_fitquality.py scripts/plot_fit_quality.py
git commit -m "$(cat <<'EOF'
Fix load_split_parcels -> load_split_targets import in 3 scripts

load_split_parcels was renamed/moved to eeg_targets.load_split_targets
during the D2 mapping refactor; these three scripts were never updated
and failed at import time. Also reuses the shared z-score/peak helper
in insilico_mmn_attn.py instead of duplicating the baseline logic.
EOF
)"

git add scripts/insilico_mmn_electrodes.py
git commit -m "$(cat <<'EOF'
Switch electrode-level MMN verdict to baseline-normalized peak

Replaces the raw mean-amplitude mmn_metric() with the ROI-averaged
z-scored peak now computed upstream in analyze_method, matching the
parcel-level criterion.
EOF
)"

git add scripts/build_mmn_results_table.py
git commit -m "$(cat <<'EOF'
Add combined MMN results table builder

Assembles the 10-pair x 4-model x {mTRF,encoder} baseline_normalized_peak
table (plus the N7/var1 diagnostic) from the per-run prediction HDF5s.
EOF
)"

git add aux/sophies_repository_overview.md
git commit -m "$(cat <<'EOF'
Document stimulus-design clarification and phase_00a non-interoperability

Records that the new method_37/44/55 oddball stimuli are an unrelated
design from the old identity-MMN method_ids of the same number (not a
stale-copy bug), and that phase_00a_generate_activations.py's output
format/truncation is confirmed incompatible with this repo's own
extract_features_delta_t.py pipeline.
EOF
)"

git push origin main
```

(If you'd rather land this as one PR for review instead of pushing straight to `main`, do the same commits on a branch and open a PR with `gh pr create` — say so and I'll adjust.)

## Part B — Cluster runbook (for you to run on `jed`; I cannot execute this)

First, get the updated code onto the cluster. If this is the first time cloning there:

```bash
ssh jed
git clone https://github.com/hanme/multimodal-brain-scaling.git
cd multimodal-brain-scaling
```

If the repo already exists on `jed` from earlier work, just pull the new commits instead (adjust path to wherever it actually lives, e.g. matching the `/work/upschrimpf1/sigfstea/...` tree referenced throughout the handover):

```bash
ssh jed
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling   # or wherever the clone lives
git pull origin main
```


One important catch found while reviewing the existing scripts: `whisper-tiny` and `whisper-base` **both** have `chosen_layer = blocks.0` for the mTRF parcels mapping. Since `insilico_mmn.py` defaults to writing `predictions__<layer>.h5` into a single shared `outputs/insilico_mmn_predictions/` directory, running tiny then base back-to-back with default `--data_dir`/`--out_dir` would silently overwrite tiny's output. **Always pass model-namespaced `--out_dir outputs/figures/insilico_mmn/<model>` and `--data_dir outputs/insilico_mmn_predictions/<model>`** (same for the `_attn`/electrodes variants) so A5's table builder can glob predictably and nothing collides.

Ordered steps (full commands as already in the handover, with the directory-namespacing fix folded in):
1. Stimulus verification shasum loop (handover §1) against `SOPHIE_WAV=/work/upschrimpf1/sigfstea/scz_updated_pipeline_071226/data/audio_outputs_literature/audio_outputs_regular/whisper`; overwrite `outputs/mmn_stimuli/<method>/` fresh for all 10 methods from the verified source.
2. `rm -rf` any stale `outputs/features/mmn-method_{55,37,44}-delta-t` from the old design, then `sbatch --array=0-15 scripts/slurm_mmn_extract.sh` for all 4 models × 10 methods (400 array tasks total) — stagger against the partition's array-size cap rather than firing all 40 `sbatch` calls in one burst.
3. `sbatch --array=0-7 scripts/kuma_probe_d2_final.sh`; confirm via `sacct` (`COMPLETED 0:0` ×8) and `find outputs/results -name "model__*.pt"` (want 8 files).
4. Run Method A (`insilico_mmn.py`, `insilico_mmn_electrodes.py`) and Method B (`insilico_mmn_attn.py`, after step 3's checkpoints exist) per model, using the committed parcels-layer table, `--lag_max_ms 800`, and the model-namespaced `--out_dir`/`--data_dir` from above.
5. Run `scripts/build_mmn_results_table.py` to assemble the combined CSV from all the per-model/per-method prediction h5s.

## Part C — Write-up (appended to `aux/sophies_repository_overview.md`)

New section documenting:
- The two-stimulus-design clarification — new `method_37/44/55` (classic oddball) is unrelated to the old identity-MMN `method_37/44/55`; not a stale-copy bug.
- The confirmed `phase_00a_generate_activations.py` vs. `src/mbs/extraction/extract_features_delta_t.py` non-interoperability (different output schema, different truncation mechanics) — `phase_00a` stays out of scope for this deliverable.
- The resolved `load_split_parcels`→`load_split_targets` import bug (3 files) — was open per old §13.2, now fixed.
- The new METHODS registry + the z-scored, SOA-derived-window `baseline_normalized_peak` verdict criterion, replacing the old mean-amplitude metric.
- Method B status: CV layer-selection sweep is done (committed layers, §21), but the final reusable checkpoints (`model__*.pt`) are not trained yet — that's a cluster-side prerequisite (Part B step 3) before Method B figures can be produced.

## Verification

- Local: after each edit, `python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())"` (or just import the module) on the 5 touched scripts to catch syntax/import errors without needing real data; specifically confirm `insilico_mmn_attn.py`, `score_mtrf_fitquality.py`, `plot_fit_quality.py` no longer reference `load_split_parcels` anywhere (`grep -rn load_split_parcels scripts/`).
- Cluster (your side, per Part B): after step 2, spot-check one extracted feature dir's file count; after step 3, `sacct` + `find ... model__*.pt`; after step 4, open 2–3 of the 80 figures and confirm 3 rows (frontal/central/temporal) × 3 columns (deviant/standard/deviant−standard, all mean-baseline-corrected, none z-scored) with the `baseline_normalized_peak` value annotated as text on the third column; after step 5, sanity-check the combined CSV has 10×4×2=80 rows with finite `baseline_normalized_peak` values and a separate N7/var1 diagnostic column.
