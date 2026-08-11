# Handoff — persist the per-trial deviant stack, to unblock the MMN N-effect

**Status:** driver patched + tested locally; **two cluster re-runs are required** before Part B
(MMN amplitude vs N) can be built. Part C (deviance scaling) needed none of this and is already
shipped.

---

## Why a re-run is needed at all

The N-effect asks whether the trough deepens as more standards separate the deviants. Each method's
15 deviants are a 3 × 5 grid of `N ∈ {3,5,7}` × `variation ∈ {1..5}`, so **N lives entirely inside
the average the pipeline takes**: `insilico_mmn_electrodes.py` wrote `deviant_mean` and discarded
`res["dev_stack"]`. Confirmed absent in every committed electrode h5, LIT and NOVEL-P2 alike:

```
outputs/insilico_mmn_predictions_soafix/*/electrode_predictions__*.h5
outputs/insilico_mmn_predictions_novel_phase2/*/electrode_predictions__*.h5
  -> datasets: ['deviant_mean', 'n7v1_peak', 'peak', 'standard', 'time_ms']   # no stack
  -> attrs:    n_deviants = 15                                               # they existed
```

`analyze_method → insilico_mmn.finalize_method` already returns `dev_stack [15, n_t, n_elec]` and
`dev_ids`, so the data was in hand and thrown away. **One patch, two re-runs.** Only the prediction
step re-runs — the `delta_T` features already exist for both sets and are untouched. This is an
mTRF fit + apply, not the 700–1200 core-hour extraction.

---

## A1 — the patch (done, committed locally)

`scripts/insilico_mmn_electrodes.py` now also writes, per method group:

| dataset | shape / dtype | contents |
|---|---|---|
| `deviants_fc` | `[n_dev, n_t, n_fc]` float32, gzip-4 | `dev_stack` sliced to the fronto-central cluster |
| `deviants_fc_electrodes` | `S8` | that axis's electrode names, in slice order |
| `deviant_ids` | `S40` | `res["dev_ids"]`, labelling the trial axis |

Design points, each load-bearing:

- **Fronto-central slice, not the full stack.** `CLUSTERS["frontal"] + CLUSTERS["central"]`
  (Fz, F3, F4, FCz, Cz, C3, C4), intersected with the electrodes that survived the NC floor rather
  than assumed present. The full 47-electrode stack would take each h5 from ~26 MB to ~230 MB for
  40 electrodes of unused scalp; this slice costs ~35 MB/model and removes a second cluster
  round-trip if Fz is ever wanted. **The Part B analysis is FCz and nothing else** — this is a
  storage decision only.
- **`n_dev` is never hardcoded to 15.** It is sized from `res["dev_stack"]`, because `n_deviants` is
  `--trial_levels × --num_variations` and other configs of this same driver legitimately produce 1.
  The 15-trial assertion lives in the analysis script, which only ever runs against the two roots
  named below.
- **Nothing else moved.** Not the parcel driver, the mTRF fit, the layer map, or the baseline
  multipliers — both runs must stay comparable to their committed screens.

Tests added to `tests/test_insilico_mmn.py` (3 new, 23 pass in the file, 465 pass repo-wide):

- `deviants_fc` is written, sliced to the FC cluster in `CLUSTERS` order, with Pz excluded;
- `n_dev == len(deviant_ids)`;
- **`deviants_fc[:, :, FCz].mean(0)` reproduces the stored `deviant_mean` FCz column** to float32
  tolerance — the guarantee that the per-trial stack and the committed method-level results
  describe the same traces;
- `n_dev == 1` is stored correctly (no 15 hardcoded);
- only the FC electrodes that survived the NC floor are stored.

> One pre-existing, unrelated failure repo-wide: `tests/test_static.py::test_no_private_cluster_paths_or_usernames`,
> caused by the uncommitted SBATCH tuning in `scripts/*.sh`. It fails identically with these
> changes stashed.

---

## A2 — the two re-runs

> ### ⚠️ Both runs OVERWRITE their existing electrode h5s
> `insilico_mmn_electrodes.py` opens with `h5py.File(path, "w")` — truncating. **Back up first**, or
> write to sibling roots and reconcile afterwards. The novel-phase2 root is ~602 MB.
> ```bash
> cp -r outputs/insilico_mmn_predictions_soafix       outputs/insilico_mmn_predictions_soafix.bak
> cp -r outputs/insilico_mmn_predictions_novel_phase2 outputs/insilico_mmn_predictions_novel_phase2.bak
> ```

```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling
git pull   # this checkout carries uncommitted SBATCH tuning: reconcile PER FILE, never force
source env.sh
```

### A2a — LIT (soafix), 6 models

Step 5 of `aux/handoff_soafix_trailing_floor.md`; its symlinked 48-condition feature and stimuli
roots are already assembled, so only the screen re-runs. **whisper-large is deliberately dropped**
from `SOAFIX_MODELS` — it is out of scope for this task, so re-running it would burn the most
expensive model in the set for a file nothing reads. Its existing soafix h5 is left untouched and
simply never opened.

```bash
export SOAFIX_MODELS="whisper-tiny whisper-base whisper-small whisper-medium wav2vec2-medium wav2vec2-large"

DRY_RUN=1 METADATA_CSV=data/metadata/literature_frequency_intensity_duration_metadata.csv \
FEATURES_TAG=soafix \
MODELS="$SOAFIX_MODELS" \
PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_soafix \
FIGURES_ROOT=outputs/figures/insilico_mmn_electrodes_soafix \
WHISPER_STIM=outputs/mmn_stimuli_soafix \
WAV2VEC2_STIM=outputs/mmn_stimuli_soafix_wav2vec2 \
    scripts/submit_novel_insilico.sh
```

Confirm **6** sbatch lines and that **whisper-large is not among them**, then drop `DRY_RUN=1`.

> This inverts the usual house warning that `MODELS` silently drops whisper-large. Here that
> omission is *intended* and must be **verified**, not corrected.

### A2b — NOVEL-P2, 6 models

Stage J of the novel-search runbook. Phase 2 topped up the phase-1 feature directories in place, so
`FEATURES_TAG` does **not** change from its default.

```bash
DRY_RUN=1 METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv \
PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_novel_phase2 \
    scripts/submit_novel_insilico.sh
```

`METADATA_CSV` is **not optional**: the submitter refuses when the method list and the metadata CSV
describe different condition sets, and the default full-grid CSV yields 1,806 conditions against
the list's 254.

**Fallback — `novel_grid_phase2_subset.csv` is absent from the local checkout.** If it is also
missing on the cluster, rebuild it (verified locally: **127 pairs → 254 method dirs**):

```bash
python scripts/build_novel_phase2_csv.py \
    --grid_csv data/metadata/novel_grid_frequency_metadata.csv \
    --selected_pairs outputs/results_novel_search/phase2_selected_pairs.csv \
    --out data/metadata/novel_grid_phase2_subset.csv

# must print 127 before you use it
awk -F, 'NR>1 && $4=="Frequency"' data/metadata/novel_grid_phase2_subset.csv | wc -l
```

---

## Verification — all three gates, in this order, before syncing anything back

Each catches a different failure; passing one does not imply the others.

**Gate 1 — group counts and the new datasets.** Over the 6 in-scope models only: 48 groups/model
for LIT, 254 for NOVEL-P2, every group carrying `deviants_fc` with `n_dev == 15`.

```bash
python - <<'PY'
import h5py, glob
for root, n_expect in (("outputs/insilico_mmn_predictions_soafix", 48),
                       ("outputs/insilico_mmn_predictions_novel_phase2", 254)):
    for f in sorted(glob.glob(f"{root}/*/electrode_predictions__*.h5")):
        model = f.split("/")[-2]
        if model == "whisper-large":       # out of scope, stale file, not re-run
            continue
        with h5py.File(f) as h:
            gs = [k for k, v in h.items() if isinstance(v, h5py.Group)]
            bad = [k for k in gs if "deviants_fc" not in h[k]
                   or h[k]["deviants_fc"].shape[0] != 15
                   or h[k]["deviants_fc"].shape[0] != len(h[k]["deviant_ids"])]
            print(f"{model:<17} {len(gs):>4}/{n_expect} groups   {len(bad)} bad", 
                  "OK" if len(gs) == n_expect and not bad else "*** FAIL ***")
PY
```

**Gate 2 — the held-out `r` printed by `fit_mapping` matches the previous run's log, per model.**
The mapping refit is independent of the stimuli and should be effectively identical. If it moved,
something other than the driver changed and the run is **not** comparable to the committed screen.

**Gate 3 — re-score and reproduce the committed verdicts exactly.** The patch adds a dataset and
must not perturb a single verdict; any drift means the re-run is not the same experiment.

```bash
python scripts/analyze_mmn_s7_roi.py --dip_uv_threshold 0.75 \
    --predictions_root outputs/insilico_mmn_predictions_soafix \
    --out outputs/results_soafix_full/mmn_s7_roi.csv
python scripts/analyze_mmn_s7_roi.py --dip_uv_threshold 0.75 \
    --predictions_root outputs/insilico_mmn_predictions_novel_phase2 \
    --out outputs/results_novel_search/phase2_mmn_s7_roi.csv
```

FCz S2 / S7@0.75 counts must reproduce these **exactly** (all verified against the current CSVs):

| model | LIT S2/48 | LIT S7/48 | P2 S2/254 | P2 S7/254 |
|---|---|---|---|---|
| whisper-tiny | 44 | 16 | 219 | 125 |
| whisper-base | 42 | 13 | 165 | 122 |
| whisper-small | 45 | 15 | 185 | 126 |
| whisper-medium | 46 | 26 | 203 | 155 |
| wav2vec2-medium | 47 | 29 | 227 | 187 |
| wav2vec2-large | 38 | 26 | 186 | 141 |
| **pooled** | **262** | **125/288** | **1185** | **856/1524** |

`analyze_mmn_s7_roi.py` will still pick up whisper-large's **stale** soafix h5 if it is present in
the root. Ignore those rows — they are not a mismatch.

---

## After the sync — building Part B

```bash
# per-trial scoring, FCz (both assert their row counts and the n7v1 sanity residual)
python aux/analysis_MMN_realness_checks/analyze_mmn_per_trial_n.py \
    --predictions_root outputs/insilico_mmn_predictions_soafix --dataset lit \
    --out outputs/results_soafix_full/mmn_per_trial_n_fcz.csv          # 4,320 rows

python aux/analysis_MMN_realness_checks/analyze_mmn_per_trial_n.py \
    --predictions_root outputs/insilico_mmn_predictions_novel_phase2 --dataset p2 \
    --out outputs/results_novel_search/phase2_mmn_per_trial_n_fcz.csv  # 22,860 rows

# the 5 figures + stats
python aux/analysis_MMN_realness_checks/n_effect_plots.py
```

**Both scripts are already validated end-to-end against real data.** The soafix *parcel* h5s do
carry the legacy full-axis `deviants` stack, so the whole path was exercised at the frontal and
central parcels before any cluster time was spent:

- `analyze_mmn_per_trial_n.py` reproduced the pipeline's own stored `n7v1_peak` to
  **2.4 × 10⁻⁷** across all 288 (model × method) cells — the per-trial scoring is bit-for-bit the
  committed criterion;
- it emitted exactly **4,320 rows** = 6 models × 48 conditions × 15 trials, with all count,
  N-level and S7 ⊆ S2 assertions passing;
- `n_effect_plots.py` rendered all 5 figures and both stats CSVs from that input.

Only the FCz **electrode** input is missing. Those parcel numbers are a plumbing check, **not** a
result, and are not reported anywhere.
