# Handoff — 400 ms trailing floor for the short-SOA literature stimuli

**Status (2026-08-02):** Step 1 **DONE and pushed** (generator change + tests + two verification
scripts, this file). Steps 2–6 are cluster work and are **pending** — run them from the cluster
PROJECT_DIR in the order below. Step 7 is local and happens once the prediction h5s are synced back.

---

## Why

`compute_tone_slots` lays a clip out as `[leftover][ISI][tone 1][ISI] … [tone K][ISI]`, pushing all
slack to the front, so the clip ends exactly **one SOA** after the final tone's onset. The MMN
criteria read out to **360 ms** past that tone — S2's recovery search
(`analyze_mmn_criteria.py:117-118`) is a bare `post = z[imin+1 : imin+1+6]` slice with no end guard,
and the trough may sit as late as 240 ms. For short-SOA literature rows the epoch ends first, so the
criterion is silently evaluated on whatever samples exist; where the trough lands on the final
sample, `post` is empty and **S2 returns False without ever being tested**.

Measured on the committed predictions (`scripts/verify_soafix_predictions.py --predictions_root
outputs/insilico_mmn_predictions` reproduces this):

| | per model | total |
|---|---|---|
| conditions ending before 360 ms | 24 of 48 | — |
| truncated recovery searches | 9–14 | **62** over the six SEARCH_MODELS |
| trough on the final sample (S2 forced False) | 3, in 4 of 7 models | **12 cells** |

**The fix:** one uniform layout rule — trailing silence = `max(400, SOA)`, with as many whole tone
cycles as fit before it and the remainder becoming extra leading silence. 400 rather than 360
because the floor is measured in audio while the requirement is on the epoch, which is one feature
frame shorter: 19 ms on whisper, 39 ms on wav2vec2.

Conformance to that single rule is the property — there is no "was it regenerated" flag. Half the
committed files already satisfy it; after this run all 48 conditions do.

## What Step 1 already changed

`scripts/00aa_generate_audio_stimuli.py` gained `--trailing_floor_ms` (float, **default 0.0**).
The default reproduces the historical layout byte-for-byte, so the flag is a strict opt-in and no
already-generated set — in particular the ~1 TB novel grid — can drift. `StimulusGrid` carries the
value into the multiprocessing workers, and every emitted `metadata.csv` records it in a new
`trailing_floor_ms` column.

Verified locally on real generated audio (methods 12 and 17, both families, all 16 clips, both
directions):

- SOA 400 (method 12) comes out **byte-identical** with and without the floor; SOA 200 (method 17)
  is rebuilt, its tail going 200.875 → 400.875 ms.
- The onset detector's bias is exactly **+0.875 ms** (14 samples @ 16 kHz), uniform across SOA and
  clip length.
- **0 suffix mismatches**: every deviant keeps its `[S, D×N, S]` critical suffix, and where a slot
  is dropped the shorter prefix is the same RNG stream with its last element removed.

The `(K, leftover)` arithmetic is pinned by unit tests over all 24 literature rows × both families
plus the novel row — see `tests/test_stimulus_generation.py`, section "The trailing-silence floor".

### The 12/12 partition

| | method_ids |
|---|---|
| **unchanged** (SOA ≥ 400) | 9, 12, 20, 21, 27, 33, 43, 44, 55, 72, 74, 75 |
| **rebuilt** (SOA < 400) | 10, 17, 18, 19, 28, 29, 30, 31, 32, 37, 53, 60 |

Only 3 of the 10 changed method/family pairs drop a tone — SOA 200 (both families) and SOA 350
whisper. The other seven keep every tone and just shrink the leading silence by `400 − SOA`.
Post-fix tails are uniform: **381 ms** whisper, **361 ms** wav2vec2.

---

## Prerequisite — pull Step 1 onto the cluster

```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling
git pull            # the checkout carries uncommitted SBATCH tuning: reconcile per-file, NEVER force
source env.sh
```

Confirm the flag arrived: `python scripts/00aa_generate_audio_stimuli.py --help | grep trailing_floor`.

Everything below writes to `*_soafix` siblings. **Never write to
`outputs/insilico_mmn_predictions/`, `outputs/mmn_stimuli{,_wav2vec2}/`, or
`outputs/features/<model>-mmn/`** — they are the committed comparison baseline and have no backup.

---

## Step 2 — regenerate all 24 method_ids, then prove 12 are identical

Generate the **full** set, not just the changed 12: it costs minutes and it makes the byte-identity
claim directly checkable.

```bash
sbatch --export=ALL,OUTPUT_DIR=outputs/stim_gen_soafix \
       scripts/slurm_generate_stimuli.sh --trailing_floor_ms 400
```

Pass no other flags — the defaults give the full 16-clip grid and all 4 model families, which is
what the literature run used. `"$@"` is last in the wrapper, so the trailing flag reaches the
generator; the job's banner must print `Trail floor:  400 ms (tail is max(floor, SOA))`.

Then verify. It md5s ~3,000 wavs, so it belongs on a compute node:

```bash
sbatch scripts/slurm_check_soafix_stimuli.sh                      # default --onset_scope key
sbatch scripts/slurm_check_soafix_stimuli.sh --onset_scope all    # time every clip
```

Minutes either way. **The exit code is the contract** — check it rather than eyeballing the log:
`sacct -j <jobid> --format=JobID,State,ExitCode`.

It asserts the partition the CSV implies, md5s every clip against the staged baseline, and measures
the trailing audio on the new wavs with `insilico_mmn.detect_final_tone_onset_s`. Expect:

```
md5 partition : 12 identical, 12 differing (expect 12/12)
onset bias whisper  : delta in [+0.875, +1.000] ms
PASS -- 12 ids byte-identical, 12 rebuilt, trailing audio is max(floor, SOA) everywhere.
```

> **If any of the 12 "unchanged" ids differ, STOP.** The layout moved where it must not have and the
> whole comparison is invalid — do not stage, do not extract.

## Step 3 — stage only the changed conditions

```bash
# 12-row subset. The schema is unchanged: SOA/dur/isi are NOT edited by this fix.
awk -F, 'NR==1 || ($2 ~ /^(10|17|18|19|28|29|30|31|32|37|53|60)$/)' \
    data/metadata/literature_frequency_intensity_duration_metadata.csv \
    > data/metadata/literature_soafix_subset.csv

METADATA_CSV=data/metadata/literature_soafix_subset.csv \
SRC=outputs/stim_gen_soafix \
WHISPER_ROOT=outputs/mmn_stimuli_soafix \
WAV2VEC2_ROOT=outputs/mmn_stimuli_soafix_wav2vec2 \
METHOD_LIST_OUT=outputs/soafix_methods.txt \
    scripts/slurm_stage_novel_stimuli.sh
```

Expect **24 dirs, 384 wavs, clips-per-dir {16}** per root and a 24-entry method list.

> Caveat, harmless here: the staging script globs `method_${id}_*.wav` with the raw CSV value, while
> the generator writes `method_{id:02d}`. All 12 changed ids are two-digit, so they match. A
> single-digit id (only method 9, which is in the *unchanged* set and is never staged here) would
> fail loudly with `MISSING`.

## Step 4 — extract features for the 24 changed conditions

⚠️ **`MODELS` must be passed explicitly.** Both submitters default to the six `SEARCH_MODELS`;
without the override **whisper-large is silently skipped** and you find out two days later.

```bash
export SOAFIX_MODELS="whisper-tiny whisper-base whisper-small whisper-medium whisper-large wav2vec2-medium wav2vec2-large"

lfs quota -g upschrimpf1 /work          # check headroom first

DRY_RUN=1 METHOD_LIST=$PWD/outputs/soafix_methods.txt \
METADATA_CSV=data/metadata/literature_soafix_subset.csv \
FEATURES_TAG=soafix \
MODELS="$SOAFIX_MODELS" \
WHISPER_STIM=$PWD/outputs/mmn_stimuli_soafix \
WAV2VEC2_STIM=$PWD/outputs/mmn_stimuli_soafix_wav2vec2 \
    scripts/submit_novel_extraction.sh
```

Read the **7** sbatch lines, confirm whisper-large is among them, then drop `DRY_RUN=1`.

**Gate before fanning out**, per the house pattern — run one model × 2 methods first, then:

```bash
python scripts/check_novel_features.py --model_id whisper-tiny \
    --method_list $PWD/outputs/soafix_methods.txt --features_tag soafix --expect_clips 16
```

Writes `outputs/features/<model>-mmn-soafix/mmn-<method>-delta-t`. 7 models × 24 conditions ×
16 clips = **2688 tasks**; budget **700–1200 core-hours** (whisper-large hooks 32 layers and
wav2vec2-large is materially slower than the 13 min/clip whisper-base rate the header quotes).
Wall clock ≈ 3 h at the submitter's default `CONCURRENCY=200`, ≈ 4.5 h at 128.

whisper-large is a 30 s model, so it reads the same `WHISPER_STIM` root and 30 s window as the other
whisper models — no extra audio is needed for it.

## Step 5 — build a 48-condition feature root, then run the screen

`insilico_mmn_electrodes.py` opens its output with `h5py.File(path, "w")` — **truncating** — so a
`--methods` subset would produce an h5 holding only those methods. Assemble a complete root by
symlinking the 24 unchanged condition dirs next to the 24 newly extracted ones:

```bash
UNCHANGED="09 12 20 21 27 33 43 44 55 72 74 75"

# features
for m in $SOAFIX_MODELS; do
  # whisper-base's literature features live at the bare outputs/features root; others are scoped
  case $m in whisper-base) src=outputs/features ;; *) src=outputs/features/${m}-mmn ;; esac
  for id in $UNCHANGED; do
    for d in method_${id} method_${id}_counter; do
      ln -sfn "$PWD/$src/mmn-${d}-delta-t" "outputs/features/${m}-mmn-soafix/mmn-${d}-delta-t"
    done
  done
done

# stimuli -- stim_dir feeds detect_final_tone_onset_s, so it MUST match the feature clips
for id in $UNCHANGED; do
  for d in method_${id} method_${id}_counter; do
    ln -sfn "$PWD/outputs/mmn_stimuli/$d"          "outputs/mmn_stimuli_soafix/$d"
    ln -sfn "$PWD/outputs/mmn_stimuli_wav2vec2/$d" "outputs/mmn_stimuli_soafix_wav2vec2/$d"
  done
done
```

Confirm each of the seven feature roots now holds **48** `mmn-*-delta-t` entries and that every
symlink resolves — **a dangling link makes the driver skip the method without erroring**:

```bash
for m in $SOAFIX_MODELS; do
  echo "$m: $(ls outputs/features/${m}-mmn-soafix | wc -l) entries, \
$(find outputs/features/${m}-mmn-soafix/ -maxdepth 1 -xtype l | wc -l) dangling"
done
for r in outputs/mmn_stimuli_soafix outputs/mmn_stimuli_soafix_wav2vec2; do
  echo "$r: $(ls $r | wc -l) dirs, $(find $r/ -maxdepth 1 -xtype l | wc -l) dangling"
done
```

Then run **all 48** conditions against the full, **unedited** literature CSV:

```bash
DRY_RUN=1 METADATA_CSV=data/metadata/literature_frequency_intensity_duration_metadata.csv \
FEATURES_TAG=soafix \
MODELS="$SOAFIX_MODELS" \
PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_soafix \
FIGURES_ROOT=outputs/figures/insilico_mmn_electrodes_soafix \
WHISPER_STIM=outputs/mmn_stimuli_soafix \
WAV2VEC2_STIM=outputs/mmn_stimuli_soafix_wav2vec2 \
    scripts/submit_novel_insilico.sh
```

The submitter refuses any `PREDICTIONS_ROOT` under `outputs/insilico_mmn_predictions` — that guard
is why we use a sibling. **Do not override** `--lag_max_ms 800`, the `-3.0` / `0.0` baseline
multipliers, the committed layers, or the `surprisal_10s.h5` / `surprisal_30s.h5` choice: they are
hardcoded in `slurm_insilico_mmn_electrodes.sh`, which already carries the 7-entry layer map
(whisper-tiny `blocks.0`, whisper-base `blocks.0`, whisper-small `blocks.1`, whisper-medium
`blocks.12`, whisper-large `blocks.21`, wav2vec2-medium `encoder.layers.2`, wav2vec2-large
`encoder.layers.12`), and they are what keep these results comparable to the committed screen.

**Sanity check while the jobs run:** the held-out `r` printed by `fit_mapping` must match the
literature run's log per model. The mapping refit is independent of the stimuli and should be
bit-identical; if it moved, something other than the audio changed.

Each job writes `outputs/insilico_mmn_predictions_soafix/<model>/electrode_predictions__<layer>.h5`
with **48 groups**.

## Step 5b — score and verify on the cluster, before pulling anything

Catching a bad screen here saves syncing 7 h5s to find out. One job does both:

```bash
sbatch scripts/slurm_verify_soafix_predictions.sh
```

It runs `analyze_mmn_s7_roi.py` (X = 0.75) into `outputs/results_soafix/mmn_s7_roi.csv`, then
`verify_soafix_predictions.py` over the h5s and that CSV. It refuses to write over
`outputs/results_24freq_7models/` or `outputs/results_with_counter/`, and warns if it finds fewer
than 7 prediction files — the whisper-large tell. Again, the exit code is the contract.

## Step 6 — pull the predictions back (run this locally)

```bash
export SOAFIX_MODELS="whisper-tiny whisper-base whisper-small whisper-medium whisper-large wav2vec2-medium wav2vec2-large"
for m in $SOAFIX_MODELS; do
  mkdir -p outputs/insilico_mmn_predictions_soafix/$m
  rsync -av "jed:/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/insilico_mmn_predictions_soafix/$m/"'*.h5' \
        "outputs/insilico_mmn_predictions_soafix/$m/"
done
```

Expect 7 files, one per model. Also pull `outputs/results_soafix/mmn_s7_roi.csv`.

---

## Step 7 — re-score and verify (local; I do this once the h5s land)

The same wrapper, with the cluster's environment setup skipped:

```bash
conda activate mbs-env
USE_ENV_SH=0 PROJECT_DIR=$PWD bash scripts/slurm_verify_soafix_predictions.sh
```

Or the two scripts directly, which is what that reduces to:

```bash
PYTHONPATH="$PWD/src:$PWD/scripts" conda run -n mbs-env python scripts/analyze_mmn_s7_roi.py \
  --predictions_root outputs/insilico_mmn_predictions_soafix \
  --dip_uv_threshold 0.75 \
  --out outputs/results_soafix/mmn_s7_roi.csv

PYTHONPATH="$PWD/src:$PWD/scripts" conda run -n mbs-env python scripts/verify_soafix_predictions.py \
  --predictions_root outputs/insilico_mmn_predictions_soafix \
  --new_s7_csv outputs/results_soafix/mmn_s7_roi.csv \
  --baseline_predictions_root outputs/insilico_mmn_predictions
```

Over all 336 cells (7 models × 48 conditions) at FCz / mtrf, this asserts:

- `max(400, soa_ms) − max(time_ms)` is **19.0–19.25 ms** for the five whisper models (whisper-large
  included) and **38.875–39.25 ms** for the two wav2vec2 models.
- every instance has `max(time_ms) ≥ 360`; minimum **381** whisper / **361** wav2vec2.
- **zero** cells with the trough on the final sample; **zero** truncated recovery searches.
- the 24 unchanged conditions score bit-identically against
  `outputs/results_24freq_7models/mmn_s7_roi.csv`, in all seven models — and the 24 changed ones
  actually moved, so a no-op run can't pass silently.

> **Correction to the original brief.** Step 7's first assertion was written as
> `soa_ms − max(time_ms)` staying at 19.0–19.25 / 38.875–39.25. That holds only where SOA ≥ 400.
> The epoch ends one edge before the **reserved tail**, which is `max(400, SOA)` — so post-fix a
> SOA-200 condition has `max(time_ms) = 380.875` and `soa − max(time_ms) = −180.875` **by design**.
> The brief's own companion expectations (381 whisper / 361 wav2vec2 = `400 − 19.125` / `400 −
> 39.125`) are only consistent under the reserved-tail reading, which is what the script uses.
> Verified against the committed baselines, where `max_t = soa − 19.125` for all 48 whisper
> conditions.

### Then report

The published baseline to beat:

| arm | n_agree | mean_uv |
|---|---|---|
| literature top 10, as committed | 3.80 | −1.5643 |
| novel top 10 (unchanged by this work) | 6.00 | −2.2089 |

`mean_uv` is the mean `trough_uv` over the S7-agreeing models only, not all six. Five of the
literature top 10 are in the changed set: ranks 3 `method_53_counter`, 4 `method_10_counter`,
7 `method_60_counter`, 8 `method_19_counter`, 9 `method_18_counter`.

Expect literature `n_agree` to **rise** — the truncation biased toward S2 = False (12 cells forced
False outright), so restoring the window can only add passes or leave them. Watch
`method_17/18/19_counter` on **wav2vec2-large**, whose troughs sat on the final sample at
−15.6 / −15.9 µV — an order of magnitude deeper than anything else in the screen — and were scored
S2 = False by the empty-slice path. If those now pass, the literature arm moves materially and the
memo's headline comparison must be **restated**, not just re-run.

Note whisper-large is not in the six-model `SEARCH_MODELS`/LAYER set used for the memo's top-10
ranking, so redoing it changes the 7-model screen's tables but not the memo's top-10 arithmetic.

---

## Out of scope

Updating Tables 6/7 and the figures in `aux/analysis_novel_search/novel_stimulus_search_results.md`.
Report the new numbers and stop — re-rendering those figures is now a two-format job (every one has
a tracked SVG twin as of 2026-08-02), so it is a larger diff than it looks and belongs in its own
session. When it happens:

- The literature-vs-novel comparison figures use a **−120 to 360 ms** window, not 460. At that right
  edge every literature *and* every novel instance is complete, so there are no ragged edges and no
  per-trace clipping — one shared axis for both arms. 360 is not a compromise: it is exactly the
  span the criteria read. Use **360, not 361** — `soa − max(time_ms)` measures up to 39.25 ms on
  wav2vec2, so a 361 ms limit would leave some traces fractionally short.
- Post-fix minimum tails (381 / 361) leave 21 ms and 0.75 ms of margin on that axis. **Nothing may
  widen it without re-running the audio**, and the 400 ms floor must not be quietly raised to 500 or
  620 to chase a 460 ms window — the 400 ms floor plus a 360 ms axis is the chosen combination.
- Existing novel-only figures keep whatever window they were published with; the novel arm has never
  truncated and does not need re-plotting.
- Do not call `fig.savefig` directly. Import `savefig_both(fig, out_dir, name, svg_dir=None)` from
  `aux/analysis_novel_search/plots/novel_search_plots.py` and call
  `configure_svg_output(out_dir, svg_dir, no_svg)` once from `main()` first.

## The novel grid is untouched

One fixed SOA of 580 ms (`build_novel_grid_csv.py`: tone 80 + ISI 500) gives tails of 561 / 541 ms,
so `max(400, 580) == 580` and the layout is provably unchanged — `(K, leftover)` is `(50, 500)`
whisper / `(16, 220)` wav2vec2 with and without the floor, and this is pinned by a unit test.
**Do not re-run or regenerate any part of it.** It costs ~1 TB and buys nothing.
