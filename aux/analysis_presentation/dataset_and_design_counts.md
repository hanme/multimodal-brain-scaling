# Appendix source note — D2 dataset parameters and MMN stimulus-design counts

Produced 2026-08-17. Pure extraction/verification, no statistics. Machine-readable companion:
`aux/analysis_presentation/dataset_and_design_counts.csv` (quantity, value, unit, source_file,
source_field).

---

## A. D2 — the speech EEG the mTRF mapping was trained on

**Every MMN result in the deck uses D2 and only D2.** D1 (Broderick) and D3 (D1∪D2) are not the
mapping training set; D1's fronto-central channels fail the noise-ceiling floor (Cz r≈0.16), which
is why D2 was chosen (FCz r≈0.99).

**Primary source (added 2026-08-17).** <https://zenodo.org/records/7775260> — "EEG Dataset for
'Cortical Tracking of Surprisal during Continuous Speech Comprehension'", Weissbart, Kandylaki &
Reichenbach, v2.0.0 (2022-09-16), CC-BY-4.0, DOI `10.5281/zenodo.7775260`. Source publication:
Weissbart H, Kandylaki KD, Reichenbach T., *J Cogn Neurosci* 32(1):155-166, 2020,
DOI `10.1162/jocn_a_01467`. The Zenodo record distributes **unprocessed** EEG time-aligned to the
audiobook onsets — the paper's own 125 Hz / 0.3 Hz / ICA preprocessing is *not* what our pipeline
applies, so do not quote it as ours.

| quantity | value | source |
|---|---|---|
| dataset | Weissbart "Cortical Surprisal" naturalistic speech EEG | Zenodo 7775260 |
| subjects | **13** | `surprisal_30s.h5` `attrs.n_subjects`; Zenodo `P00.h5`…`P12.h5`; paper, Participants |
| demographics | aged 25 ± 3 y, 6 women, all right-handed native English speakers | paper, Participants |
| storage in our pipeline | **group-averaged** (`subjects = ['group']`); no per-subject arrays | `format_eeg_hdf5.py:397` schema; `XX_handover_for_Sophie.md:112` |
| corpus | 3 public-domain short stories, single male narrator (librivox.org); text from Project Gutenberg ebook 32846 | paper, Experimental Design |
| stories → stems | *An Undergraduate's Aunt* (F. Anstey) = `AUNP01–08`; *My Brother Henry* (J. M. Barrie) = `BROP01–03`; *Gilray's Flower Pot* (J. M. Barrie) = `FLOP01–04` | paper + `align_data.py` `story_part_names` |
| parts | **15**, each 2.6 ± 0.43 min | paper, Experimental Design |
| total per subject | **40 min** (paper); **2382 s = 39.7 min** from the exact HDF5 file size | paper; Zenodo file sizes |
| coverage | all 13 subjects heard all 15 parts; block order counterbalanced per participant | `stimulus_order.csv` — 13 rows, each a full permutation of 0–14 |
| task | passive listening; 30 multiple-choice comprehension questions in total, after each part | paper, Experimental Design |
| acquisition | 64 active electrodes (actiCAP) + actiCHamp amplifier, BrainProducts; **1 kHz**; **left-earlobe reference**; audio recorded in parallel via StimTrak for alignment | paper, EEG Acquisition; `align_data.py` `srate = 1000` |
| channels | **63** — `align_data.py` does `raw.drop_channels('Sound')`, so the StimTrak channel occupies one of the 64 slots | `align_data.py`; Zenodo file sizes; `project_plan_20260611.md:715` |
| audio | 16 kHz wav (`audiobooks.zip`, 62.0 MB) | `preprocess.py` `assert fs == 16000` |
| our resampling | 50 Hz (20 ms bins), 0.5 Hz high-pass | `scripts/eeg_targets.py:20-21`; `eeg_mapping/*.json` `highpass_hz` |
| windows | 30 s duration / 10 s stride | overview §1.4 |
| train / test | **157 / 43** windows (12 / 3 audiobook parts) | `surprisal_30s.h5` `train/stimulus_ids`, `test/stimulus_ids` (read 2026-08-17) |
| NC floor | Pearson r > 0.2 → **47 of 63** electrodes retained (16 dropped) | `surprisal_30s.h5` `noise_ceilings/group`; `eeg_mapping/*__electrodes__D2.json` `targets` |
| FCz noise ceiling | **r = 0.9924** (max of the 47; min retained 0.2074) | `eeg_mapping/whisper-tiny__electrodes__D2.json` `nc_r` |
| parcels | 5 (frontal, central, temporal, parietal, occipital) | `eeg_mapping/*__parcels__D2.json` `targets` |
| CV | grouped 4-fold **by audiobook part** (30 s windows at 10 s stride overlap by 20 s) | `n_folds`; `scripts/eeg_targets.py:46` |
| mTRF lags | 0–800 ms | `lag_max_ms` |
| PCA | `pca_var = None` for every model | `pca_var` |

### The 63-channel count, settled

Previously a single unsourced claim in `project_plan_20260611.md:715`. Now corroborated three ways:

1. `align_data.py` reads the BrainVision file and calls `raw.drop_channels('Sound')` before saving,
   so the distributed array excludes the StimTrak channel.
2. The Zenodo participant files come in exactly **two** sizes, 1200596308 B and 1200596812 B. The
   difference is **504 B = 63 channels × 8 B (float64)** — one extra time sample. At 64 channels the
   quantum would be 512 B. (The ±1 sample comes from `int(start_time*1000 + length)` truncation in
   `align_eeg`.)
3. It matches the pre-existing repo claim.

So the cap carried **64 active electrodes**, one slot fed by StimTrak, leaving **63 scalp channels**
in the HDF5 — of which 47 survive our NC floor.

### Subject-hours — settled, three ways

Read from the cluster on 2026-08-17 with `aux/analysis_presentation/probe_d2_cluster.py`. The h5's
own attributes confirm `n_subjects 13`, `window_duration_s 30.0`, `window_stride_s 10.0`,
`target_sr 50`, `subjects ['group']`, and `test_parts ['AUNP02','BROP02','BROP03']`.

Windows tile each part from t=0 at a 10 s stride while `start + 30 s ≤ part duration`, so `n`
windows cover `10·(n−1) + 30` s:

```
train : 157 windows over 12 parts = 10×157 + 20×12 = 1810 s = 30.2 min
test  :  43 windows over  3 parts = 10× 43 + 20× 3 =  490 s =  8.2 min
        (AUNP02 13, BROP02 16, BROP03 14)
total : 200 windows over 15 parts                  = 2300 s = 38.3 min per subject
subject-hours entering the mTRF = 13 × 2300/3600 = 8.31 h
```

**The tiling assumption is closed.** For all 15 parts, the window count predicted from the wav
duration equals the count in the h5 — so `format_eeg_hdf5_surprisal.py` tiles exactly like
`format_eeg_hdf5.py`. Part durations run 109.79 s (FLOP03) to 196.97 s (FLOP04); every untiled tail
falls in [0.00, 9.79) s as the rule requires, AUNP08 landing on exactly 180.00 s with zero tail.

```
total audio = 2382.1 s = 39.70 min per subject   (sum of the 15 wavs)
full-recording subject-hours = 13 × 2382.1/3600 = 8.60 h
```

Three independent routes agree on that duration: the wavs (2382.1 s), the Zenodo HDF5 file size
(1200596308 B / (63 ch × 8 B) = 2382.1 s), and the paper's "total length of the stories was 40 min".
The 200 windows cover **96.6 %** of it; the 82 s shortfall is the per-part tail, 5.5 s on average.

**Quote 8.3 subject-hours** as what the mapping is fit on, or **8.6** as what was recorded — say
which. Nothing about the duration is inferred any more.

### Which channels survive the noise-ceiling floor

The file carries **69 `rois`** = 63 real electrodes + 6 pseudo-channels (5 `*_cluster` averages plus
`whole_brain`). The mapping drops the pseudo-channels via `eeg_targets.NON_ELECTRODE`, so:

```
63 electrodes − 16 below the floor = 47 mapping targets   ← matches eeg_mapping/*.json exactly
```

**Dropped (16), NC r ≤ 0.2:** `AF4 .002, AF7 .002, AFz .001, C1 .001, CP3 .196, CP4 .001, CP5 .001,
CP6 .002, F5 .002, FC1 .001, FC5 .133, Fp1 .002, Oz .169, P8 .085, POz .001, T8 .132`.

Two things follow. First, **11 of the 16 sit at r ≈ 0.001–0.002 — functionally exact zero**, the same
signature that disqualified D1 Broderick ("several at exact 0.000, likely a montage artifact",
overview §1.4). It does not undermine D2, whose fronto-central channels are the best in the file
(FCz 0.992, F2 0.984, Cz 0.777, Fz 0.658), but it is worth one honest sentence: a sixth of D2's
montage carries no usable cross-subject signal.

Second, it **explains the parcel memberships** quoted in `XX_handover_for_Sophie.md:116` — temporal
is `T7` alone because T8 is dropped, parietal omits P8, occipital omits Oz. And it explains the
fronto-central ROI actually used by the MMN read-out, `Fz,FCz,Cz,FC2,F1,F2` (the h5 `fc_roi` attr):
FC1 is the seventh nominal member and fails the floor at r = 0.001.

### Committed mTRF layer per model (electrodes, D2)

| model | layer | mean test r | FCz test r | source |
|---|---|---|---|---|
| whisper-tiny | `blocks.0` | +0.212 | +0.139 | `eeg_mapping/whisper-tiny__electrodes__D2.json` |
| whisper-base | `blocks.0` | +0.173 | +0.145 | `eeg_mapping/whisper-base__electrodes__D2.json` |
| whisper-small | `blocks.1` | +0.118 | +0.118 | `eeg_mapping/whisper-small__electrodes__D2.json` |
| whisper-medium | `blocks.12` | +0.079 | +0.125 | `eeg_mapping/whisper-medium__electrodes__D2.json` |
| whisper-large | `blocks.21` | — | — | JSON still absent locally; layer from `slurm_insilico_mmn_electrodes.sh:48` and the committed h5 filename |
| wav2vec2-medium | `encoder.layers.2` | +0.223 | +0.139 | `eeg_mapping/wav2vec2-medium__electrodes__D2.json` |
| wav2vec2-large | `encoder.layers.12` | +0.244 | +0.133 | `eeg_mapping/wav2vec2-large__electrodes__D2.json` |

The two wav2vec2 JSONs landed in this checkout on 2026-08-17 and confirm the layers the SLURM map
and the prediction filenames already implied, plus `neural = surprisal_10s.h5`, `pca_var = None`,
`lag_max_ms = 800`, `nc_r_threshold = 0.2`, 47 electrode targets — i.e. the fit-protocol caveat
below, read straight off the sweep rather than inferred. Their parcel counterparts agree
(`encoder.layers.2` / `encoder.layers.12`, mean test r +0.200 / +0.212). **wav2vec2's higher mean
test r is not evidence it maps better** — it was fit on 10 s windows against a different EEG file.

**Fit-protocol caveat.** All five whisper models were mapped on **30 s windows / 10 s stride**
against `surprisal_30s.h5`; both wav2vec2 models on **10 s / 10 s** against `surprisal_10s.h5`
(features extracted at 10 s/5 s, the odd-offset windows simply never matched). `pca_var = None`
throughout, so the mapping window is the only fit-protocol difference — wav2vec2's test r is not
strictly comparable to whisper's.

---

## B. Stimulus-design counts

### What one MMN value is

One MMN value = one model × one condition. A **condition** is one frequency pair in one direction
(`method_XX` or `method_XX_counter`). Within a condition:

```
1 standard wav  +  15 deviant wavs  =  16 wavs
15 = N ∈ {3, 5, 7}  ×  variation ∈ {1..5}
MMN = mean(15 deviant predictions) − standard prediction
      at FCz, baseline = [−3×SOA, 0) ms before final-tone onset,
      most-negative point in 100–240 ms
```

Verified: `n_deviants = 15` on all 48 literature conditions × 7 models and all 254 Phase-2
conditions × 6 models; `n_deviants = 1` on all 1806 Phase-1 conditions × 6 models (Phase 1 only
synthesized the N7/var1 deviant).

**Oddball probability, in role terms.** In a deviant clip the *frequent background* tone is the
row's `deviant_freq` and the *rare oddball* is the row's `standard_freq` — the generator's variable
names are the inverse of the roles (`00aa_generate_audio_stimuli.py:307-350`). The critical suffix
is `[oddball, background × N, oddball]`, and each prefix slot is independently the oddball with
probability **1/(N+1)**:

| N | oddball probability |
|---|---|
| 3 | 25.0 % |
| 5 | 16.7 % |
| 7 | 12.5 % |

The `p_deviant_pc` column of the metadata CSV is the *source paper's* reported probability and is
**not read** by the generator or by `insilico_mmn.py` (grep: zero hits) — do not quote it as the
realised design.

### ⚠ The eliciting tone is physically identical in standard and deviant

Measured by FFT on the committed audio, not inferred:

| file | last 6 tones (Hz) |
|---|---|
| `data/audio_stimuli/method_09_standard.wav` | 597 597 597 597 597 597 |
| `data/audio_stimuli/method_09_N7_var1_deviant.wav` | 1006 1006 1006 1006 1006 **597** |
| `data/audio_stimuli/method_37_standard.wav` | 1006 ×6 |
| `data/audio_stimuli/method_37_N7_var1_deviant.wav` | 1043 1043 1043 1043 1043 **1006** |
| `audio_presentation_samples/.../method_1766_standard.wav` | 1748 ×6 |
| `audio_presentation_samples/.../method_1766_N7_var1_deviant.wav` | 6985 ×5, **1748** |

So `deviant − standard` at the final tone contrasts two physically identical tones differing only
in preceding context. That is the **physically-controlled** design, which the 2026-06-21 relabelling
in overview §16.1 calls **Definition 2** and asserts is "not currently used". Overview §1.6 and
§16.1 both state the opposite ("the deviant train's *last* tone differs in frequency from the
standard's"). The audio and `generate_deviant_sequence` agree with each other and disagree with the
prose; the generator's role logic is unchanged since its first commit (`021574c`). **Do not describe
the design on the slide as a classic oddball with a frequency-different eliciting tone.**

### Literature set

| quantity | value | source |
|---|---|---|
| frequency methods | **24** (of 50 rows; the rest are Duration/Intensity/Frequency_Long_ISI) | `literature_frequency_intensity_duration_metadata.csv` `change_type=="Frequency"` |
| method ids | 9,10,12,17,18,19,20,21,27,28,29,30,31,32,33,37,43,44,53,55,60,72,74,75 | same |
| conditions | **48** = 24 × {regular, counter} | `outputs/results_soafix_full/mmn_s7_roi.csv`, distinct `method` |
| standard frequencies | 600, 633, 1000 Hz | CSV |
| deviant frequencies | 700, 1000, 1050, 1064, 1122, 1200, 1500, 1850, 2000 Hz | CSV |
| deviance | 5 % (1000→1050) to 100 % (1000→2000) | derived |
| SOA | 200–1000 ms (12 distinct values) | `standard_soa` |
| tone duration | 50–200 ms | `standard_dur` |
| intensity | 75–85 dB | `standard_int` |
| tones per 30 s clip | 29–148 | `compute_tone_slots(30000, dur, isi, 400.0)` |
| models | **7** (5 whisper + 2 wav2vec2) | results CSV |
| model × condition cells | **336** with whisper-large, **288** without | 48 × 7 / 48 × 6 |

Row-count check: 21168 = 7 models × 48 conditions × 9 ROIs × 7 µV thresholds. ✓
FCz-only at X=0.75: 336 rows, exactly 48 per model. ✓

The current literature numbers come from `outputs/results_soafix_full/`, i.e. the set regenerated
with `trailing_floor_ms = 400` so the 100–240 ms scoring window plus S2's recovery search fits
inside every epoch (`aux/handoff_soafix_trailing_floor.md`).

### Novel grid

| quantity | value | source |
|---|---|---|
| frequencies | **43** rungs | `grid_index.csv`, distinct `f_low ∪ f_high` |
| step | **1.500 semitones** (12/1.5 = 8, so octaves land on the grid) | `build_novel_grid_csv.py` `SEMITONE_STEP` |
| range | 200 – 7611 Hz (under the 8 kHz Nyquist of the 16 kHz models) | `grid_index.csv` |
| deviance per step | 8.8 % (min); 3705.5 % across the full span | `grid_index.csv` `pct_deviance` |
| unordered pairs | **903** = 43×42/2 | `grid_index.csv` row count |
| directions | 2 (regular + counter); diagonal excluded | `build_novel_grid_csv.py` |
| **direction-instances** | **1806** = 903 × 2 ✓ | `phase1_ranked_directions.csv` row count = 1806, 903 distinct `pair_id`, 903 regular + 903 counter |
| fixed tone params | 80 ms tone, 500 ms ISI, **580 ms SOA**, 80 dB, 50 tones/30 s clip | `build_novel_grid_csv.py`; `compute_tone_slots` |
| Phase-1 models | **6** (whisper-large excluded — alone it was 51 % of the literature screen's cost) | `phase1_mmn_s7_roi.csv` |
| Phase-1 deviants/condition | **1** (N7/var1 only) | h5 `n_deviants` |
| Phase-1 cells | **10836** = 1806 × 6 | derived; CSV rows 530964 = 6 × 1806 × 7 ROIs × 7 thresholds ✓ |

### Phase 2

**Selection rule** (`novel_search_common.py`, `MIN_AGREE_PHASE2 = 5`): a *pair* enters Phase 2 iff
**at least one of its two directions** has `n_agree ≥ 5`, where `n_agree` = how many of the 6 models
show S7 at FCz (S7 = S2 ∧ trough ≤ −0.75 µV). Both directions of a selected pair are then carried,
including sub-threshold reversals — the direction asymmetry is the frequency-preference artifact the
counterbalancing exists to detect.

Recomputed from `phase1_ranked_directions.csv`:

```
direction-instances by n_agree : 0→35  1→249  2→562  3→540  4→289  5→114  6→17
instances at n_agree ≥ 5       : 131  (17 at 6/6 + 114 at 5/6)
distinct pairs among them      : 127  (17 topping out at 6/6 + 110 at 5/6)
sub-threshold reversals carried: 254 − 131 = 123
Phase-2 conditions             : 127 × 2 = 254
```

Cross-checks: the derived 127-pair set is **identical** to `phase2_selected_pairs.csv` (127 rows);
`outputs/novel_methods_phase2.txt` has 254 unique lines over exactly those 127 base ids;
`phase2_mmn_s7_roi.csv` has 254 distinct conditions and 74676 rows = 6 × 254 × 7 × 7. ✓

| quantity | value |
|---|---|
| Phase-2 conditions | **254** |
| Phase-2 deviants/condition | **15** (full grid) |
| Phase-2 models | **6** |
| model × condition cells | **1524** = 254 × 6 (as run). 1778 = 254 × 7 if whisper-large were added — it was not. |

---

## COULD NOT VERIFY

Revised 2026-08-17 after the Zenodo record. Nothing below is filled with a plausible value.

**Closed by <https://zenodo.org/records/7775260> and the source paper:** exact recording duration
per subject (40 min / 2382 s); whether all 13 subjects heard all 15 parts (yes); the 63-channel
montage (now corroborated three ways); the audiobook titles and narrator; native sampling rate,
reference, amplifier and cap; participant demographics; the task; dataset DOI and licence.

**Closed 2026-08-17 by a cluster read** (`aux/analysis_presentation/probe_d2_cluster.py`, output
pasted into this session): the h5's own attributes; per-part window counts for both splits including
the 3 test parts; the tiling-rule equivalence with `format_eeg_hdf5.py`; the 16 sub-floor electrodes;
and `n_subjects = 13` as the formatter recorded it. Also closed the same day: the wav2vec2
`chosen_layer` JSONs, which appeared in the local checkout.

**Still open:**

1. **`chosen_layer` JSON for whisper-large.** `outputs/results/eeg_mapping/` holds the 4 small
   whisper models and both wav2vec2 models × {parcels, electrodes}, but not whisper-large. Its
   `blocks.21` is corroborated twice (SLURM layer map + the layer embedded in the committed
   prediction h5 filename); the JSON `chosen_layer`, `cv_score_chosen` and `test_r_chosen` were not
   read, so whisper-large has no test r in the layer table. Command B below closes it.
2. **Held-out test r for whisper-large** — same reason.
3. **The audiobook parts are not equally split across stories in the test set.** `test_parts` is
   `AUNP02, BROP02, BROP03` — two of the three held-out parts come from *My Brother Henry*, and
   *Gilray's Flower Pot* contributes none. Whether that was deliberate is not recorded anywhere in
   this checkout. It is a property of the split, not an error, but do not describe the test set as
   "one part per audiobook".

### Cluster commands that close the open items

Run from `/work/upschrimpf1/sigfstea/multimodal-brain-scaling` after `source env.sh`. All of it is
read-only.

**A. The one that matters — `aux/analysis_presentation/probe_d2_cluster.py`.** Closes items 1, 2, 3,
4 and 7 in a single read. It prints the h5's own attributes, per-part window counts for *both*
splits, and — the real prize — predicts each part's window count from its wav duration and diffs it
against the h5, which is what actually decides whether `format_eeg_hdf5_surprisal.py` tiles like
`format_eeg_hdf5.py`. It also lists which of the 63 channels clear the NC floor and which 16 do not.

```bash
python aux/analysis_presentation/probe_d2_cluster.py                              # 30 s file
python aux/analysis_presentation/probe_d2_cluster.py --h5 outputs/neural_data/surprisal_10s.h5
```

Defaults: `--h5 outputs/neural_data/surprisal_30s.h5`, `--audio_dir` the recovered stimuli root
`.../multimodal-brain-scaling-temporal-analysis/data/cortical_suprisal_dataset/audiobooks`.
Smoke-tested locally against a synthetic D2-shaped fixture; the mismatch branch was verified to fire.

**Already run, 2026-08-17.** It returned exactly the expected values: `n_subjects 13`,
`window_duration_s 30.0`, `window_stride_s 10.0`, `target_sr 50`; 157 windows over 12 parts and 43
over 3 (AUNP02 13, BROP02 16, BROP03 14); all 15 rows in block 3 matching, tails in [0.00, 9.79) s,
total 2382.1 s; 47 of 63 electrodes surviving with FCz = 0.992. Re-run it after any re-format of the
D2 HDF5 — block 3 is the regression test for the tiling rule.

**B. Items 5 and 6 - the missing whisper-large layer JSON.** (The wav2vec2 pair landed locally on
2026-08-17; the loop still covers all three so the run doubles as a cross-check that the cluster
copies match.)

```bash
for m in whisper-large wav2vec2-medium wav2vec2-large; do
  for lvl in electrodes parcels; do
    python - "$m" "$lvl" <<'EOF'
import json, sys, numpy as np
m, lvl = sys.argv[1], sys.argv[2]
d = json.load(open(f"outputs/results/eeg_mapping/{m}__{lvl}__D2.json"))
print(f"{m:16s} {lvl:10s} {d['chosen_layer']:20s} "
      f"cv={d['cv_score_chosen']:.3f} test_r={np.mean(d['test_r_chosen']):+.3f} "
      f"pca={d['pca_var']} nc={d['nc_r_threshold']} lag={d['lag_max_ms']}")
EOF
  done
done
```

This must print `blocks.21`, `encoder.layers.2`, `encoder.layers.12` - the layers the committed
prediction HDF5s were produced at. Anything else means the screen and the sweep disagree. The two
wav2vec2 rows should reproduce cv +0.220 / +0.237 and mean test r +0.223 / +0.244 exactly.

**C. Read the D2 formatter itself** (belt-and-braces on item 3, in case block 3 above reports a
mismatch and you need to see why):

```bash
sed -n '1,140p' ../multimodal-brain-scaling-temporal-analysis/src/mbs/data_prep/format_eeg_hdf5_surprisal.py
```

**D. Ground-truth the durations straight off the audio** (independent of any h5):

```bash
python - <<'EOF'
import glob, os, wave
d = ("/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis"
     "/data/cortical_suprisal_dataset/audiobooks")
paths = sorted(glob.glob(f"{d}/*.wav"))
tot = 0.0
for p in paths:
    with wave.open(p, "rb") as w:
        s = w.getnframes() / w.getframerate()
        print(f"{os.path.basename(p):14s} {s:8.2f} s  ({s/60:.2f} min)  sr={w.getframerate()}")
    tot += s
print(f"{len(paths)} parts, total {tot:.1f} s = {tot/60:.2f} min")
print(f"subject-hours = 13 x {tot:.0f}/3600 = {13*tot/3600:.2f} h")
EOF
```

Expect 15 parts, about 2382 s total, 16 kHz - matching the paper's "40 min" and the Zenodo
file-size derivation. If the wavs are 16 kHz here, that also confirms the stimulus IDs' 160000-sample
stride is exactly 10 s.

**E. Raw-dataset cross-check** (only if A disagrees with the Zenodo numbers) - the 63-channel count
and 1 kHz rate straight from a participant file:

```bash
python - <<'EOF'
import h5py
p = ("/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis"
     "/data/cortical_suprisal_dataset/P00.h5")
with h5py.File(p, "r") as f:
    print("srate:", f["srate"][()])
    print("n ch_names:", len(f["ch_names"]), [c.decode() for c in f["ch_names"][:8]], "...")
    tot = 0
    for k in f["data"]:
        n = f["data"][k].shape[1]
        tot += n
        print(f"  {k:8s} {f['data'][k].shape}  {n/1000:8.2f} s")
    print(f"total {tot/1000:.1f} s = {tot/60000:.2f} min")
EOF
```

## Reproducibility

**Files read** (all read-only):
`aux/sophies_repository_overview.md` §1.4–1.7, §5, §16, §17;
`aux/XX_handover_for_Sophie.md`; `aux/project_plan_20260611.md`;
`aux/handoff_enable_large_wav2vec2_models.md`; `aux/handoff_24freq_7models_screen.md`;
`aux/handoff_soafix_trailing_floor.md`;
`src/mbs/data_prep/format_eeg_hdf5.py`; `scripts/eeg_targets.py`;
`scripts/00aa_generate_audio_stimuli.py`; `scripts/insilico_mmn.py`;
`scripts/build_novel_grid_csv.py`; `scripts/novel_search_common.py`;
`scripts/slurm_insilico_mmn.sh`; `scripts/slurm_insilico_mmn_electrodes.sh`;
`tests/test_attn_probe_temporal.py`;
`data/metadata/literature_frequency_intensity_duration_metadata.csv`;
`data/metadata/novel_grid_frequency_metadata.csv`;
`outputs/results/eeg_mapping/*__{electrodes,parcels}__D2.json`;
`outputs/results_soafix_full/mmn_s7_roi.csv`;
`outputs/results_novel_search/{grid_index,phase1_mmn_s7_roi,phase1_ranked_directions,phase2_selected_pairs,phase2_mmn_s7_roi}.csv`;
`outputs/novel_methods_phase2.txt`;
`outputs/insilico_mmn_predictions_{soafix,novel,novel_phase2}/*/electrode_predictions__*.h5`;
`data/audio_stimuli/method_{09,37}_{standard,N{3,5,7}_var1_deviant}.wav`;
`audio_presentation_samples/phase_2_top_10/method_1766/*.wav`.

**External sources read (2026-08-17):** <https://zenodo.org/records/7775260> (record page, API
metadata, exact file sizes) and its files `stimulus_order.csv`, `align_data.py`, `preprocess.py`;
the source paper PDF (<https://www.neurotech.tf.fau.eu/files/2021/08/weissbart_jocn_2019.pdf>),
Methods sections *Participants*, *Experimental Design*, *EEG Acquisition and Preprocessing*.

**Files created:** `aux/analysis_presentation/dataset_and_design_counts.csv`, this file.

**Local environment:** `conda activate mbs-env`; where a script is imported,
`PYTHONPATH="$(pwd)/src:$(pwd)/scripts"` (mbs-env's editable install points at a sibling checkout).

**Key recomputations** — each is a one-liner over a committed artefact and is embedded in the
relevant section above: the 48/1806/254 condition counts and their row-count identities; the
n_agree tier table and the 127-pair reproduction of `phase2_selected_pairs.csv`; the
`n_deviants` sweep over all three prediction roots; `compute_tone_slots` over all 24 literature
rows at `trailing_floor_ms=400`; the FFT final-tone check on the committed wavs.
