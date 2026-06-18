# Handover: Auditory EEG Encoding Models — Sophie

_Last updated 2026-06-18. Full technical log (scaling, env/GPU, superseded methods, repo layout) is
in `aux/project_plan_20260611.md`; MMN design in §17 there._

**The task.** Feed MMN tone stimuli through a Whisper audio model, apply a trained model→EEG
mapping, and get a predicted EEG time course at the **parcel** level (5 coarse 10-20 parcels) or the
**electrode** level (47 NC-passing electrodes) — then run your deviant−standard MMN on it. The model
and the per-model layer are fixed by the layer-selection analysis (§3). Everything below §1 is brief
background.

---

## 1. Run it

**Prerequisite (per model, once):** extract the MMN stimuli into that model's `delta_T` features —
`sbatch scripts/slurm_mmn_extract.sh` (set `MODEL_ID` + method list) → `outputs/features/mmn-<method>-delta-t/`.

> ⚠️ **Choose your own frequency pairs from the literature — and only the *right* kind of study.**
> The 8 pairs currently present (`method_09, method_12(+_counter), method_37, method_44(+_counter),
> method_55(+_counter)`) are **our placeholders**, not a vetted set. Replace them with standard/deviant
> pairs taken from MMN papers whose **paradigm matches ours = Definition 2 (physically controlled)**:
> the **last/eliciting tone is physically identical** in the standard and deviant, and the deviance
> lives in the *preceding context* (see §3). **Do NOT** lift pairs from the classic early oddball
> studies (e.g. Sams 1985, Tiitinen 1994) where the standard and deviant sequences **end in different
> tones** (Definition 1) — those confound surprise with the plain acoustic difference of the probe
> tone, so they are incompatible with our subtraction. Good template: Weber 2022 (counterbalanced
> roving oddball). Pick the pairs, generate each as a `mmn-<name>` stimulus dir, extract its features,
> then pass the names to `--methods`. (Screening guidance: `aux/mmn_screening_plan.md`; design
> rationale: `project_plan §17`.)

**Fixed layer** (from the D2 sweep; grab via
`python -c "import json;print(json.load(open('outputs/results/eeg_mapping/whisper-small__parcels__D2.json'))['chosen_layer'])"`):

| model | parcels | electrodes |
|---|---|---|
| whisper-tiny | `blocks.1` | `blocks.0` |
| whisper-base | `blocks.4` | `blocks.4` |
| whisper-small | `blocks.10` | `blocks.10` |
| whisper-medium | `blocks.22` | `blocks.22` |

**Method A — mTRF** (recommended; ridge re-fit on D2 at `--layer`, amplitude-preserving):
```bash
# parcels:
python scripts/insilico_mmn.py \
  --train_features <MODEL D2 feats>/merged --train_neural outputs/neural_data/surprisal_30s.h5 \
  --mmn_features_root outputs/features --layer blocks.10 --lag_max_ms 800 --methods all
# electrodes (topographic MMN + present/absent verdict):
python scripts/insilico_mmn_electrodes.py \
  --train_features <MODEL D2 feats>/merged --train_neural outputs/neural_data/surprisal_30s.h5 \
  --layer blocks.10 --lag_max_ms 800 --methods all
```
Use `--lag_max_ms 800` to match the sweep that chose the layer (driver default is 500).

**Method B — attention encoder** (MSE-trained, amplitude-calibrated; layer fixed by the checkpoint):
```bash
python scripts/insilico_mmn_attn.py \
  --checkpoint outputs/results/whisper-small-probe-group-d2-parcels/model__blocks.10.pt \
  --mmn_features_root outputs/features/whisper-small-mmn --method method_09 \
  --features_dir <MODEL D2 feats>/merged --neural outputs/neural_data/surprisal_30s.h5
# electrodes: --checkpoint .../whisper-small-probe-group-d2-electrodes/model__blocks.10.pt
```
(`insilico_mmn_attn.py` reuses the parcel-grid plotter — an electrode checkpoint gives correct
prediction *arrays* but not an electrode topography plot.)

Outputs per run: figures in `outputs/figures/insilico_mmn*`, and raw prediction arrays in
`outputs/insilico_mmn_predictions*/predictions__<layer>*.h5`.

---

## 2. The MMN analysis

Each prediction h5 holds, per method: `time_ms` (0 = final/eliciting-tone onset), `standard
[n_t, T]`, `deviant_mean [n_t, T]`, `deviants [n_dev, n_t, T]` (T = 5 parcels or 47 electrodes),
plus `parcels`/`parcel_members`/`parcel_nc_r`.

```python
import h5py, numpy as np
with h5py.File("outputs/insilico_mmn_predictions_small/predictions__blocks.10__attn.h5") as h:
    names = [p.decode() for p in h["parcels"][:]]
    g = h["method_09"]; t = g["time_ms"][:]
    mmn = g["deviant_mean"][:] - g["standard"][:]      # deviant − standard
mmn -= mmn[(t >= -150) & (t < 0)].mean(0, keepdims=True)   # baseline-correct (pre-onset)
neg_peak = mmn[(t >= 100) & (t <= 250)].min(0)             # MMN-band negative peak per target
for n, v in zip(names, neg_peak): print(f"{n:9s} {v:+.3f}")
```

**Amplitude.** Method A and the **new** Method B `…-d2-{parcels,electrodes}` checkpoints are
amplitude-calibrated → read **magnitude** (for B, first invert with
`checkpoint.predictions_to_units`). The **older** `…-d2-mmn` Method-B checkpoint used 1−Pearson loss
→ amplitude is arbitrary, read **sign/shape** only.

---

## 3. Background (minimal)

- **Mapping.** Trained on D2 "Weissbart Cortical Surprisal" speech EEG (`surprisal_30s.h5`, 13 subj,
  group-averaged, 157 train / 43 test 30 s windows). Two swappable methods: **A** = closed-form
  lagged ridge (`evaluate_features_mtrf.py`), **B** = gradient-trained attention encoder
  (`evaluate_features_attn_probe_temporal.py`, reusable I/O in `attn_probe/checkpoint.py`).
- **Targets.** Parcels = raw channel averages after an NC floor (cross-subject r > 0.2): frontal
  (Fz,F3,F4,FCz), central (Cz,C3,C4), temporal (T7), parietal (Pz,P3,P4,P7), occipital (O1,O2).
  Electrodes = the 47 individual channels passing r > 0.2. Same definitions for both methods and for
  the mTRF/encoder sweeps (`scripts/eeg_targets.py`).
- **Why D2 not D1 (Broderick).** D1's fronto-central electrodes — where the MMN lives — fail the NC
  floor (Cz r≈0.16, several at exact 0.000, likely a montage artifact); D2 is clean there (FCz r≈0.99).
- **MMN design = Definition 2 (physically controlled).** Read time-locked to the last (eliciting)
  tone; the final tone is *physically identical* in standard and deviant (method_09: deviance is in
  the preceding context, 1000→600 Hz), so deviant−standard isolates surprise. Matches Weber 2022, not
  the classic Sams/Tiitinen oddball. (`project_plan §17`.)
- **Layer-selection figures.** mTRF: `outputs/figures/eeg_mapping/` (`layer_selection*`,
  `test_fit_quality*`, CV-chosen layer circled). Encoder (same plotter, after
  `sbatch scripts/jed_collect_encoder_mapping.sh`): `outputs/figures/eeg_mapping_encoder/`.
- **Fit quality on held-out speech EEG** (does the mapping work): figures in
  `outputs/figures/insilico_mmn_small/fit_quality__*`; per-parcel test r ≈ 0.10 (mТРF) / 0.14 (enc).

---

## Open questions for you

1. **Tones vs speech.** Mapping is fit on speech EEG, but the MMN stimuli are pure tones — a
   speech-trained model extrapolating to non-speech. Want a speech-based oddball (phoneme/syllable
   deviants) instead?
2. **D2 split may be "easy"** (test shares the audiobook with train; windows overlap) — treat the
   A-vs-B gap cautiously (`project_plan §14`).
3. **Only `method_09` is run** — the full deviance-size / up-vs-down battery can be added.
