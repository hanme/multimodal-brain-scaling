# Handover: Auditory EEG Encoding Models — Sophie

_Last updated 2026-06-16. This is the focused deliverable note. The full technical log (scaling
across the Whisper family, the PCA/overflow saga, env/GPU notes, superseded methods, repo layout) is
in `aux/project_plan_20260611.md`._

This note describes **what we built for your MMN pipeline** and **where it is**. It reports the
numbers as-is and does not interpret them — that part is yours.

---

## 1. What this is

A trained **model→EEG mapping**: given a Whisper audio-model layer's activations for a sound, it
predicts a recorded-EEG time course at the scalp-parcel level. Once trained on speech EEG, the same
mapping can be applied to **your MMN tones** to ask whether an MMN-like deflection appears in the
predicted EEG (the "in-silico MMN").

You get two figures plus the underlying data:
1. **Fit-quality figure** — how well the mapping predicts held-out *speech* EEG (does the mapping work).
2. **In-silico MMN figure + per-parcel prediction data** — the predicted EEG for the MMN
   standard/deviant tones, so you can run your own deviant−standard MMN metric on it.

Two **swappable methods** produce both figures (§4). The model and layer are now **fixed** (§2).

---

## 2. Model and layer — NOW FIXED

| | value |
|---|---|
| Model | **whisper-small** (the OpenAI Whisper-small *encoder*) |
| Layer | **`blocks.10`** — the 11th of whisper-small's 12 encoder blocks |

⚠️ **`blocks.10` is specific to whisper-small.** Smaller Whisper variants have fewer blocks
(whisper-base has 6, tiny has 4), so the index does **not** transfer — `blocks.10` doesn't exist in
them. If you change model size you must re-pick a layer. Everything below is whisper-small / `blocks.10`.

---

## 3. Dataset and parcels

**Training EEG = D2 "Weissbart Cortical Surprisal"** (`outputs/neural_data/surprisal_30s.h5`):
13 subjects, 63-channel native 10-20 montage, **continuous-speech audiobook listening**,
group-averaged. Split by story part (one part held out per audiobook) → **157 train / 43 test** 30 s
windows. (There is a second speech dataset, D1 = Broderick; see §6 for why we did **not** use it here.)

**Parcels = D2-native 5** — each is the raw average of its channels, after a noise-ceiling floor
(cross-subject reliability **r > 0.2**) drops unreliable channels:

| parcel | channels | NC r |
|---|---|---|
| frontal | Fz, F3, F4, FCz | 0.72 |
| central | Cz, C3, C4 | 0.69 |
| temporal | T7 | 0.86 |
| parietal | Pz, P3, P4, P7 | 0.77 |
| occipital | O1, O2 | 0.69 |

`FCz` and the `central` parcel are **kept here** (they sit over the MMN generator). The cross-dataset
scaling tables in the project plan use a different, canonical-4 scheme — this single-dataset MMN
illustration uses D2's own reliable parcels.

---

## 4. The two methods (swappable)

Both predict the same 5 parcels from the same `blocks.10` features and are scored the same way, so
they sit side by side. Either one produces both figures.

- **Method A — mTRF (linear lagged ridge).** Closed-form RidgeCV predicting `EEG[t]` from features
  at lags 0–800 ms (20 ms steps), 0.5 Hz high-pass, per-channel alpha. CPU, cheap to re-fit.
  Code: `src/mbs/evaluation/evaluate_features_mtrf.py`.
- **Method B — attention encoder (MIRAGE-style).** A small gradient-trained network
  (`LatentAttentionTrunk` + group readout) attending over the 800 ms lookback window. Trained with
  `1 − Pearson` loss, which is scale/shift-invariant: **predicted amplitude is arbitrary, the sign
  is preserved.** Config: d_model=64, num_latents=4, 1 cross-attn layer, dropout=0.3, wd=1e-2,
  200 epochs. GPU to train; the trained checkpoint is saved and reused. Code:
  `src/mbs/evaluation/evaluate_features_attn_probe_temporal.py`; reusable I/O in
  `src/mbs/evaluation/attn_probe/checkpoint.py` (`load_probe_checkpoint`, `predict_parcels`,
  `predict_timecourse`).

**Trained B checkpoint (the reusable mapping):**
`outputs/results/whisper-small-probe-group-d2-mmn/model__blocks.10.pt`

---

## 5. Figure 1 — fit quality on held-out speech EEG

Per-parcel Pearson r between predicted and recorded EEG, scored **along time on all 43 held-out D2
test windows** (identical scoring for both methods). `r/NC` = r divided by the parcel's noise ceiling.

| parcel | NC r | A (mTRF) r | A r/NC | B (attn-enc) r | B r/NC |
|---|---|---|---|---|---|
| frontal | 0.72 | 0.132 | 0.183 | 0.163 | 0.226 |
| central | 0.69 | 0.097 | 0.141 | 0.063 | 0.091 |
| temporal | 0.86 | 0.134 | 0.156 | 0.180 | 0.209 |
| parietal | 0.77 | 0.086 | 0.111 | 0.144 | 0.186 |
| occipital | 0.69 | 0.064 | 0.093 | 0.131 | 0.190 |
| **mean** | | **0.103** | **0.137** | **0.136** | **0.190** |

(A scored by `scripts/score_mtrf_fitquality.py` → `outputs/results/mtrf_fitquality_d2_blocks10.json`;
B from `outputs/results/whisper-small-probe-group-d2-mmn/attn_probe_temporal_summary.json`. All values
positive. B is higher on 4/5 parcels; A is higher on central.)

Figure: `outputs/figures/insilico_mmn_small/fit_quality__attn__blocks.10__attn.png`.

---

## 6. Why D2 (Weissbart) and not D1 (Broderick) for the parcels / MMN

Both are continuous-speech audiobook datasets. We define the parcels on **D2** because Broderick's
**fronto-central electrodes — exactly where the MMN is generated — fail the NC floor**: Cz r≈0.16,
C3≈0.09, C4≈0.00, with several channels at suspicious **exact 0.000** (likely a montage/mapping
artifact in Broderick's BioSemi layout). On D2 the same region is clean (central NC r≈0.69) and the
single best-tracked channel is **FCz (r≈0.99), over the MMN generator**. So D2 gives reliable
fronto-central parcels where the MMN lives; D1 does not. (D1 still appears in the cross-dataset
scaling tables in the project plan.)

---

## 7. Figure 2 — in-silico MMN + per-parcel prediction data

We feed the MMN stimuli through whisper-small `blocks.10`, apply the mapping, and get a predicted EEG
time course per parcel. Current run = **`method_09`** (identity-MMN design: the final/critical tone
is *physically identical* in standard and deviant; the deviance is in the preceding **context**
frequency, 1000→600 Hz, DOWN ~40 %), 15 deviant variants.

**Prediction data (RAW, not baseline-corrected — you pick the baseline window):**
```
outputs/insilico_mmn_predictions_small/predictions__blocks.10.h5         # Method A (mTRF)
outputs/insilico_mmn_predictions_small/predictions__blocks.10__attn.h5   # Method B (attention encoder)
```
Each file:
```
attrs: layer, fs(50), time_step_ms(20), highpass_hz(0.5), note   (+ heldout_r/_nc on the mTRF file)
parcels        [5]      b'frontal' b'central' b'temporal' b'parietal' b'occipital'
parcel_members [5]      e.g. b'Fz+F3+F4+FCz'
parcel_nc_r    [5]      cross-subject reliability per parcel
method_09/
   time_ms       [1460]      0 = final/critical-tone onset; negatives = pre-onset baseline
   standard      [1460, 5]   raw predicted EEG, standard clip (cols = parcels)
   deviant_mean  [1460, 5]   mean over the 15 deviant variants
   deviants      [15,1460,5] per-variant, for trial-level stats
   deviant_ids   [15]
   attrs: context_final, direction, final_tone_onset_s, n_deviants
```

**Compute the MMN metric (parcel × method):**
```python
import h5py, numpy as np
with h5py.File("outputs/insilico_mmn_predictions_small/predictions__blocks.10__attn.h5") as h:
    parcels = [p.decode() for p in h["parcels"][:]]
    g = h["method_09"]
    t   = g["time_ms"][:]
    mmn = g["deviant_mean"][:] - g["standard"][:]          # [n_t, 5] = your D_A
base = (t >= -150) & (t < 0)                                # pre-onset baseline
mmn_bc = mmn - mmn[base].mean(0, keepdims=True)
band = (t >= 100) & (t <= 250)                             # MMN band
neg_peak = mmn_bc[band].min(0)                             # most-negative value per parcel
for p, v in zip(parcels, neg_peak):
    print(f"{p:9s} 100-250 ms negative peak = {v:+.3f}")
```
⚠️ For **Method B**, predicted amplitude is arbitrary (corr-loss is scale-invariant) — read the
**sign/shape** of the deflection, not its magnitude. Method A preserves amplitude.

Figures: `outputs/figures/insilico_mmn_small/insilico_mmn__method_09__blocks.10.png` (A) and
`…__blocks.10__attn.png` (B). Rows = the 5 parcels, columns = deviant / standard / deviant−standard,
100–250 ms band shaded, x = time from final-tone onset.

**To regenerate / swap method / add more MMN methods:**
```bash
# Train B (GPU, Kuma):       sbatch scripts/kuma_probe_mmn.sh
# Build both figures (CPU):  sbatch scripts/slurm_insilico_mmn_attn.sh \
#   --checkpoint outputs/results/whisper-small-probe-group-d2-mmn/model__blocks.10.pt \
#   --mmn_features_root outputs/features/whisper-small-mmn --method method_09 \
#   --features_dir <D2 features> --neural outputs/neural_data/surprisal_30s.h5 \
#   --out_dir outputs/figures/insilico_mmn_small --data_dir outputs/insilico_mmn_predictions_small
```

---

## 8. Open questions for you

1. **Stimulus modality at the MMN step (tones vs speech).** Whisper is trained on speech, and we fit
   the mapping on continuous-speech EEG, so the *training* side is modality-matched. But the **MMN
   stimuli themselves are pure tones, not speech** — we are asking a speech-trained model + speech-fit
   mapping to extrapolate to non-speech inputs. Do we want a **speech-based oddball paradigm** (e.g.
   phoneme/syllable deviants) so the MMN stimuli match the model's and mapping's domain, rather than
   pure tones?
2. **D2's split may be "easy."** D2 is split by story part with one part held out *per audiobook*, so
   test windows share the same audiobook as train (and the 10 s-stride windows overlap). The
   high-capacity attention encoder (B) may exploit this more than the rigid ridge (A). Treat the
   A-vs-B gap on D2 cautiously; verification agenda is in `project_plan_20260611.md §14`.
3. **Only `method_09` is currently run.** If you want the full deviance-size / up-vs-down sweep (the
   8-method identity-MMN battery is documented in the project plan), we can extend the run.

---

_Background (repo layout, datasets, hook architecture, how features are extracted, the full method
history and scaling results) lives in `aux/project_plan_20260611.md` and the MMN pipeline doc
`00_schizophrenia_pipeline_Sophie_2026.md`._
