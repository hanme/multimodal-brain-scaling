# Figure Overview — In-Silico MMN

A map of every figure produced for the MMN analysis, so you can find the visual instead of the
table. Three groups: **(A)** the 9 curated example figures committed to the repo (render in
`results_analysis.md`'s Appendix), **(B)** the full systematic MMN figure set in Sophie's shared
tree, and **(C)** the layer-selection / mapping-quality figures.

Every per-method MMN figure has the same layout: columns = **deviant / standard / deviant−standard**
(all mean-baseline-corrected), time 0 = final/eliciting-tone onset, the 100–240 ms MMN band shaded;
the deviant−standard column is annotated with the `baseline_normalized_peak` (BNP) verdict value.
Parcel figures show rows = **frontal / central / temporal**; electrode figures show the 10-20 montage.

⚠️ Design caveat that applies to **all** MMN figures below: the 10 stimulus methods are **Definition 1
(classic oddball)** — the deviant's eliciting tone *differs* from the standard's — so every
deviant−standard trace mixes prediction-error with the plain acoustic difference of the probe tone.
Read them as "does the model show *a* deviance response," not a confound-free MMN.

---

## A. Curated example figures — `aux/images_for_analysis/` (in git; embedded in `results_analysis.md` Appendix)

All from **whisper-tiny, parcels**. The `__attn` files are the **encoder** (its chosen layer
`blocks.3`); the two without `__attn` are **mTRF** (`blocks.0`). They illustrate the MMN
**shape-criteria** (C0 magnitude-only, then S1–S6) and the mTRF-vs-encoder gap.

- **`insilico_mmn__method_27__blocks.3__attn.png`** — *C0 (magnitude-only) example.* Trough is negative but sits at the window's right edge (220.9 ms) and never recovers; passes magnitude-only C0 yet every shape criterion (S1–S6) rejects it — the case that magnitude alone is too weak.
- **`insilico_mmn__method_72__blocks.3__attn.png`** — *C0 failure / ramp artifact.* A huge −6.5 "trough" that is actually a smooth monotonic ramp through the window (near-zero baseline std inflates the z-score); looks strongest exactly where the shape is *least* MMN-like.
- **`insilico_mmn__method_75__blocks.3__attn.png`** — *C0 failure / ramp artifact (twin of method_72).* Same −6.5 ramp pathology; both are the Section-4 outliers and both are rejected by all shape criteria.
- **`insilico_mmn__method_37__blocks.3__attn.png`** — *S1 (interior argmin) example.* Trough is comfortably inside the window (210.9 ms) so S1 passes, but recovery is negligible — shows an interior trough alone isn't a genuine dip-and-recover.
- **`insilico_mmn__method_60__blocks.3__attn.png`** — *S2/S3/S6 "textbook" example.* Interior trough at 200.9 ms that recovers ~58% of its depth and sits inside the literature latency envelope; the one run in this slice that passes every criterion C0→S6.
- **`insilico_mmn__method_44__blocks.3__attn.png`** — *S5 (unbound search) example.* No dip in the fixed 100–240 ms window, but searching the whole trace rescues an early trough (~31 ms) that recovers strongly — a real trough the fixed window misses.
- **`insilico_mmn__method_55__blocks.3__attn.png`** — *S4 example + encoder side of the method_55 comparison.* The fixed window is *positive* (no dip → C0 fails), but a tone-end-relative scan finds a later genuine dip-and-recover; also the encoder half of the sign-flip vs mTRF below.
- **`insilico_mmn__method_55__blocks.0.png`** — *mTRF side of method_55.* On the identical stimulus the mTRF shows a textbook interior dip-and-recover (−1.43 at 120.9 ms) — the sharp mTRF-vs-encoder disagreement (clear trough vs no dip) on one stimulus.
- **`insilico_mmn__method_60__blocks.0.png`** — *mTRF side of method_60.* Off-center, non-recovering shallow trough (−0.56), and its deeper unbound trough is implausibly early (~21 ms) → fails the envelope guard; the mirror of the encoder's clean pass on the same input.

---

## B. Full systematic MMN figure set — `/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/figures/`

268 PNGs total (not in git; `*.png` is gitignored — view them directly on the shared tree). Three
families, one figure per cell:

- **Method A (mTRF), parcels** — `insilico_mmn/<model>/insilico_mmn__<method>__<layer>.png`
  One per (model × 10 methods) = 40. The frontal/central/temporal deviant−standard grid for the
  linear mapping at that model's committed mTRF layer.
- **Method B (encoder), both levels** — `insilico_mmn/<model>-<level>/<method>/insilico_mmn__<method>__<layer>__attn.png`
  One per (model × {parcels,electrodes} × 10 methods) = 80, each with a companion
  **`fit_quality__attn__<layer>__attn.png`** showing recorded-vs-predicted held-out EEG (how well the
  encoder actually fits, separate from the MMN verdict).
- **Method A (mTRF), electrodes** — `insilico_mmn_electrodes/<model>/insilico_mmn_electrodes__<method>__<layer>.png`
  One per (model × 10 methods) = 40. The same deviant−standard read-out laid out on the 10-20
  electrode montage, with the fronto-central ROI driving the present/absent verdict.

Use these for any specific (model, method, mapping, level) you want to eyeball; the Appendix (group A)
is the curated subset that makes the shape-criteria points.

---

## C. Layer-selection / mapping-quality figures (group-by-part CV)

These are about *which layer best predicts the EEG* and *how well* — upstream of the MMN. mTRF JSONs
are tracked in git (`outputs/results/eeg_mapping/*.json`); the PNGs are regenerable (encoder set built
by `scripts/jed_collect_encoder_cv.sh`). Both live under `outputs/figures/`.

- **`eeg_mapping/layer_selection__{parcels,electrodes}__D2.png`** (mTRF) and
  **`eeg_mapping_encoder/layer_selection__{parcels,electrodes}__D2.png`** (encoder) — mean held-out r
  vs model depth, one line per model, **CV-chosen layer circled** (solid = group-by-part CV-on-train,
  dashed = held-out test). Shows the de-inflated, non-overlapping layer pick.
- **`eeg_mapping{,_encoder}/test_fit_quality__{parcels,electrodes}__D2.png`** — per-target held-out
  test r bars at each model's chosen layer; the direct mTRF-vs-encoder fit-quality comparison (encoder
  wins all 8 model×level cells).

---

_Sources: `aux/results_analysis.md` (Sections 1–6 + Appendix), `aux/sophies_repository_overview.md`
(§5, §16), and the figure trees above. Stimulus-design status (Def 1 vs Def 2) is unresolved — see
`results_analysis.md` §0 and the overview §16.2._
