# Encoder vs. mTRF Analysis Notes

Five decisions needed before moving to the stimuli-search component:

1. **Stimulus set** — gather a stimulus set that matches the Definition 2 (Zurich-lab,
   final-tone-identical) design, or revert the analysis to Definition 1 (classic oddball) stimuli
   instead.
2. **MMN metric/criterion** — which of C0 through S6 to adopt as the standard verdict.
3. **Mapping method** — encoder vs. mTRF.
4. **Analysis level** — parcel-level vs. electrode-level MMN.
5. **ROI definition within that level** e.g. a single-site ROI (Fz for electrodes,
   central for parcels), per the Section 5/6 findings.
6. **Threshold for model inclusion in stimuli search** (Later)


## Trace plot locations (cluster)

Per (model, layer, method), the MMN trace plots are written as separate row-block PNGs:

- **Electrodes — Fz/FCz only (includes a C0–S6 criteria table):**
  - mTRF: `/work/upschrimpf/sigfstea/multimodal-brain-scaling/outputs/figures/insilico_mmn_electrodes/<model>/insilico_mmn_electrodes_fz_fcz__<method>__<layer>.png`
  - Encoder: `/work/upschrimpf/sigfstea/multimodal-brain-scaling/outputs/figures/insilico_mmn/<model>-electrodes/insilico_mmn_fz_fcz__<method>__<layer>__attn.png`
- **Parcels — frontal/central/temporal (includes a C0–S6 criteria table):**
  - mTRF: `/work/upschrimpf/sigfstea/multimodal-brain-scaling/outputs/figures/insilico_mmn/<model>/insilico_mmn__<method>__<layer>.png`
  - Encoder: `/work/upschrimpf/sigfstea/multimodal-brain-scaling/outputs/figures/insilico_mmn/<model>-parcels/insilico_mmn__<method>__<layer>__attn.png`
- **Electrodes — If wanting to view additional electrodes + topography (difference-traces-only, no criteria table):**
  - mTRF: `/work/upschrimpf/sigfstea/multimodal-brain-scaling/outputs/figures/insilico_mmn_electrodes/<model>/insilico_mmn_electrodes__<method>__<layer>.png`
  - Encoder: `/work/upschrimpf/sigfstea/multimodal-brain-scaling/outputs/figures/insilico_mmn/<model>-electrodes/insilico_mmn_electrodes__<method>__<layer>__attn.png`

The Fz/FCz-only and parcels row-blocks each embed a table showing whether that electrode/parcel meets the C0–S6 criteria.


## 1. Stimulus Set

**DECISION** :We either reproduce results with Weber 2022 + other stimuli from the Zurich lab (either published or they provide) or we revert to Def 1 stimuli, which requires re-run of analysis from start but would have plenty available. Recall:

Per `aux/results_analysis.md` §0 ("Note on MMN stimulus definitions — correction"), the stimuli
analyzed throughout `results_analysis.md` are **not** built according to either literature
definition cleanly:

- The 10 sourced papers all use **Definition 1** (classic frequency-oddball — the deviant tone's
own frequency differs from the standard's, ERP locked to that tone's own onset).
- The stimuli actually generated and analyzed are built the **Definition 2** way (multi-tone
trains where only the final/eliciting tone is scored, and that tone is physically identical
between standard and deviant trains) — even though the (standard, deviant) frequency pairs
feeding those trains are sourced from the Definition 1 papers above.
- Paper(s) outside Weber 2022 that implement a genuine Definition 2 design have yet to be identified in the literature search.

That is, every result below and in results_analysis.md should be read as: *Definition-2-style stimuli, built from Definition-1-sourced frequency pairs* thus provides a proxy of results that would be obtained using Definition 2 stimuli.

## 2. MMN Metric (C0–S6)
**DECISION**: Choose a metric that balances prioritization of MMN-shape without being too stringent
### Summary of C0–S6 metrics


| ID  | Name                             | Definition                                                                                                                                                                                        |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C0  | current (magnitude only)         | Most negative point in the 100–240 ms window is below 0. No shape requirement.                                                                                                                    |
| S1  | interior argmin                  | Like C0, plus the trough must sit strictly inside the window (not at either edge).                                                                                                                |
| S2  | trough + recovery                | Like C0, plus the curve must recover ≥50% of the trough's depth within 120 ms.                                                                                                                    |
| S3  | interior & recovery              | S1 and S2 both required.                                                                                                                                                                          |
| S4  | tone-end-relative dip + recovery | Scans every sample in a per-method window after the tone ends for any qualifying dip that recovers ≥50% by an absolute deadline; doesn't require the qualifying point to be the window's deepest. |
| S5  | unbound dip + recovery           | Like S2, but the trough search runs over the entire post-onset trace instead of the fixed window.                                                                                                 |
| S6  | envelope-guarded unbound search  | Like S5, plus the trough's latency must fall inside a literature-derived plausibility envelope (90–250 ms).                                                                                       |


The criterion choice trades off false positives (curve shapes that aren't a real MMN trough) against false negatives (rejecting genuine troughs on a technicality):

- **C0** is too permissive — it is satisfied by any curve that is most-negative somewhere in the window, including monotonic ramps with no real trough.
- **S1/S3** are too strict — the interior-only requirement rejects legitimate mTRF troughs that sit right at the window edge.
- **S2/S4** are the recommended middle ground — they require a genuine dip-and-recover without penalizing edge-sitting troughs, and hold up under both the multi-site and single-site ROI (`results_analysis.md` Sections 4 and 6).
- **S5/S6** trade the fixed window for an unbound search; S6's envelope guard is the strictest of all seven and may be excluding genuine but later/earlier troughs.
 
- Comparison of Results across metrics for mTRF and Encoder methods: (*Each cell is scored out of 20 as there exists parcel-based MMN and electrode-based MMN for each model*)

**Table 29. mTRF, combined** (= Table 25 + Table 26, n/20 per model)

| Model            | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) |
| ---------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny             | 20/20     | 11/20     | 15/20     | 8/20      | 18/20     | 18/20     | 7/20      |
| base             | 12/20     | 6/20      | 12/20     | 6/20      | 16/20     | 16/20     | 5/20      |
| small            | 18/20     | 11/20     | 18/20     | 11/20     | 20/20     | 19/20     | 8/20      |
| medium           | 19/20     | 17/20     | 19/20     | 17/20     | 19/20     | 15/20     | 11/20     |
| **Total (n/80)** | **69/80** | **45/80** | **64/80** | **42/80** | **73/80** | **68/80** | **31/80** |


**Table 30. Encoder, combined** (= Table 27 + Table 28, n/20 per model)


| Model            | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) |
| ---------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny             | 12/20     | 8/20      | 3/20      | 3/20      | 4/20      | 1/20      | 0/20      |
| base             | 13/20     | 6/20      | 8/20      | 4/20      | 11/20     | 11/20     | 4/20      |
| small            | 9/20      | 3/20      | 5/20      | 2/20      | 6/20      | 4/20      | 1/20      |
| medium           | 14/20     | 3/20      | 3/20      | 2/20      | 6/20      | 6/20      | 1/20      |
| **Total (n/80)** | **48/80** | **20/80** | **19/80** | **11/80** | **27/80** | **22/80** | **6/80**  |

## 2. Mapping Method — Encoder vs. mTRF
### Encoder vs. mTRF Fit
- mTRF traces are more expressive — more genuine dip-and-recover troughs — which is the likely
driver of the gap: the appendix's same-stimulus comparison ("mTRF vs. Encoder — expressiveness
gap on the same stimulus") shows the encoder producing a flat or monotonic ramp with no real
trough where mTRF shows a clean recovering trough.
- **Encoder fit-quality beats mTRF in all 8 model × level cells**, though — held-out test r at each mapping's chosen layer, from `outputs/results/eeg_mapping/*.json` (mTRF) vs.
`outputs/results/eeg_mapping_encoder/*.json` (encoder); rendered in
`outputs/figures/eeg_mapping{,_encoder}/test_fit_quality__{parcels,electrodes}__D2.png`:


| Model  | Level      | mTRF layer | mTRF test r | Encoder layer | Encoder test r | Δ (encoder − mTRF) |
| ------ | ---------- | ---------- | ----------- | ------------- | -------------- | ------------------ |
| tiny   | parcels    | blocks.0   | 0.187       | blocks.3      | 0.234          | +0.047             |
| base   | parcels    | blocks.0   | 0.153       | blocks.2      | 0.271          | +0.118             |
| small  | parcels    | blocks.3   | 0.073       | blocks.10     | 0.197          | +0.124             |
| medium | parcels    | blocks.11  | 0.078       | blocks.4      | 0.238          | +0.160             |
| tiny   | electrodes | blocks.0   | 0.212       | blocks.3      | 0.312          | +0.100             |
| base   | electrodes | blocks.0   | 0.173       | blocks.0      | 0.286          | +0.113             |
| small  | electrodes | blocks.1   | 0.118       | blocks.10     | 0.234          | +0.116             |
| medium | electrodes | blocks.12  | 0.079       | blocks.3      | 0.228          | +0.149             |


- The gap **widens with model size**: `whisper-tiny` shows the smallest encoder advantage (+0.047 to +0.100); base/small/medium show a consistently larger one (+0.11 to +0.16).
- mTRF's chosen-layer test r degrades faster than the encoder's as models get bigger.
- That is, encoder predicts the actual held-out EEG more accurately at every model/level, but mTRF's predicted traces are the ones producing more (and more shape-confirmed) MMN troughs. Fit quality and MMN-detection rate oppose each other.
  - MTRF is a defensible method, as been around for quite some time. --> Any way we can we get MTRF r higher such that gap closes? We shouldn't be penalized for accuracy of new MIRAGE method as others wouldn't even try it?

### Encoder vs mTRF MMN Results 
- mTRF beats encoder under every criterion across models for MMN-positive, under the single-site Fz/central ROI (`results_analysis.md` Tables 25–28; combined in **Table 29 + 30 which are provided in above section**)


## 4. Analysis Level — Parcels vs. Electrodes
- TODO: Email Zurich lab if they can manage parcels vs. electrodes. 
- **Under the single-site Fz/central ROI (Section 6), neither level consistently wins.** The per-level gap is small and flips
depending on criterion: For example S2 retains encoder slightly worse at electrodes/Fz (9/40, 22.5%) than
parcels/central (10/40, 25%), but S4 reverses this (electrodes/Fz 14/40=35% vs. parcels/central
13/40=32.5%). 
- See `results_analysis.md` Section 5 (Tables 15–16 for electrodes, Tables 17–18 for parcels,
continuous-magnitude check in Tables 19–22) and Section 6 (Tables 25–30) for the full
level-specific MMN-positive counts under both mTRF and encoder.

## 5. ROI Definition (within a level)
**DECISION: Decide which regions of interest (ROI) to use to define the MMN**
**Table 23. Electrodes, combined (mTRF + Encoder)** (Table 15 + Table 16, n/20 per model)


| Model            | Fz        | FCz       | Fz_FCz    | current7  |
| ---------------- | --------- | --------- | --------- | --------- |
| tiny             | 15/20     | 15/20     | 14/20     | 15/20     |
| base             | 13/20     | 12/20     | 13/20     | 13/20     |
| small            | 13/20     | 13/20     | 12/20     | 12/20     |
| medium           | 16/20     | 15/20     | 16/20     | 15/20     |
| **Total (n/80)** | **57/80** | **55/80** | **55/80** | **55/80** |


Winner: Fz (using C0)  

**Table 24. Parcels, combined (mTRF + Encoder)** (Table 17 + Table 18, n/20 per model)


| Model            | frontal   | temporal  | central   | current2  |
| ---------------- | --------- | --------- | --------- | --------- |
| tiny             | 16/20     | 15/20     | 17/20     | 16/20     |
| base             | 14/20     | 13/20     | 12/20     | 12/20     |
| small            | 10/20     | 10/20     | 14/20     | 13/20     |
| medium           | 16/20     | 16/20     | 17/20     | 16/20     |
| **Total (n/80)** | **56/80** | **54/80** | **60/80** | **57/80** |


Winner: Central (using C0)  

- **Recommendation: a single-site ROI — Fz for electrodes, central for parcels***
- But, have not ran all {CO, S1, ... S6} x {frontal, central, Fz, FCz, ...} to decide best possible combination. However, I think it is encouraging that these locations were identified by this metric as the best fit.
- Have only ran {C0} x {frontal, central, Fz, FCz, mean of 2 parcels ('current2') (Above Tables, from Section 5 in results_analysis.md) + mean of 7 frontal electrodes('current7')} and {CO, S1, ... S6} x {central, FCz} (Section 6 in results_analysis.md)
- See `results_analysis.md` Section 5 (Tables 15–22, present-counts and magnitude by ROI
variant) and Section 6 (Tables 25–30) for the full ROI-variant and C0–S6 sweep data underlying
this recommendation.



