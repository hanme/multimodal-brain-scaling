# MMN screening plan — does the model reproduce the human MMN?

_2026-06-17. Companion to `XX_handover_for_Sophie.md`._

**Goal.** Feed experimental MMN stimuli to a model that predicts EEG **at the electrode level**, then
check whether a mismatch negativity appears for the frequency pairs known to elicit an MMN in humans.
Each model's score = **% of pairs with an MMN**.

## Division of labour
- **Hannes — code.** Driver that takes an MMN stimulus pair (standard + deviant sequence), runs it
  through the model + frozen mTRF mapping, and outputs predicted EEG per electrode — same machinery as
  `outputs/figures/insilico_mmn_method09_blocks.2.png`, just at electrode rather than parcel level.
- **Sophie — stimuli + evaluation.** Selects the ~10 frequency pairs, runs them through the driver,
  and judges from the figure (eyeball or simple criterion) whether each shows an MMN.

## Stimulus selection (Sophie, ~10 pairs)
- Classic oddball: rare deviant among repeated standards, MMN read time-locked to the last tone.
- **Pick pairs from recent EEG MMN studies where the last/eliciting tone is physically identical in
  standard and deviant** (Definition 2 — the deviance lives in the preceding context, so surprise is
  not confounded with acoustics; see `project_plan_20260611.md §17`). Avoid the classic Sams/Tiitinen
  pairs, which differ physically on the last tone (Definition 1).
- Span a graded range of frequency deviance + include a 0% / sub-threshold negative control, so the
  screen tests both *presence* and the right *ordering* (small Δf → small/no MMN; large Δf → MMN).

## MMN evaluation (Sophie)
- MMN present = negativity in the **100–240 ms** window at a **fronto-central** electrode
  (Umbricht & Krljes 2005; Sams 1985 peak ≈170 ms). Pre-specify the site — don't fish.
- mTRF preserves amplitude, so an amplitude-based criterion is fine. Confirm the control stays flat.
- Score = % of pairs with an MMN.

## Notes
- mTRF only for now (amplitude meaningful). The attention encoder is deferred until retrained with MSE
  (`project_plan_20260611.md §16`).
- Caveat to flag: speech-trained model + speech-fit mapping applied to pure tones is out-of-domain;
  a speech-based oddball would be cleaner (`XX_handover_for_Sophie.md §9`).
- Schizophrenia angle uses **duration** deviants more than frequency (Shelley 1991; Umbricht 2005) —
  a duration condition can be added later.
