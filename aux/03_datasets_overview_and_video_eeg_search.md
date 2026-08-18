# EEG Datasets — What We Use, Why, and How to Extend the Search to Video

_Written 2026-08-18. Part 1–3 document the current (audio-only) dataset situation and the implicit
inclusion criteria. **Part 4 is a task brief for another agent**: repeat the dataset search, but for
EEG datasets with **video / audiovisual** stimuli._

Companion docs: [`01_setup.md`](01_setup.md) (download + env), [`sophies_repository_overview.md`](sophies_repository_overview.md) §1.4 / §2.1
(formatters, HDF5 schema), [`project_plan_20260611.md`](project_plan_20260611.md) §21 (CV / committed layers).

---

## 1. Datasets currently in the pipeline

All numbers below were read from the HDF5 files and the source WAVs on 2026-08-18, not copied from
older docs.

| | **D2 — Weissbart Cortical Surprisal** | **D1 — Broderick 2018** | **D3 — pooled** |
|---|---|---|---|
| **Status** | ✅ **the mapping dataset** — all committed layers and all MMN results | ❌ built the pipeline, then rejected for MMN | ⚠️ pooling experiments only |
| Provenance | **Zenodo `10.5281/zenodo.7775260`** (not OpenNeuro) | **OpenNeuro `ds004408`** | derived from D1 ∪ D2 |
| Paper | Weissbart, Kandylaki & Reichenbach, *J Cogn Neurosci* 32(1):155–166, **2020**. DOI `10.1162/jocn_a_01467` | Broderick et al., *Curr Biol* 28(5):803–809, 2018; Di Liberto et al., *Curr Biol* 25(19):2457–2465, 2015 | — |
| **Subjects** | **13** (P00–P12) | **19** | 32 recordings (disjoint people) |
| Channels | 63 @ 1000 Hz | 128 @ 512 Hz (BioSemi ActiveTwo, "unfiltered, unreferenced") | 62 in common |
| **Stimuli** | **15 audiobook parts** from 3 books (AUNP01–08, BROP01–03, FLOP01–04), 110–197 s each | **20 segments** of *The Old Man and the Sea*, 166–202 s each | 35 parts |
| **Unique audio** | **39.7 min** (0.66 h) | **60.5 min** (1.01 h) | 100.2 min (1.67 h) |
| **Total recording** | **8.6 subject-hours** | **19.2 subject-hours** | 27.8 subject-hours |
| **Repetitions** | **1** — each part heard once | **1** — each segment heard once | 1 |
| Windows (30 s / 10 s stride) | 157 train / 43 test | 252 train / 62 test | 409 train / 105 test (scored separately as `test_d1`/`test_d2`) |
| Held-out parts | AUNP02, BROP02, BROP03 | audio02, 09, 13, 14 | both sets |
| Local path | `sigfstea/multimodal-brain-scaling-temporal-analysis/data/cortical_suprisal_dataset/` | `mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/` (19 GB, full EEG) | built, not re-downloadable |
| Formatter | `mbs.data_prep.format_eeg_hdf5_surprisal` (**lives only in the `-temporal-analysis` repo**) | `mbs.data_prep.format_eeg_hdf5` (in the main repo) | `combine_eeg_hdf5.py` |

Plus **`surprisal_10s.h5`** — the same D2 subjects rewindowed to 10 s for the wav2vec2 runs (181
train / 49 test). ⚠️ It was built at **stride 10 s = non-overlapping**, whereas the 30 s file is
stride 10 s = **20 s overlap**. That is not just a window-length difference: the whole group-by-part
CV machinery (§21) exists to fix leakage from that overlap, and the noise ceiling is computed over
independent windows in one file and autocorrelated windows in the other.

### Why D1 was rejected
Not for size — D1 is the *bigger* dataset. Its **fronto-central channels fail the NC floor** exactly
where the MMN lives (Cz r ≈ 0.16; several channels at exactly 0.000, attributed to a montage
artifact). D2 is clean there (FCz r ≈ 0.99). See `XX_handover_for_Sophie.md` §3.

### Considered, never used
- **ERP-CORE** — named in `02_project_plan…` Phase 5 as the open fallback with an MMN paradigm
  (N=40). Never downloaded; no code path references it.
- **A clinical MMN EEG dataset (patient + control)** — the actual scientific target, listed as "TBD"
  since June and still not obtained. Everything current is a speech-trained mapping applied to pure
  tones, which is the standing out-of-domain caveat.

### Provenance gaps worth closing
- D2's DOI appears in **exactly one place** in the whole codebase: the docstring of
  `format_eeg_hdf5_surprisal.py:3`. There is no download script for it (D1 has one in
  `01_setup.md`). D2 is the dataset every committed result rests on.
- That docstring says "Weissbart et al. (**2023**)" — that is the Zenodo deposit year. The paper is
  **2020**. Fix before it reaches a manuscript.
- `preprocess.py` / `align_data.py` / `stimulus_order.csv` ship with the Zenodo record and sit in
  the dataset dir, but are mode `600` under `sigfstea` and unreadable to others. Zenodo calls the
  recordings "unprocessed", yet a `preprocess.py` ships alongside — **what filtering/referencing
  was already applied upstream is unverified.** This matters for the noise ceiling (see §3).

---

## 2. How the pipeline consumes a dataset

Any new dataset has to end up as the same HDF5, or nothing downstream works:

```
<split>/stimulus_ids            [n_win]  "<part>_<start_at_16kHz:07d>"
<split>/neural_data/group/<ch>  [n_win, T, 1]     float32, T = window_s × 50
noise_ceilings/group/<ch>       [T, 1]            % variance (r_SB² × 100)
attrs: target_sr=50, window_duration_s, window_stride_s, test_parts, channel_names, rois
```

Three things follow that constrain what datasets are usable at all:

1. **Everything is group-averaged.** `subjects` is literally `['group']`. Per-subject EEG is never
   stored, so the mapping is fit to a 13-subject mean.
2. **The noise ceiling is a random split-half across subjects** (6 vs 7 subjects, seed 42),
   correlated **across stimuli** per (time-bin, channel), Spearman–Brown corrected, computed on
   train windows only. There are **no stimulus repetitions in either dataset**, so a within-subject
   NC is not computable — the split-half-across-subjects route is forced.
3. **Channel names must resolve to the extended 10-20 system.** `eeg_targets.montage_pos()` parses
   prefixes (`FP/AF/F/FC/FT/C/T/CP/TP/P/PO/O/I`) and drops anything it can't place; `CLUSTERS`
   hard-codes 10-20 labels for the 5 parcels. D1's BioSemi `A1…D32` names only work because
   `_find_roi_indices()` does an MNE `biosemi128` → `standard_1020` nearest-neighbour remap.

---

## 3. The inclusion criteria we were implicitly filtering on

These were never written down; they are reconstructed from what D1/D2 have in common and what the
code requires. **Use these verbatim for the video search**, changing only the modality.

**Hard requirements** (fail any → unusable):

- **H1 — Continuous naturalistic stimulus**, not trial-based ERP epochs. The mTRF/encoder need a
  continuous time axis, not averaged evoked responses.
- **H2 — The stimulus files themselves are distributed**, not just annotations/transcripts. The
  model extracts features *from the stimulus*, so a dataset shipping only word onsets is useless.
- **H3 — EEG time-aligned to stimulus onset**, with the alignment recoverable (either pre-aligned,
  or a documented onset/trigger scheme). Known native sampling rate.
- **H4 — Multiple subjects on the *same* stimuli.** Required both for the group average and for the
  split-half noise ceiling. A dataset where every subject saw different material cannot produce a
  noise ceiling at all under the current method.
- **H5 — Standard 10-20 channel names, or a documented montage** that MNE can map to
  `standard_1020` (see §2.3).
- **H6 — Openly downloadable** without an application/DUA, ideally scriptable (S3, Zenodo, OSF).
- **H7 — ≥ 4 separable stimulus "parts"** (distinct files/runs), so `grouped_kfold(k=4)` by part can
  form non-overlapping CV folds. D2 has 12 train parts → 4 folds of 3.

**Soft/ranking criteria** (more is better):

- **S1 — Total unique stimulus duration.** This is the binding constraint, not subject count. D2 is
  only **40 minutes** of unique audio; that is very likely why r tops out around 0.2–0.3.
  **Anything offering multiple hours of unique stimulus is a major upgrade.**
- **S2 — Subject count** (drives NC reliability; D2's NC rests on a single 6-vs-7 partition).
- **S3 — Stimulus repetitions.** Neither current dataset has any. A dataset with repeats would allow
  a *within-subject* noise ceiling and a ceiling computed **along time** — which would fix a real
  methodological gap (the current NC is computed across stimuli at fixed offsets, while models are
  scored along time; the two are not the same quantity, and across the 47 electrodes they are
  uncorrelated, r ≈ −0.005).
- **S4 — Healthy adults** for the mapping. (Clinical/patient data is a separate, still-unmet need.)
- **S5 — Channel count and coverage**, especially fronto-central and temporal sites.
- **S6 — Raw or minimally preprocessed**, with the preprocessing documented.

---

## 4. TASK BRIEF — extend the search to video / audiovisual EEG datasets

**Goal.** Find EEG datasets whose stimuli are **video** (silent) or **audiovisual** (film, TV,
movie clips, naturalistic viewing), meeting the **same criteria H1–H7 / S1–S6 above**. Produce a
ranked shortlist we can act on.

**Why.** Two reasons. (a) The repo is `multimodal-brain-scaling` and already contains a legacy
vision→brain scaling pipeline (`sophies_repository_overview.md` §10) — video stimuli let the same
mTRF/encoder machinery be driven by *vision* models, giving a second modality to test the same
scaling question on. (b) Naturalistic film-watching EEG datasets tend to be far larger in unique
stimulus hours than speech-EEG datasets, which directly attacks the **40-minutes-of-audio** ceiling
that is currently limiting us (S1).

### 4.1 Criteria changes for video

Keep H1–H7 and S1–S6 exactly as written, with these modality substitutions and additions:

- **H2′** — the **video files** must be distributed, at a documented frame rate and resolution. A
  dataset that ships only EEG plus "participants watched *Despicable Me*" is **disqualified**, no
  matter how good the EEG is — we cannot extract model features from a film we don't have. Flag
  this explicitly per candidate; it is the single most common failure mode for movie-EEG datasets
  (copyright).
- **H3′** — frame-accurate alignment. Note whether alignment is per-frame, per-trigger, or must be
  reconstructed. Note any dropped-frame or timing-jitter warnings in the docs.
- **NEW: audio track present or not.** Record this per dataset — it decides how we can use it:
  - **Audiovisual with audio** → best case. The existing Whisper/wav2vec2 audio path still works,
    *and* a vision model can be added; enables a direct audio-vs-video-vs-both comparison on
    identical EEG.
  - **Silent video** → a clean vision-only test, but nothing of the current feature pipeline is
    reusable.
- **NEW: is the film content itself redistributable?** Some datasets link commercial films the user
  must source themselves. Distinguish "we can download the stimulus" from "we could in principle
  obtain the stimulus" — the first is H2′-passing, the second is a maybe.

### 4.2 Where to search

Work through all of these; don't stop at the first hit.

- **OpenNeuro** (`openneuro.org`) — filter modality = EEG, then search `movie`, `film`, `video`,
  `naturalistic viewing`, `audiovisual`, `cartoon`. Datasets are mirrored on S3 at
  `s3://openneuro.org/<accession>` and downloadable with `aws s3 sync --no-sign-request` (pattern in
  `01_setup.md`).
- **Zenodo** and **OSF** — this is where D2 came from, and where a lot of EEG sits that is *not* on
  OpenNeuro. Do not treat OpenNeuro as exhaustive; that assumption is arguably why the current
  search missed things.
- **Nemar** (`nemar.org`) and the **EEGLAB/NEMAR** dataset index.
- **PhysioNet**, **Figshare**, **Dryad**, **G-Node GIN**.
- **Papers-first sweep** — search Google Scholar / PubMed for
  `EEG naturalistic movie watching dataset`, `EEG film viewing open dataset`,
  `inter-subject correlation EEG video`, `EEG audiovisual narrative`, and then chase the data
  availability statement of each hit. Many usable datasets are only discoverable this way.
- **Known large-cohort resources** to check specifically: the **Healthy Brain Network** EEG release
  (children/adolescents; includes video-watching runs) and **Nemar-hosted naturalistic viewing**
  collections.

### 4.3 Leads to verify — NOT verified facts

I have **not** checked any of these; several may not exist in the form I describe, may be fMRI
rather than EEG, or may not ship stimuli. Treat every one as a hypothesis to confirm or kill, and
report the ones that don't pan out too, so we don't re-search them later:

- Healthy Brain Network EEG (video-watching runs, e.g. short film clips)
- Dmochowski / Parra lab EEG-during-film work (inter-subject correlation literature)
- SEED / SEED-IV / DEAP / MAHNOB-HCI (emotion-elicitation film clips — likely fail H2′ and H1)
- Any OpenNeuro EEG dataset tagged `naturalistic`
- EEG companions to naturalistic fMRI collections (Cam-CAN, Naturalistic Neuroimaging Database,
  StudyForrest) — StudyForrest in particular has multiple modalities; check whether any EEG exists
  and whether the *Forrest Gump* stimulus is obtainable

Also worth one pass: **MEG** datasets meeting the same criteria. Out of scope for the current HDF5
schema, but if a large well-aligned naturalistic-video MEG dataset exists, say so in a footnote —
that is a decision for us, not a silent exclusion.

### 4.4 Per-candidate verification checklist

For every candidate that survives a first look, verify each line **from the dataset's own
documentation or files** — not from a paper's abstract and not from memory. Cite where you checked.

```
Name / accession / DOI:
Repository + direct download URL:
Paper + DOI:
Licence / any DUA or application required:

H1 continuous naturalistic stimulus?        yes/no + what
H2' stimulus VIDEO files distributed?       yes/no  ← most common failure; be strict
    frame rate / resolution:
    audio track present?                    yes/no
    content redistributable, or link-only?
H3' EEG-stimulus alignment mechanism:       pre-aligned / triggers / reconstruct
H4 same stimuli across subjects?            yes/no
H5 channel names + montage:                 (10-20? BioSemi? mappable via MNE?)
H6 scriptable download?                     yes/no + command
H7 number of separable parts/runs:

S1 TOTAL UNIQUE STIMULUS DURATION:          ___ min      ← headline number
S2 subjects:                                ___
S3 repetitions per stimulus:                ___          ← >1 is a big deal, flag loudly
S4 population:                              healthy adults? clinical? children?
S5 channels / sampling rate:
S6 preprocessing state:                     raw / filtered / referenced — and documented?

Total recording = subjects × unique duration = ___ subject-hours
Approx download size:
VERDICT: usable / usable-with-work / disqualified (which criterion) / unverifiable
```

### 4.5 Output

Write results to **`aux/04_video_eeg_dataset_candidates.md`** in this directory, containing:

1. A **ranked shortlist table** — one row per surviving candidate, sorted by S1 (unique stimulus
   duration), with columns: name, accession/DOI, subjects, unique duration, subject-hours, repeats,
   audio track y/n, channels, verdict.
2. **One filled checklist block per candidate** (the template above).
3. A **rejected list** with the single criterion that killed each — including the §4.3 leads that
   turned out not to exist or not to qualify. This is as valuable as the shortlist.
4. A short **"closest to D2" note**: which candidate would drop into the existing pipeline with the
   least new code, and what specifically would need writing (new formatter? montage remap? video
   backbone in `audio_models.py`'s sibling?).

### 4.6 Ground rules

- **Verify, don't recall.** Fetch the actual record page or file listing. If a fact cannot be
  confirmed, write `unverified` — do not fill the field with a plausible number. A shortlist with
  three confirmed entries beats one with twelve half-remembered ones.
- **Do not download anything** in this pass. Report sizes and commands; we will decide what to pull.
- **Report the honest total.** If the search turns up nothing that beats D2's 40 minutes, say that
  plainly — that is a real and useful finding, and it would tell us the stimulus-duration ceiling is
  a field-wide constraint rather than a search failure on our part.
