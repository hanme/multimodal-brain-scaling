# Delta-T causality, the full-window normalisation leak, and the transformers 5.x check

_2026-08-18. Measured on `jst137`, not inferred. Reproduce with
`scripts/check_wav2vec2_api.py` and `scripts/check_mmn_norm_leak.py`._

## TL;DR

1. **Delta-T features are causal, exactly.** The stored feature for frame `t` is bit-identical
   when post-`t` audio is replaced (`max|Δ| = 0.000e+00`). No approximation.
2. **One acknowledged exception**: the *normalisation constants* are derived from the whole
   window before truncation, so `feature[t]` sees post-`t` audio through 1–2 scalars.
   **Measured at 0.000e+00 on the MMN stimuli in both model families, ≈4e-6 on speech.**
   → **Document it; do not fix it; do not re-extract anything.**
3. **Sophie's N-effect result is not a normalisation artifact.** The stimulus generator holds
   total energy constant across `N ∈ {3,5,7}`, so all N get the same constants to within 1e-6.
4. **transformers 5.15.0 runs the wav2vec2 path correctly**, despite the July features being
   extracted under 4.x. No version pin needed.

---

## 1. What the leak is

Both delta-T paths compute their normalisation from the **full window, before truncation**:

| Family | Constant(s) | Where |
|---|---|---|
| wav2vec2 | `mean`, `std` over all samples | `_wav2vec2_norm_stats` (`extract_features_delta_t.py`) |
| whisper | `log_spec.max()` (clamp reference) and `_silence_value = mel_full.max() − 2` | `whisper.log_mel_spectrogram`, `_silence_value` |

This is deliberate and documented in the module comment — per-`t` statistics would be degenerate
for early frames (at `t=0`, wav2vec2 would normalise by the statistics of 20 ms of audio). The
consequence is that `feature[t]` is not *strictly* a function of audio up to `t`: it depends on
the future through those scalars.

Everything else is strictly causal. `_truncate_waveform` zeroes samples `≥ (t+1)·conv_stride`;
`_truncate_mel` fills frames `≥ 2(t+1)`; and `extract_delta_t_waveform` reads `arr[i, t, :]` from
the pass truncated **at** `t` — never frame `t` from a full-window pass. That distinction matters:
wav2vec2's encoder is a **bidirectional** transformer, so frame `t` of a full-window pass attends
over the entire window. Delta-T exists precisely to prevent that. Measured leakage in a
full-window pass is ≈0.62; through the delta-T procedure it is exactly 0.

## 2. What was measured

**Speech-like audio** (`check_wav2vec2_api.py` step 6, wav2vec2-medium, 10 s):
perturbing post-cut audio moved the constants `std 0.050092 → 0.049985` (0.2 %) and the frame by
`4.19e-06` — 0.00 % of the frame's max activation.

**Real MMN stimuli** (`check_mmn_norm_leak.py`, `method_09`, literature set). Method: take one
waveform, extract frame `t` twice — once with its own constants, once with another stimulus's —
so the audio is identical and the entire difference is the leak.

| | wav2vec2-medium | whisper-tiny |
|---|---|---|
| layer / frame | `encoder.layers.6`, t=425/500 (85 %) | `blocks.2`, t=1275/1500 (85 %) |
| constant spread over 16 files | std 8.77e-05 relative | mel max **0.00e+00** (identical: 1.595499158) |
| per-N constants | N3 0.174191112 ±1.0e-06 · N5 0.174192130 ±1.3e-06 · N7 0.174193403 ±6.2e-07 | identical across all N |
| **leak** `max\|Δ\|` | **0.000e+00** | **0.000e+00** |
| genuine deviant − standard, same frame | 4.393e-01 | 5.337e-01 |
| **leak / signal** | **0** | **0** |

Two different reasons for the zero, worth keeping straight:

- **Whisper is immune structurally.** Tones are synthesised at a fixed 94 dB SPL reference, so the
  loudest mel bin is the same regardless of tone count or SOA. The clamp reference and
  `_silence_value` are therefore identical across stimuli, and the leak vanishes by construction.
- **wav2vec2 is zero because the difference is below float32 resolution.** An 8.8e-05 relative
  change in `std` leaves the normalised waveform bit-identical after rounding. True for these
  stimuli; **not a structural guarantee** — it depends on the constants being close.

## 3. Why the MMN metric is protected

Scanned across the other stimulus sets (constants only, no model):

| set | dirs | `std` relative spread | peak relative spread |
|---|---|---|---|
| `mmn_stimuli_novel{,_wav2vec2}` | 1806 | 1.21e-03 | 3.05e-05 |
| `mmn_stimuli_soafix_wav2vec2` | 48 | **7.27e-01** (0.174 → 0.386) | 0.00e+00 |

The soafix set is the one place the constant genuinely moves — a **2.2× spread**, because
different SOAs pack different numbers of tones into a fixed window. But it moves **between**
method dirs, not within them, and the standard sits inside its own method's range:

```
method_09  std 0.174179–0.174194   standard 0.174179
method_10  std 0.386468–0.386474   standard 0.386468
method_12  std 0.334691–0.334695   standard 0.334691
```

`deviant − standard` is always computed **within** one method, so the constants cancel.
Worst within-method spread observed: 8.77e-05.

**The only exposure** would be comparing raw feature magnitudes *across* methods with different
SOAs, and then only for wav2vec2 — Whisper's peak is a constant 1.0 across the whole soafix set.
If such a comparison is ever made, re-run the probe with `--stim_dir` pointed at that set first.

## 4. For the methods section

> Delta-T features are causal: the feature stored for time `t` is computed from a forward pass on
> the stimulus truncated at `t`, and is bit-identical under perturbation of all later audio. The
> sole dependence on post-`t` input is through full-window normalisation constants (wav2vec2:
> waveform mean/std; Whisper: the log-mel clamp reference), held fixed across `t` to avoid
> degenerate statistics at early frames. This was measured at 0.000e+00 for the MMN stimuli in
> both model families and ≈4e-6 on continuous speech, against a deviant-minus-standard signal of
> ≈0.44–0.53 at the same frame.

## 5. transformers 5.x

The July wav2vec2 features were extracted under transformers 4.x; this checkout has **5.15.0**
(torch 2.12.0+cu126). `check_wav2vec2_api.py` verifies the whole contract the extractor relies on:

- `transform.conv_stride == 320` (⇒ 20 ms frames, matching the 50 Hz EEG grid) ✅
- `do_normalize` (True) and `norm_eps` (1e-07) present ✅
- all 12 `encoder.layers.N` in `wav2vec2_medium_layers.json` resolve as module names ✅
- 499 frames for a 10 s window; 12 tensors returned; `d_model` 768 ✅
- causality of the stored feature: `max|Δ| = 0.000e+00` post-cut, `5.20e-01` pre-cut ✅

**No version pin required.** The `UNEXPECTED` keys on load (`quantizer.*`, `project_q.*`,
`project_hid.*`) are the wav2vec2 *pretraining* heads, correctly discarded by `Wav2Vec2Model`
for feature extraction — expected, not an error.

## 6. Re-running

```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
export HF_HOME=/work/upschrimpf1/mehrer/.cache/huggingface   # keep HF off /home
./.venv/bin/python scripts/check_wav2vec2_api.py                      # API + causality
./.venv/bin/python scripts/check_mmn_norm_leak.py                     # wav2vec2, method_09
./.venv/bin/python scripts/check_mmn_norm_leak.py --model whisper-tiny
./.venv/bin/python scripts/check_mmn_norm_leak.py --stim_dir <set>    # any new stimulus set
```

Weights cache to `cache/model_weights` (on `/work`); the script refuses to run if the cache
resolves under `$HOME`. Run any new stimulus set through §A of the leak probe before trusting a
cross-condition magnitude comparison on it — that part needs no model and takes seconds.
