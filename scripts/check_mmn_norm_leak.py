#!/usr/bin/env python
"""Quantify the full-window normalisation leak on REAL MMN tone-train stimuli.

Both delta-t paths derive their normalisation from the WHOLE window, before truncation:
  * wav2vec2: `_wav2vec2_norm_stats` -> (mean, std) over all samples.
  * whisper : `log_mel_spectrogram` clamps at `log_spec.max() - 8` (a full-clip max), and
              `_silence_value` = mel_full.max() - 2.
So feature[t] depends on post-t audio through 1-2 scalars. On speech that is ~1e-6 and
irrelevant (see check_wav2vec2_api.py step 6). The worry is MMN: if trains with different
N (3/5/7 tones) or standard-vs-deviant differ in total energy, they get *different*
normalisation constants, and an across-condition contrast partly compares those constants
rather than the model's response.

This measures it three ways, cheapest first:

  A. Constants only, no model. Spread of (mean, std) / mel-max across a stimulus dir.
     If the generator holds energy constant across N, the leak is zero by construction
     and B/C are a formality.
  B. Leak at the feature level. Take ONE waveform; extract frame t twice -- once with its
     own constants, once with another stimulus's constants. Audio identical, so the whole
     delta is the leak.
  C. Leak vs signal. Compare that delta to the genuine deviant-minus-standard difference
     at the same frame. This ratio is the number that matters.

Run (already on a compute node):
    cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
    export HF_HOME=/work/upschrimpf1/mehrer/.cache/huggingface
    ./.venv/bin/python scripts/check_mmn_norm_leak.py                      # wav2vec2, method_09
    ./.venv/bin/python scripts/check_mmn_norm_leak.py --model whisper-tiny
    ./.venv/bin/python scripts/check_mmn_norm_leak.py --stim_dir <other set>   # novel / soafix

Informational: it prints numbers and a verdict, and always exits 0.
"""
from __future__ import annotations
import argparse, json, wave
from pathlib import Path

import numpy as np
import torch

SIGF = Path("/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs")


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000, f"{p} is not 16 kHz"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (x.astype(np.float32) / 32768.0)


def pick(files, tag):
    hit = [f for f in files if tag in f.name]
    return hit[0] if hit else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-medium")
    ap.add_argument("--stim_dir", default=None,
                    help="default: mmn_stimuli_wav2vec2/method_09 (wav2vec2) or mmn_stimuli/method_09 (whisper)")
    ap.add_argument("--model_cache_dir", default="cache/model_weights")
    a = ap.parse_args()

    is_w2v = a.model.startswith("wav2vec2-")
    stim = Path(a.stim_dir) if a.stim_dir else SIGF / (
        "mmn_stimuli_wav2vec2/method_09" if is_w2v else "mmn_stimuli/method_09")
    files = sorted(stim.glob("*.wav"))
    if not files:
        print(f"No wavs in {stim}"); return 0
    print(f"\nmodel     : {a.model}")
    print(f"stimuli   : {stim}  ({len(files)} files)\n")

    from mbs.extraction.extract_features_delta_t import (
        _wav2vec2_norm_stats, _truncate_waveform, _truncate_mel, _silence_value,
    )

    # ── A. constants only ────────────────────────────────────────────────────
    print("A. normalisation constants across the stimulus set (no model)")
    waves = {f.name: read_wav(f) for f in files}
    if is_w2v:
        consts = {n: _wav2vec2_norm_stats(x, True, 1e-7) for n, x in waves.items()}
        stds = np.array([c[1] for c in consts.values()])
        means = np.array([c[0] for c in consts.values()])
        print(f"   std  : min {stds.min():.9f}  max {stds.max():.9f}  "
              f"relative spread {(stds.max()-stds.min())/stds.mean():.2e}")
        print(f"   mean : min {means.min():+.9f}  max {means.max():+.9f}")
        spread = float((stds.max() - stds.min()) / stds.mean())
    else:
        import whisper
        mels = {n: whisper.log_mel_spectrogram(
                    whisper.pad_or_trim(torch.from_numpy(x).float()), n_mels=80)
                for n, x in waves.items()}
        maxes = np.array([float(m.max()) for m in mels.values()])
        print(f"   mel max (drives both the clamp and _silence_value = max-2):")
        print(f"          min {maxes.min():.9f}  max {maxes.max():.9f}  "
              f"relative spread {(maxes.max()-maxes.min())/abs(maxes.mean()):.2e}")
        spread = float((maxes.max() - maxes.min()) / max(abs(maxes.mean()), 1e-12))
    by_n = {}
    for n in ("N3", "N5", "N7"):
        v = [c[1] if is_w2v else float(mels[k].max())
             for k, c in (consts.items() if is_w2v else [(k, None) for k in mels]) if n in k]
        if v: by_n[n] = (float(np.mean(v)), float(np.std(v)))
    for n, (m, s) in by_n.items():
        print(f"   {n}: mean {m:.9f}  sd {s:.2e}")
    if spread < 1e-4:
        print(f"   -> spread {spread:.1e} is negligible; the across-N leak is ~zero BY CONSTRUCTION\n"
              f"      (the generator holds total energy constant across N).")

    # ── B/C. feature level ───────────────────────────────────────────────────
    dev = pick(files, "N3_var1_deviant") or files[0]
    other = pick(files, "N7_var1_deviant") or files[-1]
    std_f = pick(files, "standard")
    print(f"\nB. feature-level leak   probe={dev.name}   borrowed constants from={other.name}")

    from mbs.extraction.modeling.backbones.audio_models import load_model_audio
    from mbs.extraction.modeling.encoder_hooks import HookedEncoder
    backbone, transform = load_model_audio(a.model, model_cache_dir=a.model_cache_dir)
    backbone = backbone.eval()
    cfg = Path(f"configs/extraction/audio/{a.model.replace('-','_')}_layers.json")
    layers = [e["name"] for e in json.load(open(cfg))]
    layer = layers[len(layers) // 2]
    nodes = {f"backbone.{layer}": layer.replace(".", "-")}
    alias = nodes[f"backbone.{layer}"]
    hooked = HookedEncoder(backbone=backbone, feat_layers=nodes, include_output=False).eval()

    def frame(x_in):
        with torch.no_grad():
            return hooked(x_in[None])[alias][0]

    wa, wb = waves[dev.name], waves[other.name]
    if is_w2v:
        cs = transform.conv_stride
        T = len(wa) // cs
        t = int(0.85 * T)                       # near the eliciting tone, where MMN is read
        ma, sa = _wav2vec2_norm_stats(wa, transform.do_normalize, transform.norm_eps)
        mb, sb = _wav2vec2_norm_stats(wb, transform.do_normalize, transform.norm_eps)
        f_own = frame(torch.from_numpy(_truncate_waveform(wa, t, cs, ma, sa)))[t]
        f_bor = frame(torch.from_numpy(_truncate_waveform(wa, t, cs, mb, sb)))[t]
    else:
        import whisper
        mel_a, mel_b = mels[dev.name], mels[other.name]
        T = mel_a.shape[-1] // 2
        t = int(0.85 * T)
        f_own = frame(_truncate_mel(mel_a, t, _silence_value(mel_a)))[t]
        # same audio, but the silence fill / clamp reference taken from the other stimulus
        f_bor = frame(_truncate_mel(mel_a, t, _silence_value(mel_b)))[t]

    leak = float(torch.abs(f_own - f_bor).max())
    scale = float(torch.abs(f_own).max())
    print(f"   layer {layer}, frame t={t}/{T} ({100*t/T:.0f}% into the window)")
    print(f"   max|delta| from swapping constants = {leak:.3e}   "
          f"({100*leak/max(scale,1e-12):.4f}% of frame max |activation| {scale:.3f})")

    if std_f is not None:
        ws = waves[std_f.name]
        if is_w2v:
            ms, ss = _wav2vec2_norm_stats(ws, transform.do_normalize, transform.norm_eps)
            f_std = frame(torch.from_numpy(_truncate_waveform(ws, t, cs, ms, ss)))[t]
        else:
            mel_s = mels[std_f.name]
            f_std = frame(_truncate_mel(mel_s, t, _silence_value(mel_s)))[t]
        signal = float(torch.abs(f_own - f_std).max())
        print(f"\nC. leak vs signal")
        print(f"   genuine deviant-minus-standard at the same frame = {signal:.3e}")
        print(f"   leak / signal = {leak/max(signal,1e-12):.2e}")
        verdict = ("NEGLIGIBLE - the leak cannot account for any across-condition effect"
                   if leak / max(signal, 1e-12) < 1e-3 else
                   "NON-TRIVIAL - investigate before trusting across-N contrasts")
        print(f"   -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
