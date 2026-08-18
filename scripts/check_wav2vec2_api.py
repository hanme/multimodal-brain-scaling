#!/usr/bin/env python
"""Smoke-test the wav2vec2 path against the installed transformers version.

Sophie ran the July wav2vec2 extraction on transformers 4.x; this checkout now has 5.x.
This exercises the REAL code path (load_wav2vec2 -> HookedEncoder -> the delta-t helpers)
on one synthetic 10 s waveform, so a breaking API change surfaces in ~1 min instead of
inside a 20-task array job.

Run on a compute node (no SLURM script needed). If you are already on one, e.g. via
    Sinteract -t 10:00:00 -c 20 -m 80G
then just:
    cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
    export HF_HOME=/work/upschrimpf1/mehrer/.cache/huggingface
    ./.venv/bin/python scripts/check_wav2vec2_api.py

Calling ./.venv/bin/python by path avoids any conda `base` env on PATH; no `source env.sh`
needed for this test (it imports nothing that requires the gcc module).

From a login node instead:
    srun -p standard -c 4 --mem 16G -t 00:20:00 --pty ./.venv/bin/python scripts/check_wav2vec2_api.py

Downloads ~360 MB on first run (facebook/wav2vec2-base) into cache/model_weights, i.e. on
/work -- NOT into ~/.cache on /home. Add --model wav2vec2-large for the 1.2 GB one.
Exit code 0 = all green.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

OK = True
def chk(label: str, cond: bool, extra: str = "") -> bool:
    global OK
    OK &= bool(cond)
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}" + (f" — {extra}" if extra else ""), flush=True)
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-medium",
                    choices=["wav2vec2-medium", "wav2vec2-large"])
    ap.add_argument("--window_s", type=float, default=10.0, help="wav2vec2 window used for D2")
    # Repo convention: cache/model_weights, relative to the repo root -> lives on /work.
    # Do NOT leave this None: HF would fall back to ~/.cache/huggingface on /home.
    ap.add_argument("--model_cache_dir", default="cache/model_weights")
    args = ap.parse_args()

    import transformers
    cache = Path(args.model_cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if str(cache).startswith(str(Path.home())):
        print(f"REFUSING: cache dir {cache} is under $HOME. Pass --model_cache_dir on /work.")
        return 1
    args.model_cache_dir = str(cache)
    print(f"\ntransformers {transformers.__version__} | torch {torch.__version__}")
    print(f"weights cache: {cache}\n")

    # ── 1. loader ────────────────────────────────────────────────────────────
    print("1. load_wav2vec2 (the repo's own loader)")
    t0 = time.time()
    from mbs.extraction.modeling.backbones.audio_models import load_model_audio
    backbone, transform = load_model_audio(args.model, model_cache_dir=args.model_cache_dir)
    backbone = backbone.eval()
    print(f"     loaded in {time.time()-t0:.0f}s")
    chk("returns (backbone, transform)", backbone is not None and transform is not None)

    # ── 2. the three attributes the extractor reads off the transform ────────
    print("\n2. transform attributes the delta-t extractor depends on")
    cs = getattr(transform, "conv_stride", None)
    chk("transform.conv_stride == 320", cs == 320, f"got {cs} (320 => 50 Hz frames, matching the EEG grid)")
    chk("transform.do_normalize present", hasattr(transform, "do_normalize"),
        f"= {getattr(transform, 'do_normalize', None)}")
    chk("transform.norm_eps present", hasattr(transform, "norm_eps"),
        f"= {getattr(transform, 'norm_eps', None)}")

    # ── 3. layer paths in the config still resolve as module names ───────────
    print("\n3. layer config vs actual module names")
    cfg_path = Path(f"configs/extraction/audio/{args.model.replace('-', '_')}_layers.json")
    target_layers = [e["name"] for e in json.load(open(cfg_path))]
    modules = {n for n, _ in backbone.named_modules()}
    missing = [l for l in target_layers if f"backbone.{l}" not in modules]
    chk(f"all {len(target_layers)} layers in {cfg_path.name} exist as modules",
        not missing, f"missing: {missing[:4]}" if missing else f"e.g. {target_layers[0]}")

    # ── 4. HookedEncoder, built exactly as the extractor builds it ───────────
    print("\n4. HookedEncoder forward on a full window")
    from mbs.extraction.modeling.encoder_hooks import HookedEncoder
    return_nodes = {f"backbone.{l}": l.replace(".", "-") for l in target_layers}
    aliases = list(return_nodes.values())
    hooked = HookedEncoder(backbone=backbone, feat_layers=return_nodes, include_output=False).eval()

    n_samples = int(args.window_s * 16000)
    rng = np.random.default_rng(0)
    wav = (0.05 * rng.standard_normal(n_samples)).astype(np.float32)

    from mbs.extraction.extract_features_delta_t import (
        _infer_wav2vec2_frame_count, _wav2vec2_norm_stats, _truncate_waveform,
    )
    T = _infer_wav2vec2_frame_count(wav, hooked, aliases[0], torch.device("cpu"))
    expected = n_samples // 320
    chk(f"frame count for {args.window_s:g}s", abs(T - expected) <= 1,
        f"got {T}, expected ~{expected} (= {n_samples}/320)")
    with torch.no_grad():
        feats = hooked(torch.from_numpy(wav)[None])
    chk("every layer alias returned", all(a in feats for a in aliases), f"{len(feats)} tensors")
    d = feats[aliases[0]].shape[-1]
    chk("hidden size", d in (768, 1024), f"d_model={d}")

    # ── 5. causality of what the extractor ACTUALLY stores ───────────────────
    # wav2vec2's encoder is a BIDIRECTIONAL transformer: frame t in a full-window pass
    # attends over the whole window. That is exactly why delta-t exists. The extractor
    # never reads frame t from a full-window pass -- for each t it runs a pass on the
    # waveform truncated at t and reads arr[t] from THAT pass (extract_delta_t_waveform).
    # So the property to test is whether that stored vector depends on audio after the cut.
    print("\n5. causality of the stored delta-t feature")
    cs, t_probe = transform.conv_stride, T // 2
    cut = (t_probe + 1) * cs

    wav_b = wav.copy()                       # identical up to `cut`, different after
    wav_b[cut:] = (0.05 * rng.standard_normal(n_samples - cut)).astype(np.float32)

    mean, std = _wav2vec2_norm_stats(wav, transform.do_normalize, transform.norm_eps)
    x_a = _truncate_waveform(wav,   t_probe, cs, mean, std)
    x_b = _truncate_waveform(wav_b, t_probe, cs, mean, std)
    chk("truncation zeroes samples >= (t+1)*conv_stride",
        np.allclose(x_a[cut:], np.float32((0.0 - mean) / std)),
        f"cut at sample {cut} of {n_samples}")
    chk("waveforms differing ONLY after the cut become identical inputs",
        np.array_equal(x_a, x_b))

    def frame_of(x):
        with torch.no_grad():
            return hooked(torch.from_numpy(np.asarray(x, np.float32))[None])[aliases[0]][0, t_probe]
    fa, fb = frame_of(x_a), frame_of(x_b)
    chk("stored frame t IDENTICAL when post-cut audio changes", torch.equal(fa, fb),
        f"max|delta| = {float(torch.abs(fa-fb).max()):.2e} — the causality guarantee")

    wav_c = wav.copy(); wav_c[: cut // 2] += 0.05      # sanity: truncation is not a no-op
    dc = float(torch.abs(fa - frame_of(_truncate_waveform(wav_c, t_probe, cs, mean, std))).max())
    chk("stored frame t DOES change when pre-cut audio changes", dc > 1e-3, f"max|delta| = {dc:.2e}")

    # ── 6. the one acknowledged leak: normalization stats come from the FULL window ──
    # _wav2vec2_norm_stats uses the whole window's mean/var, so those two scalars do see
    # the future. Deliberate (mirrors Whisper's full-clip mel normalization), documented in
    # the module comment. Quantify it; do not fail on it.
    print("\n6. magnitude of the acknowledged normalization leak (informational)")
    mb, sb = _wav2vec2_norm_stats(wav_b, transform.do_normalize, transform.norm_eps)
    d_norm = float(torch.abs(fa - frame_of(_truncate_waveform(wav_b, t_probe, cs, mb, sb))).max())
    scale = float(torch.abs(fa).max())
    print(f"     mean {mean:+.6f} -> {mb:+.6f} | std {std:.6f} -> {sb:.6f}")
    print(f"     max|delta| on frame t = {d_norm:.2e}  ({100*d_norm/max(scale,1e-12):.2f}% of frame max |activation|)")
    print("     -> feature[t] sees post-t audio ONLY through these two scalars.")

    print("\n" + ("ALL GREEN — wav2vec2 works under this transformers version"
                  if OK else "FAILURE — do not launch the array job; see above"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
