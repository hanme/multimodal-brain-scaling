"""Transcribe an audio file (mp3/wav/m4a/…) to Markdown with openai-whisper.

Uses the same `whisper` package the feature-extraction pipeline uses (full encoder+decoder, so
`model.transcribe()` works directly). Audio loading is ffmpeg-free by default: if `ffmpeg` is on
PATH we let whisper use it, otherwise we decode with `soundfile` (libsndfile ≥1.1 reads mp3) and
resample to 16 kHz mono ourselves, then hand whisper a numpy array.

  python scripts/transcribe_audio.py                       # newest audio file in cache/, model=medium
  python scripts/transcribe_audio.py --audio cache/talk.mp3 --model small --out cache/talk.md
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

SR = 16000  # whisper's required sample rate
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma")


def newest_audio(folder: Path) -> Path:
    cands = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    assert cands, f"no audio files ({', '.join(AUDIO_EXTS)}) in {folder}"
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Return a float32 mono 16 kHz waveform. Prefer whisper's ffmpeg path; else soundfile."""
    if shutil.which("ffmpeg"):
        import whisper
        return whisper.load_audio(str(path))            # ffmpeg → float32 16 kHz mono
    # ffmpeg-free fallback: soundfile decode (incl. mp3) + resample
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)                               # → mono
    if sr != SR:
        try:                                            # high-quality if scipy is available
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), SR)
            wav = resample_poly(wav, SR // g, int(sr) // g).astype(np.float32)
        except Exception:                               # crude linear fallback (no scipy)
            n_out = int(round(len(wav) * SR / sr))
            wav = np.interp(np.linspace(0, len(wav), n_out, endpoint=False),
                            np.arange(len(wav)), wav).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


def to_markdown(result: dict, audio: Path, model_id: str, n_samples: int) -> str:
    dur = n_samples / SR
    lang = result.get("language", "?")
    lines = [
        f"# Transcript — {audio.name}",
        "",
        f"- **Model:** whisper-{model_id}",
        f"- **Source:** `{audio}`",
        f"- **Duration:** {dur/60:.1f} min ({dur:.0f} s)",
        f"- **Detected language:** {lang}",
        "",
        "## Full text",
        "",
        result["text"].strip(),
        "",
        "## Timestamped segments",
        "",
    ]
    for s in result.get("segments", []):
        t0, t1 = float(s["start"]), float(s["end"])
        lines.append(f"- `[{t0//60:02.0f}:{t0%60:05.2f} → {t1//60:02.0f}:{t1%60:05.2f}]` {s['text'].strip()}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Transcribe an audio file to Markdown via openai-whisper.")
    p.add_argument("--audio", default="", help="audio file; default = newest in --audio_dir")
    p.add_argument("--audio_dir", default="cache", help="folder to scan when --audio is omitted")
    p.add_argument("--model", default="medium",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--out", default="", help="output .md; default = <audio>.md next to the audio")
    p.add_argument("--model_cache_dir", default="cache/whisper_models",
                   help="where whisper downloads/loads the model weights")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--language", default="", help="force a language (e.g. 'en'); '' = auto-detect")
    args = p.parse_args()

    import torch
    import whisper

    audio = Path(args.audio) if args.audio else newest_audio(Path(args.audio_dir))
    assert audio.exists(), f"audio not found: {audio}"
    out = Path(args.out) if args.out else audio.with_suffix(".md")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading whisper-{args.model} on {device} (cache={args.model_cache_dir}) …")
    model = whisper.load_model(args.model, device=device, download_root=args.model_cache_dir)

    print(f"Decoding {audio} (ffmpeg={'yes' if shutil.which('ffmpeg') else 'no → soundfile'}) …")
    wav = load_audio_16k_mono(audio)

    print(f"Transcribing {len(wav)/SR/60:.1f} min … (segments stream below with [mm:ss] timestamps "
          f"→ compare to total to gauge progress)", flush=True)
    # verbose=True streams each decoded segment to stdout (the .out log) as "[start --> end] text",
    # so progress is visible by timestamp; cleaner in SLURM logs than the default tqdm stderr bar.
    kw = {"fp16": device == "cuda", "verbose": True}
    if args.language:
        kw["language"] = args.language
    result = model.transcribe(wav, **kw)

    out.write_text(to_markdown(result, audio, args.model, len(wav)))
    print(f"wrote {out}  ({len(result.get('segments', []))} segments, lang={result.get('language','?')})")


if __name__ == "__main__":
    main()
