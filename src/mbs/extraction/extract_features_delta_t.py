"""Delta-T (causal) feature extraction for audio models.

For each stimulus and each time step t in 0..T-1:
    1. Build a truncated mel spectrogram: keep frames [0, 2*(t+1)), fill the rest
       with the per-stimulus silence value (matching Whisper's normalization).
    2. Run a single batched forward pass over batch_t truncated mels at once.
    3. Collect the encoder output at position t for each hooked layer.

The output HDF5 format is identical to extract_features.py (temporal mode), so
evaluate_features_temporal.py works without modification.

Parallelisation note: for large runs, parallelise at the stimulus level — launch
N processes each with a disjoint --wav_subset_indices range, then combine the
output directories (they share no state). See the handover doc for cluster commands.
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
import h5py
from tqdm.auto import tqdm

from mbs.core import str2bool
from mbs.extraction.data.datasets_audio import AudioSegmentDataset
from mbs.extraction.modeling.backbones.audio_models import load_whisper
from mbs.extraction.modeling.encoder_hooks import HookedEncoder


# ---------------------------------------------------------------------------
# Constants for Whisper 30 s audio
# ---------------------------------------------------------------------------
WHISPER_MEL_FRAMES = 3000   # mel frames for a 30 s clip (100 frames/s)
WHISPER_T = 1500            # encoder output bins  (stride-2 convs: 3000→1500)
# Encoder bin k covers mel frames [2k, 2*(k+1)) in the input spectrogram.


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Delta-T (causal) feature extraction for Whisper-family audio models."
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Audio model id, e.g. whisper-base.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory containing .wav stimulus files.")
    parser.add_argument("--target_feature_layers", type=str, required=True,
                        help="JSON file listing layer names (same format as extract_features.py).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write feature HDF5 files.")
    parser.add_argument("--window_duration", type=float, default=30.0,
                        help="Sliding-window duration in seconds.")
    parser.add_argument("--window_stride", type=float, default=10.0,
                        help="Sliding-window stride in seconds.")
    parser.add_argument("--batch_t", type=int, default=16,
                        help="Number of time-step truncations to run in one forward pass. "
                             "Larger values use more memory but run faster on GPU. "
                             "Use batch_t=1500 on a well-provisioned GPU to do one pass per stimulus.")
    parser.add_argument("--t_stride", type=int, default=1,
                        help="Sub-sample time bins: extract every t_stride-th bin. "
                             "Use t_stride=50 for a fast pilot (30 bins at 1-s resolution). "
                             "Use t_stride=1 for the full 1500-bin run.")
    parser.add_argument("--stim_start_idx", type=int, default=0,
                        help="Start at this stimulus index (for parallelising across processes).")
    parser.add_argument("--n_stimuli", type=int, default=0,
                        help="Process only N stimuli starting from stim_start_idx (0 = all remaining). For pilots.")
    parser.add_argument("--save_every", type=int, default=8,
                        help="Accumulate this many stimuli in memory before writing one HDF5 file.")
    parser.add_argument("--model_cache_dir", type=str, default="cache/model_weights",
                        help="Cache directory for model weights.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", type=str2bool, default=False,
                        help="Overwrite existing output files.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core Delta-T logic
# ---------------------------------------------------------------------------

def _silence_value(mel_full: torch.Tensor) -> float:
    """Fill value for silenced mel frames, matching Whisper's global normalization.

    Whisper normalises mel as: (log10(amp).clamp(floor=mel_max_raw-8) + 4) / 4
    For silence (amp→0, log10→-10), this evaluates to mel_full.max() - 2.0.
    """
    return float(mel_full.max()) - 2.0


def _truncate_mel(mel_full: torch.Tensor, t: int, silence_val: float) -> torch.Tensor:
    """Return mel with frames >= 2*(t+1) replaced by silence_val (in-place clone)."""
    mel_t = mel_full.clone()
    cut = 2 * (t + 1)
    if cut < mel_full.shape[-1]:
        mel_t[:, cut:] = silence_val
    return mel_t


def extract_delta_t(
    mel_full: torch.Tensor,      # [n_mels, 3000]  on CPU
    hooked_encoder: HookedEncoder,
    layer_aliases: list,         # aliases matching HookedEncoder feat_layers values
    t_values: list,              # time-bin indices to compute (subset of 0..T-1)
    batch_t: int,
    device: torch.device,
) -> dict:
    """
    Returns dict: alias -> np.ndarray  shape [T_out, d_model]  dtype float16.
    """
    silence_val = _silence_value(mel_full)
    T_out = len(t_values)
    accum = {alias: [] for alias in layer_aliases}

    for start in range(0, T_out, batch_t):
        batch_ts = t_values[start: start + batch_t]

        # Build a batch of truncated mel spectrograms
        batch = torch.stack(
            [_truncate_mel(mel_full, t, silence_val) for t in batch_ts]
        ).to(device)  # [B, n_mels, 3000]

        with torch.no_grad():
            feats = hooked_encoder(batch)   # dict alias -> [B, T_model, d]

        for alias in layer_aliases:
            arr = feats[alias]              # [B, T_model, d_model]
            for i, t in enumerate(batch_ts):
                vec = arr[i, t, :].cpu().float().numpy().astype(np.float16)
                accum[alias].append(vec)

    return {alias: np.stack(accum[alias], axis=0) for alias in layer_aliases}


# ---------------------------------------------------------------------------
# HDF5 I/O (same schema as extract_features.py temporal output)
# ---------------------------------------------------------------------------

def _write_batch(output_dir: Path, batch_idx: int, batch_size: int,
                 stim_start_idx: int,
                 stim_features: list, stim_ids: list, model_id: str,
                 backbone_source: str, target_layers: list, config: dict):
    """Write one HDF5 batch file from accumulated stimuli.

    Filename encodes stim_start_idx (global offset for this SLURM task) and
    batch_idx (within-task batch counter), ensuring uniqueness across parallel tasks.
    """
    path = output_dir / f"feats_delta_t-start_{stim_start_idx:05d}-batch_{batch_idx}-seed_42.h5"
    with h5py.File(path, "w") as hf:
        # stim_features: list of dicts  alias -> [T_out, d]
        layer_aliases = list(stim_features[0].keys())
        for alias in layer_aliases:
            arr = np.stack([sf[alias] for sf in stim_features], axis=0)  # [n_stim, T_out, d]
            hf.create_dataset(f"features/{alias}", data=arr, dtype=np.float16,
                              compression="gzip", compression_opts=4)
        hf.create_dataset("ids", data=np.array(stim_ids, dtype=h5py.string_dtype()))
        hf.attrs["model_id"] = model_id
        hf.attrs["backbone_source"] = backbone_source
        hf.attrs["target_feature_layers"] = json.dumps(target_layers)
        hf.attrs["extraction_mode"] = "delta_t"
        hf.attrs["config_json"] = json.dumps(config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load layer list
    with open(args.target_feature_layers) as f:
        layer_list_raw = json.load(f)
    target_layers = [entry["name"] for entry in layer_list_raw]  # e.g. ["blocks.0", ...]

    # Build HookedEncoder  (same construction as _create_audio_hook_feature_extractor)
    backbone, transform = load_whisper(args.model_id, model_cache_dir=args.model_cache_dir)
    backbone = backbone.to(device).eval()

    return_nodes = {f"backbone.{layer}": layer.replace(".", "-") for layer in target_layers}
    layer_aliases = list(return_nodes.values())  # e.g. ["blocks-0", "blocks-1", ...]

    hooked_encoder = HookedEncoder(backbone=backbone, feat_layers=return_nodes, include_output=False)
    hooked_encoder = hooked_encoder.to(device).eval()

    # Dataset  (transform=None → raw float32 waveform)
    wav_files = sorted(Path(args.data_root).glob("*.wav"))
    assert wav_files, f"No .wav files found in {args.data_root}"
    dataset = AudioSegmentDataset(
        wav_files=wav_files,
        window_duration=args.window_duration,
        stride=args.window_stride,
        target_sr=16000,
        transform=None,   # we need raw waveform for mel truncation
    )

    n_dataset = len(dataset)
    start = args.stim_start_idx
    end = n_dataset if args.n_stimuli <= 0 else min(start + args.n_stimuli, n_dataset)
    total = end - start

    t_values = list(range(0, WHISPER_T, args.t_stride))
    T_out = len(t_values)
    print(f"Stimuli: {total} (idx {start}..{end-1} of {n_dataset})  |  "
          f"Time bins: {T_out} (t_stride={args.t_stride})  |  batch_t: {args.batch_t}")
    print(f"Layers: {target_layers}")

    accum_feats = []
    accum_ids = []
    file_idx = 0
    t_total_start = time.time()

    for stim_idx in tqdm(range(start, end), desc="Stimuli"):
        waveform_np, stim_id = dataset[stim_idx]

        # Compute full mel once for this stimulus
        mel_full = transform(waveform_np).to(device)  # [n_mels, 3000]

        t_stim = time.time()
        feats = extract_delta_t(
            mel_full=mel_full.cpu(),  # keep on CPU for cloning; batches sent to device inside
            hooked_encoder=hooked_encoder,
            layer_aliases=layer_aliases,
            t_values=t_values,
            batch_t=args.batch_t,
            device=device,
        )
        elapsed_stim = time.time() - t_stim

        accum_feats.append(feats)
        accum_ids.append(stim_id)

        # Flush to disk when we've collected save_every stimuli or reached the end.
        # Compare against end-1 (not total-1) so this works when stim_start_idx > 0.
        if len(accum_feats) == args.save_every or stim_idx == end - 1:
            _write_batch(
                output_dir=output_dir,
                batch_idx=file_idx,
                batch_size=args.save_every,
                stim_start_idx=args.stim_start_idx,
                stim_features=accum_feats,
                stim_ids=accum_ids,
                model_id=args.model_id,
                backbone_source="audio",
                target_layers=target_layers,
                config=vars(args),
            )
            file_idx += 1
            accum_feats = []
            accum_ids = []

        # Progress timing
        elapsed_total = time.time() - t_total_start
        rate = (stim_idx + 1) / elapsed_total
        eta = (total - stim_idx - 1) / rate if rate > 0 else float("inf")
        tqdm.write(f"  stim {stim_idx+1}/{total}: {elapsed_stim:.1f}s  "
                   f"ETA {eta/3600:.1f}h  ({rate*3600:.0f} stim/hr)")

    print(f"\nDone. {file_idx} HDF5 files written to {output_dir}")
    print(f"Feature shape per stimulus: [T_out={T_out}, d_model] per layer")


def cli():
    args = parse_args()
    print("Arguments:", vars(args))
    main(args)


if __name__ == "__main__":
    cli()
