#!/usr/bin/env python3
"""
Phase 0aa: Generate Audio Stimuli for MMN Cross-Validation Pipeline

Generates pure-tone audio sequences modelling the oddball paradigm for mismatch
negativity (MMN) experiments. For each frequency-change row in a metadata CSV,
produces one standard stimulus and multiple deviant stimulus variations across
configurable trial levels, for each of four target audio models. A counterbalanced
copy of each set (standard/deviant frequencies swapped) is also produced.

Alongside the WAV files, a metadata CSV is written for each configuration
(regular and counterbalanced) documenting every generated stimulus: its filename,
duration class, source parameters, trial type, and full tone sequence.

All stimuli within a metadata row share identical temporal alignment at every
tone slot. The script is metadata-agnostic: it accepts any CSV with the required
columns, so the same entry point serves both the literature and stimuli_search
pipelines. Parallelization is supported via multiprocessing within a job and via
SLURM array chunking across jobs.

Usage:
    python 00aa_generate_audio_stimuli.py \\
        --metadata_csv metadata/frequency_metadata.csv \\
        --output_dir data/audio_inputs_literature

    python 00aa_generate_audio_stimuli.py \\
        --metadata_csv stimuli_search_metadata.csv \\
        --output_dir data/stimuli_search_audio \\
        --chunk_idx $SLURM_ARRAY_TASK_ID --n_chunks 16

Output structure:
    {output_dir}/
    ├── audio_outputs_regular/{model}/method_{ID:02d}_*.wav
    ├── audio_outputs_regular_metadata/metadata.csv
    ├── audio_outputs_counter/{model}/method_{ID:02d}_*.wav
    └── audio_outputs_counter_metadata/metadata.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Tuple, List
from multiprocessing import Pool, cpu_count
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000          # Hz — universal sample rate for all models
RAMP_DURATION_MS = 5         # ms — linear rise/fall envelope per tone
REFERENCE_DB = 94            # dB SPL at which a full-scale sine has unit amplitude

MODEL_DURATIONS = {          # Total sequence length per model (ms)
    'whisper': 30000,
    'wav2vec2': 10000,
    'vggish': 10000,
    'ast': 10000,
}

# Unique durations and which models map to each (avoids duplicate metadata rows)
DURATION_GROUPS = {
    30: ['whisper'],
    10: ['wav2vec2', 'vggish', 'ast'],
}

TRIAL_LEVELS = [3, 5, 7]     # N values: standard tones between final two deviants
NUM_VARIATIONS = 5           # Stochastic deviant variations per trial level

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ms_to_samples(duration_ms: float) -> int:
    """
    Convert a duration in milliseconds to an integer sample count.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        Number of samples (integer floor).
    """
    return int(duration_ms * SAMPLE_RATE / 1000)


def db_to_amplitude(db_spl: float) -> float:
    """
    Convert a dB SPL value to a linear amplitude scaling factor.

    Args:
        db_spl: Sound pressure level in decibels.

    Returns:
        Linear amplitude multiplier relative to REFERENCE_DB.
    """
    return 10 ** ((db_spl - REFERENCE_DB) / 20)


def sequence_to_string(sequence: List[str]) -> str:
    """
    Convert a tone-identity sequence to a compact comma-separated S/D string.

    Args:
        sequence: List of "standard"/"deviant" labels.

    Returns:
        String like "S,S,D,S,S,S,D" for logging and metadata output.
    """
    return ",".join("S" if s == "standard" else "D" for s in sequence)


# =============================================================================
# AUDIO SYNTHESIS
# =============================================================================

def generate_tone(frequency: float, duration_ms: float,
                  intensity_db: float) -> np.ndarray:
    """
    Synthesize a single pure tone with linear onset/offset ramps.

    Generates a sine wave at the given frequency and intensity, applies symmetric
    linear amplitude ramps of RAMP_DURATION_MS at the start and end, and clips
    the result to [-1, 1].

    Args:
        frequency:    Tone frequency in Hz.
        duration_ms:  Tone duration in milliseconds.
        intensity_db: Tone intensity in dB SPL.

    Returns:
        1-D numpy array of audio samples, clipped to [-1.0, 1.0].
    """
    duration_samples = ms_to_samples(duration_ms)
    ramp_samples = ms_to_samples(RAMP_DURATION_MS)

    # For very short tones, shorten the ramp so it doesn't exceed half the tone
    if duration_samples < 2 * ramp_samples:
        ramp_samples = duration_samples // 2

    t = np.arange(duration_samples) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * frequency * t)
    tone *= db_to_amplitude(intensity_db)

    if ramp_samples > 0:
        tone[:ramp_samples] *= np.linspace(0, 1, ramp_samples)
        tone[-ramp_samples:] *= np.linspace(1, 0, ramp_samples)

    return np.clip(tone, -1.0, 1.0)


def generate_silence(duration_ms: float) -> np.ndarray:
    """
    Generate a block of digital silence.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        1-D float32 numpy array of zeros.
    """
    return np.zeros(ms_to_samples(duration_ms), dtype=np.float32)


# =============================================================================
# TONE-SLOT GRID
# =============================================================================

def compute_tone_slots(total_duration_ms: float, tone_duration_ms: float,
                       isi_ms: float) -> Tuple[int, float]:
    """
    Determine how many equal-spaced tone slots fit within a model's time window.

    The audio layout is:
        [leftover silence][ISI][tone 1][ISI][tone 2][ISI] ... [tone K][ISI]

    K tones are placed, each preceded and followed by one ISI. Any remaining
    time is prepended as additional leading silence.

    Args:
        total_duration_ms: Model window length in ms (e.g. 10000 or 30000).
        tone_duration_ms:  Shared tone duration in ms.
        isi_ms:            Shared inter-stimulus interval in ms.

    Returns:
        (K, leftover_ms): number of tone slots and extra leading silence in ms.
    """
    cycle = tone_duration_ms + isi_ms
    K = int((total_duration_ms - isi_ms) / cycle)
    leftover = total_duration_ms - isi_ms - K * cycle
    return K, leftover


def validate_tone_slots(K: int, method_id: int, model_name: str) -> None:
    """
    Assert that K is large enough to accommodate every configured trial level.

    Each trial level N requires a fixed suffix of N + 2 tone slots. If K is too
    small, an error is raised so the user can remove the offending row from the
    metadata.

    Args:
        K:          Number of available tone slots.
        method_id:  Metadata row identifier (for error reporting).
        model_name: Model name string (for error reporting).

    Raises:
        ValueError: If K < N + 2 for any trial level N.
    """
    for N in TRIAL_LEVELS:
        required = N + 2
        if K < required:
            raise ValueError(
                f"Insufficient tone slots for method_id={method_id}, "
                f"model={model_name}: K={K} but trial level N={N} requires "
                f"at least {required} slots. Remove this row from the metadata."
            )


# =============================================================================
# SEQUENCE GENERATION
# =============================================================================

def generate_standard_sequence(K: int) -> List[str]:
    """
    Create the tone-identity sequence for a standard (all-same) stimulus.

    Args:
        K: Total number of tone slots.

    Returns:
        List of K "standard" labels.
    """
    return ["standard"] * K


def generate_deviant_sequence(K: int, N: int, v: int,
                              method_id: int) -> List[str]:
    """
    Create the tone-identity sequence for one deviant stimulus variation.

    In a deviant sequence the background (frequent) tone is the deviant
    frequency and the rare (oddball) tone is the standard frequency. The
    sequence has two parts:
      - A fixed suffix of length N + 2: [standard, deviant x N, standard].
        The final tone is always "standard", matching the standard stimulus.
      - A stochastic prefix filling the remaining K - (N + 2) slots, where each
        position is independently "standard" (the rare oddball) with probability
        1/(N+1) and "deviant" (the frequent background) otherwise.

    Randomness is seeded deterministically from (method_id, N, v) so that
    identical sequences are reproduced on every run.

    Args:
        K:         Total number of tone slots.
        N:         Trial level (deviant tones between the final two standard oddball tones).
        v:         Variation index (1-based).
        method_id: Metadata row identifier (used for seeding).

    Returns:
        List of K strings, each "standard" or "deviant".
    """
    seed = method_id * 10000 + N * 100 + v
    rng = np.random.RandomState(seed)

    suffix = ["standard"] + ["deviant"] * N + ["standard"]
    prefix_len = K - (N + 2)

    p_standard = 1.0 / (N + 1)
    prefix = [
        "standard" if rng.random() < p_standard else "deviant"
        for _ in range(prefix_len)
    ]

    return prefix + suffix


# =============================================================================
# WAVEFORM CONSTRUCTION
# =============================================================================

def build_audio_from_sequence(sequence: List[str], standard_freq: float,
                              deviant_freq: float, tone_duration_ms: float,
                              isi_ms: float, intensity_db: float,
                              total_duration_ms: float,
                              leftover_ms: float) -> np.ndarray:
    """
    Render a tone-identity sequence into a complete audio waveform.

    Maps each "standard"/"deviant" label to the appropriate frequency,
    synthesizes each tone with shared duration and intensity, inserts ISI
    silence between tones, prepends the leading silence block, enforces the
    exact target sample count, and normalizes to peak amplitude 1.0.

    Args:
        sequence:          List of K "standard"/"deviant" labels.
        standard_freq:     Frequency for "standard" tones (Hz).
        deviant_freq:      Frequency for "deviant" tones (Hz).
        tone_duration_ms:  Shared tone duration (ms).
        isi_ms:            Shared inter-stimulus interval (ms).
        intensity_db:      Shared intensity (dB SPL).
        total_duration_ms: Target total waveform duration (ms).
        leftover_ms:       Extra leading silence beyond the initial ISI (ms).

    Returns:
        Normalized float32 numpy array of length ms_to_samples(total_duration_ms).
    """
    segments = []

    # Leading silence: leftover + mandatory initial ISI
    segments.append(generate_silence(leftover_ms + isi_ms))

    # Tones and trailing ISIs
    for symbol in sequence:
        freq = standard_freq if symbol == "standard" else deviant_freq
        segments.append(generate_tone(freq, tone_duration_ms, intensity_db))
        segments.append(generate_silence(isi_ms))

    full_sequence = np.concatenate(segments)

    # Enforce exact target length (handles rounding of a few samples)
    target_samples = ms_to_samples(total_duration_ms)
    if len(full_sequence) > target_samples:
        full_sequence = full_sequence[:target_samples]
    elif len(full_sequence) < target_samples:
        padding = np.zeros(target_samples - len(full_sequence), dtype=np.float32)
        full_sequence = np.concatenate([full_sequence, padding])

    # Peak-normalize
    max_abs = np.max(np.abs(full_sequence))
    if max_abs > 0:
        full_sequence = full_sequence / max_abs

    return full_sequence.astype(np.float32)


# =============================================================================
# PER-ROW PROCESSING
# =============================================================================

def process_row_model_config(row: pd.Series, model_name: str,
                             config_output_dir: Path,
                             is_counterbalanced: bool) -> List[Dict[str, Any]]:
    """
    Generate all WAV files for one metadata row, one model, one configuration,
    and return metadata records describing each generated file.

    Produces a single standard WAV and (len(TRIAL_LEVELS) x NUM_VARIATIONS)
    deviant WAVs, written to {config_output_dir}/{model_name}/.

    Args:
        row:                 Pandas Series for one metadata row.
        model_name:          One of 'whisper', 'wav2vec2', 'vggish', 'ast'.
        config_output_dir:   Root directory for this configuration
                             (audio_outputs_regular or audio_outputs_counter).
        is_counterbalanced:  If True, swap standard and deviant frequencies.

    Returns:
        List of dicts, one per generated WAV, each containing metadata fields.
    """
    tone_duration_ms = float(row['standard_dur'])
    isi_ms = float(row['standard_isi'])
    intensity_db = float(row['standard_int'])
    method_id = int(row['method_id'])

    if not is_counterbalanced:
        standard_freq = float(row['standard_freq'])
        deviant_freq = float(row['deviant_freq'])
    else:
        standard_freq = float(row['deviant_freq'])
        deviant_freq = float(row['standard_freq'])

    total_duration_ms = MODEL_DURATIONS[model_name]
    duration_s = total_duration_ms // 1000
    K, leftover_ms = compute_tone_slots(total_duration_ms, tone_duration_ms, isi_ms)
    validate_tone_slots(K, method_id, model_name)

    model_dir = config_output_dir / model_name
    records = []

    # Shared fields for every record from this row/model/config
    shared = {
        'duration_s': duration_s,
        'method_id': method_id,
        'standard_freq': standard_freq,
        'deviant_freq': deviant_freq,
        'tone_duration_ms': tone_duration_ms,
        'isi_ms': isi_ms,
        'intensity_db': intensity_db,
    }

    # Standard stimulus
    standard_seq = generate_standard_sequence(K)
    standard_audio = build_audio_from_sequence(
        standard_seq, standard_freq, deviant_freq,
        tone_duration_ms, isi_ms, intensity_db,
        total_duration_ms, leftover_ms
    )
    standard_fname = f"method_{method_id:02d}_standard.wav"
    sf.write(str(model_dir / standard_fname), standard_audio, SAMPLE_RATE)

    records.append({
        **shared,
        'filename': standard_fname,
        'trial_type': 'standard',
        'N': '',
        'variation': 1,
        'sequence': sequence_to_string(standard_seq),
    })

    # Deviant stimuli
    for N in TRIAL_LEVELS:
        for v in range(1, NUM_VARIATIONS + 1):
            deviant_seq = generate_deviant_sequence(K, N, v, method_id)
            deviant_audio = build_audio_from_sequence(
                deviant_seq, standard_freq, deviant_freq,
                tone_duration_ms, isi_ms, intensity_db,
                total_duration_ms, leftover_ms
            )
            deviant_fname = f"method_{method_id:02d}_N{N}_var{v}_deviant.wav"
            sf.write(str(model_dir / deviant_fname), deviant_audio, SAMPLE_RATE)

            records.append({
                **shared,
                'filename': deviant_fname,
                'trial_type': 'deviant',
                'N': N,
                'variation': v,
                'sequence': sequence_to_string(deviant_seq),
            })

    return records


def process_row(row: pd.Series, output_dir: Path) -> Dict[str, Any]:
    """
    Generate all WAV files for one metadata row across every model and both
    configurations (regular and counterbalanced). Collect metadata records,
    deduplicated by duration class (30s vs 10s).

    Args:
        row:        Pandas Series for one metadata row.
        output_dir: Base output directory.

    Returns:
        Dict with keys:
          'method_id'        (int)
          'success'          (bool)
          'error'            (str|None)
          'metadata_regular' (list of dicts — one per unique duration/stimulus)
          'metadata_counter' (list of dicts — one per unique duration/stimulus)
    """
    method_id = int(row['method_id'])
    models = list(MODEL_DURATIONS.keys())

    result = {
        'method_id': method_id,
        'success': True,
        'error': None,
        'metadata_regular': [],
        'metadata_counter': [],
    }

    try:
        for config_key, is_counter in [('regular', False), ('counter', True)]:
            config_name = "audio_outputs_counter" if is_counter else "audio_outputs_regular"
            config_dir = output_dir / config_name
            meta_key = f'metadata_{config_key}'
            seen_durations = set()

            for model_name in models:
                records = process_row_model_config(
                    row, model_name, config_dir, is_counterbalanced=is_counter
                )

                # Only keep metadata records for the first model encountered
                # at each duration class to avoid duplicating 10s rows
                duration_s = MODEL_DURATIONS[model_name] // 1000
                if duration_s not in seen_durations:
                    seen_durations.add(duration_s)
                    result[meta_key].extend(records)

    except Exception as e:
        logger.error(f"Failed to process method_id={method_id}: {e}")
        result['success'] = False
        result['error'] = str(e)

    return result


# =============================================================================
# PARALLEL ORCHESTRATION
# =============================================================================

def _process_chunk(args: Tuple) -> List[Dict[str, Any]]:
    """
    Worker target for multiprocessing: process a contiguous slice of rows.

    Args:
        args: Tuple of (chunk_df, output_dir, chunk_index).

    Returns:
        List of per-row result dicts from process_row().
    """
    chunk_df, output_dir, _ = args
    return [process_row(row, output_dir) for _, row in chunk_df.iterrows()]


def generate_all_stimuli(metadata_df: pd.DataFrame, output_dir: Path,
                         n_workers: int = 1) -> Dict[str, Any]:
    """
    Top-level generation loop: iterate over all metadata rows, optionally
    distributing work across a multiprocessing pool. Collects metadata records
    from all rows.

    Args:
        metadata_df: Filtered (and optionally chunked) metadata dataframe.
        output_dir:  Base output directory.
        n_workers:   Number of parallel worker processes.

    Returns:
        Dict with 'successful', 'failed' counts and 'metadata_regular',
        'metadata_counter' lists of record dicts.
    """
    n_rows = len(metadata_df)

    if n_workers > 1 and n_rows > n_workers:
        chunk_size = (n_rows + n_workers - 1) // n_workers
        chunks = [metadata_df.iloc[i:i + chunk_size]
                  for i in range(0, n_rows, chunk_size)]
        args_list = [(chunk, output_dir, i) for i, chunk in enumerate(chunks)]

        with Pool(processes=n_workers) as pool:
            chunk_results = pool.map(_process_chunk, args_list)

        all_results = [r for batch in chunk_results for r in batch]
    else:
        all_results = _process_chunk((metadata_df, output_dir, 0))

    # Aggregate counts and metadata
    output = {
        'successful': 0,
        'failed': 0,
        'metadata_regular': [],
        'metadata_counter': [],
    }
    for r in all_results:
        if r['success']:
            output['successful'] += 1
            output['metadata_regular'].extend(r['metadata_regular'])
            output['metadata_counter'].extend(r['metadata_counter'])
        else:
            output['failed'] += 1

    return output


def write_metadata_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write a list of metadata records to a CSV file.

    Args:
        records:     List of dicts, each representing one generated stimulus.
        output_path: Full path for the output CSV file.
    """
    if not records:
        logger.warning(f"No metadata records to write to {output_path}")
        return

    columns = [
        'filename', 'duration_s', 'method_id', 'standard_freq', 'deviant_freq',
        'tone_duration_ms', 'isi_ms', 'intensity_db', 'trial_type', 'N',
        'variation', 'sequence',
    ]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, index=False)
    logger.info(f"Metadata CSV written: {output_path} ({len(df)} rows)")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 0aa — Generate audio stimuli for MMN CV pipeline'
    )
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to metadata CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--chunk_idx', type=int, default=0,
                        help='SLURM array chunk index (0-indexed)')
    parser.add_argument('--n_chunks', type=int, default=1,
                        help='Total SLURM array chunk count')

    args = parser.parse_args()
    if args.n_workers is None:
        args.n_workers = cpu_count()

    print("=" * 60)
    print("Phase 0aa: Generate Audio Stimuli")
    print("=" * 60)
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Workers:      {args.n_workers}")
    print(f"Trial levels: {TRIAL_LEVELS}")
    print(f"Variations:   {NUM_VARIATIONS}")
    if args.n_chunks > 1:
        print(f"SLURM chunk:  {args.chunk_idx + 1} of {args.n_chunks}")
    print()

    # Load metadata
    metadata_path = Path(args.metadata_csv)
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {args.metadata_csv}")
        return 1

    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded {len(metadata_df)} rows from metadata")

    numeric_cols = ['standard_freq', 'deviant_freq', 'standard_dur',
                    'standard_isi', 'standard_int']
    for col in numeric_cols:
        if col in metadata_df.columns:
            metadata_df[col] = pd.to_numeric(metadata_df[col], errors='coerce')

    # Filter to frequency-change rows
    pre_filter_count = len(metadata_df)
    metadata_df = metadata_df[
        metadata_df['change_type'].str.lower() == 'frequency'
    ].copy()
    discarded = pre_filter_count - len(metadata_df)
    print(f"Frequency-change filter: {len(metadata_df)} retained, "
          f"{discarded} discarded")

    if len(metadata_df) == 0:
        print("ERROR: No frequency-change stimuli found in metadata.")
        return 1

    # SLURM chunking
    if args.n_chunks > 1:
        metadata_df = metadata_df.sort_values('method_id').reset_index(drop=True)
        total = len(metadata_df)
        chunk_size = (total + args.n_chunks - 1) // args.n_chunks
        start_idx = args.chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        metadata_df = metadata_df.iloc[start_idx:end_idx]
        print(f"SLURM chunk {args.chunk_idx}/{args.n_chunks}: "
              f"rows {start_idx}-{end_idx - 1} ({len(metadata_df)} rows)")

    if len(metadata_df) == 0:
        print("No rows to process in this chunk.")
        return 0

    # Create output directories (audio + metadata)
    output_dir = Path(args.output_dir)
    for model_name in MODEL_DURATIONS:
        (output_dir / "audio_outputs_regular" / model_name).mkdir(
            parents=True, exist_ok=True)
        (output_dir / "audio_outputs_counter" / model_name).mkdir(
            parents=True, exist_ok=True)
    (output_dir / "audio_outputs_regular_metadata").mkdir(
        parents=True, exist_ok=True)
    (output_dir / "audio_outputs_counter_metadata").mkdir(
        parents=True, exist_ok=True)

    print(f"Output directories created under: {output_dir}\n")

    # Generate audio and collect metadata
    results = generate_all_stimuli(metadata_df, output_dir,
                                   n_workers=args.n_workers)

    # Write metadata CSVs
    if args.n_chunks > 1:
        # Per-chunk files for later concatenation
        reg_meta_path = (output_dir / "audio_outputs_regular_metadata"
                         / f"chunk_{args.chunk_idx}.csv")
        ctr_meta_path = (output_dir / "audio_outputs_counter_metadata"
                         / f"chunk_{args.chunk_idx}.csv")
    else:
        reg_meta_path = (output_dir / "audio_outputs_regular_metadata"
                         / "metadata.csv")
        ctr_meta_path = (output_dir / "audio_outputs_counter_metadata"
                         / "metadata.csv")

    write_metadata_csv(results['metadata_regular'], reg_meta_path)
    write_metadata_csv(results['metadata_counter'], ctr_meta_path)

    # Summary
    files_per_row = (1 + len(TRIAL_LEVELS) * NUM_VARIATIONS) * len(MODEL_DURATIONS) * 2

    print()
    print("=" * 60)
    print("Phase 0aa Complete")
    print("=" * 60)
    print(f"Rows processed: {results['successful'] + results['failed']}")
    print(f"Successful:     {results['successful']}")
    print(f"Failed:         {results['failed']}")
    print(f"Files per row:  {files_per_row}")
    print(f"Total files:    {results['successful'] * files_per_row}")
    print(f"Metadata:       {reg_meta_path}")
    print(f"                {ctr_meta_path}")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
