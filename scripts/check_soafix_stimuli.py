#!/usr/bin/env python3
"""Verify a --trailing_floor_ms regeneration of the literature stimuli before anything is extracted.

The trailing-silence floor (00aa_generate_audio_stimuli.py::compute_tone_slots) reserves
max(trailing_floor_ms, SOA) of audio after the final tone's onset. Because that collapses to the
historical formula whenever SOA >= the floor, a floor of 400 must rebuild exactly the 12 literature
method_ids whose SOA is below 400 and leave the other 12 byte-identical. That is the whole basis for
re-screening all 48 conditions while comparing 24 of them against the committed baseline, so it is
worth proving on the real audio rather than assuming it.

Hence the full 24-method set is regenerated (it costs minutes) rather than only the changed 12: it
makes the byte-identity claim directly checkable. This script does the checking.

Three things are asserted, and all failures are collected before exiting rather than failing fast:

  1. The changed/unchanged partition derived from the metadata CSV (SOA >= floor) equals the
     expected id lists -- so a CSV edit surfaces here instead of silently redefining the test.
  2. Every (method, family, direction) group holds the expected 16 clips in both trees, and the
     md5 partition matches: 12 ids identical in all 4 of their groups, 12 differing.
  3. On the NEW wavs, the audio after the final tone's onset is max(floor, SOA) -- measured with
     insilico_mmn.detect_final_tone_onset_s, the same function the scoring pipeline time-locks on.
     For the changed ids the baseline wavs are measured too, as a positive control that the diff is
     the intended one rather than an incidental byte change.

Run from the project directory, after sourcing the environment:

    python scripts/check_soafix_stimuli.py
    python scripts/check_soafix_stimuli.py --onset_scope all      # every clip, ~25x slower

Exit code is the contract: 0 all good, 1 something is wrong -- do not proceed to staging.
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

# The 24 change_type=="Frequency" literature ids, split at SOA 400. Declared here and cross-checked
# against the CSV below; the point is to fail loudly if the sheet and this file ever disagree.
EXPECT_IDENTICAL = [9, 12, 20, 21, 27, 33, 43, 44, 55, 72, 74, 75]
EXPECT_DIFFER = [10, 17, 18, 19, 28, 29, 30, 31, 32, 37, 53, 60]

TRIAL_LEVELS = (3, 5, 7)
NUM_VARIATIONS = 5
SAMPLE_RATE = 16000
FAMILY_FRAMES = {"whisper": 480000, "wav2vec2": 160000}      # 30 s / 10 s at 16 kHz

# detect_final_tone_onset_s thresholds a 5 ms boxcar envelope, so it reads the onset ~0.875 ms
# (14 samples) early at every SOA, and PCM_16 quantization in the written file adds up to ~0.125 ms
# more. The measured tail therefore lands just ABOVE the true one. The tolerance is one-sided on
# purpose: a tail that is SHORT is precisely the failure being guarded against.
ONSET_TOL_MS = 2.0


def expected_clip_names(method_id):
    """The 16 wavs the generator writes per (method, family, direction)."""
    names = {f"method_{method_id:02d}_standard.wav"}
    names |= {f"method_{method_id:02d}_N{n}_var{v}_deviant.wav"
              for n in TRIAL_LEVELS for v in range(1, NUM_VARIATIONS + 1)}
    return names


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_designs(metadata_csv):
    """{method_id: (tone_duration_ms, isi_ms, soa_ms)} for the Frequency rows the generator keeps."""
    with open(metadata_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["change_type"] == "Frequency"]
    out = {}
    for r in rows:
        dur, isi = float(r["standard_dur"]), float(r["standard_isi"])
        out[int(r["method_id"])] = (dur, isi, dur + isi)
    return out


def measure_tail_ms(wav_path, detect_final_tone_onset_s):
    """Audio (ms) after the final tone's ONSET, plus the clip's frame count and rate."""
    info = sf.info(str(wav_path))
    onset_s = detect_final_tone_onset_s(str(wav_path))
    if onset_s is None:
        return None, info.frames, info.samplerate
    return info.frames / info.samplerate * 1000.0 - onset_s * 1000.0, info.frames, info.samplerate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new_src", default="outputs/stim_gen_soafix",
                   help="flat generator tree from the --trailing_floor_ms run")
    p.add_argument("--baseline_whisper", default="outputs/mmn_stimuli",
                   help="staged committed baseline, whisper (30 s clips)")
    p.add_argument("--baseline_wav2vec2", default="outputs/mmn_stimuli_wav2vec2",
                   help="staged committed baseline, wav2vec2 (10 s clips)")
    p.add_argument("--metadata_csv",
                   default="data/metadata/literature_frequency_intensity_duration_metadata.csv")
    p.add_argument("--trailing_floor_ms", type=float, default=400.0)
    p.add_argument("--onset_scope", choices=("key", "all"), default="key",
                   help="which new clips to time: 'key' = the standard + N7/var1 deviant that "
                        "finalize_method reads (192 clips); 'all' = every clip (1536, ~25x slower)")
    p.add_argument("--tol_ms", type=float, default=ONSET_TOL_MS)
    p.add_argument("--repo_root", default=".",
                   help="for importing scripts/insilico_mmn.py")
    args = p.parse_args()

    repo = Path(args.repo_root).resolve()
    for sub in ("scripts", "src"):                 # insilico_mmn imports the mbs package
        sys.path.insert(0, str(repo / sub))
    try:
        from insilico_mmn import detect_final_tone_onset_s
    except ImportError as e:
        print(f"ERROR: cannot import scripts/insilico_mmn.py ({e}).")
        print(f"  Run from the project directory (or pass --repo_root) with the environment")
        print(f"  sourced. This script deliberately uses the same onset detector the scoring")
        print(f"  pipeline time-locks on, rather than a copy of it.")
        return 1

    new_src = Path(args.new_src)
    baselines = {"whisper": Path(args.baseline_whisper),
                 "wav2vec2": Path(args.baseline_wav2vec2)}
    floor = args.trailing_floor_ms

    if not new_src.is_dir():
        print(f"ERROR: new stimulus tree not found: {new_src}")
        return 1

    designs = read_designs(args.metadata_csv)
    failures = []

    print("=" * 78)
    print("Trailing-floor stimulus check")
    print("=" * 78)
    print(f"new tree      : {new_src}")
    print(f"baseline      : {baselines['whisper']} | {baselines['wav2vec2']}")
    print(f"metadata      : {args.metadata_csv} ({len(designs)} Frequency rows)")
    print(f"floor         : {floor:g} ms   onset scope: {args.onset_scope}   tol: {args.tol_ms:g} ms")
    print()

    # ── 1. the partition the CSV itself implies ────────────────────────────────
    derived_same = sorted(i for i, (_, _, soa) in designs.items() if soa >= floor)
    derived_diff = sorted(i for i, (_, _, soa) in designs.items() if soa < floor)
    if derived_same != sorted(EXPECT_IDENTICAL) or derived_diff != sorted(EXPECT_DIFFER):
        failures.append(
            "the metadata CSV no longer implies the expected partition:\n"
            f"    SOA >= {floor:g}: {derived_same}\n      expected: {sorted(EXPECT_IDENTICAL)}\n"
            f"    SOA <  {floor:g}: {derived_diff}\n      expected: {sorted(EXPECT_DIFFER)}")
        print("PARTITION MISMATCH -- see failures below; continuing with the CSV-derived split.")
    expect_same, expect_diff = set(derived_same), set(derived_diff)

    # ── 2. md5 every group ────────────────────────────────────────────────────
    group_identical = {}          # (id, family, direction) -> bool
    diff_examples = defaultdict(list)
    for mid in sorted(designs):
        want = expected_clip_names(mid)
        for family, base_root in baselines.items():
            for cfg, suffix in (("regular", ""), ("counter", "_counter")):
                new_dir = new_src / f"audio_outputs_{cfg}" / family
                base_dir = base_root / f"method_{mid:02d}{suffix}"
                new_files = {p.name: p for p in new_dir.glob(f"method_{mid:02d}_*.wav")}
                base_files = {p.name: p for p in base_dir.glob("*.wav")}
                tag = f"method_{mid:02d}{suffix}/{family}"

                if set(new_files) != want:
                    failures.append(f"{tag}: new tree has {len(new_files)} clips, expected 16 "
                                    f"(missing {sorted(want - set(new_files))[:3]})")
                    group_identical[(mid, family, cfg)] = None
                    continue
                if set(base_files) != want:
                    failures.append(f"{tag}: baseline {base_dir} has {len(base_files)} clips, "
                                    f"expected 16 (missing {sorted(want - set(base_files))[:3]})")
                    group_identical[(mid, family, cfg)] = None
                    continue

                differing = [n for n in sorted(want) if md5(new_files[n]) != md5(base_files[n])]
                group_identical[(mid, family, cfg)] = not differing
                if differing:
                    diff_examples[mid].append(f"{tag} ({len(differing)}/16: {differing[:3]})")

    # A layout change is family- and direction-independent, so every id must be identical in all
    # four of its groups or none of them. A split means a partial regeneration.
    observed_same, observed_diff = set(), set()
    for mid in sorted(designs):
        verdicts = [group_identical.get((mid, f, c))
                    for f in baselines for c in ("regular", "counter")]
        if None in verdicts:
            continue
        if all(verdicts):
            observed_same.add(mid)
        elif not any(verdicts):
            observed_diff.add(mid)
        else:
            failures.append(f"method_{mid:02d}: identical in only {sum(verdicts)}/4 groups -- "
                            f"partial regeneration. {'; '.join(diff_examples[mid])}")

    if observed_same != expect_same or observed_diff != expect_diff:
        wrongly_changed = sorted(expect_same - observed_same)
        wrongly_same = sorted(expect_diff - observed_diff)
        if wrongly_changed:
            failures.append(
                f"ids that MUST be byte-identical changed: {wrongly_changed}. The layout moved "
                f"where it must not have, so the 24 unchanged conditions are no longer comparable "
                f"to the committed baseline. STOP -- do not stage or extract.\n    "
                + "\n    ".join(d for m in wrongly_changed for d in diff_examples[m]))
        if wrongly_same:
            failures.append(f"ids that MUST change came out identical: {wrongly_same}. The floor "
                            f"did not take effect -- check that --trailing_floor_ms 400 actually "
                            f"reached the generator (the banner prints it).")

    print(f"md5 partition : {len(observed_same)} identical, {len(observed_diff)} differing "
          f"(expect {len(expect_same)}/{len(expect_diff)})")
    print(f"  identical   : {sorted(observed_same)}")
    print(f"  differing   : {sorted(observed_diff)}")
    print()

    # ── 3. trailing audio on the new wavs, with a baseline positive control ───
    print(f"{'method':>12} {'fam':<9} {'soa':>7} {'expected':>9} "
          f"{'new tail':>10} {'delta':>7} {'baseline tail':>14}")
    per_family_delta = defaultdict(list)
    for mid in sorted(designs):
        _, _, soa = designs[mid]
        expected = max(floor, soa)
        for family, base_root in baselines.items():
            new_dir = new_src / "audio_outputs_regular" / family
            if args.onset_scope == "all":
                probes = sorted(expected_clip_names(mid))
            else:
                probes = [f"method_{mid:02d}_standard.wav",
                          f"method_{mid:02d}_N7_var1_deviant.wav"]

            tails = []
            for name in probes:
                wav = new_dir / name
                if not wav.exists():
                    continue
                tail, frames, rate = measure_tail_ms(wav, detect_final_tone_onset_s)
                if rate != SAMPLE_RATE or frames != FAMILY_FRAMES[family]:
                    failures.append(f"method_{mid:02d}/{family}/{name}: {frames} frames @ {rate} Hz, "
                                    f"expected {FAMILY_FRAMES[family]} @ {SAMPLE_RATE}")
                    continue
                if tail is None:
                    failures.append(f"method_{mid:02d}/{family}/{name}: no tone onset detected")
                    continue
                tails.append(tail)
                delta = tail - expected
                per_family_delta[family].append(delta)
                if not (0.0 <= delta <= args.tol_ms):
                    failures.append(
                        f"method_{mid:02d}/{family}/{name}: trailing audio {tail:.3f} ms, expected "
                        f"{expected:.3f} (delta {delta:+.3f}, tolerance 0..{args.tol_ms:g})")

            # Positive control: the changed ids must have had a SHORT tail before.
            base_tail = None
            if mid in expect_diff:
                base_wav = base_root / f"method_{mid:02d}" / f"method_{mid:02d}_standard.wav"
                if base_wav.exists():
                    base_tail, _, _ = measure_tail_ms(base_wav, detect_final_tone_onset_s)
                    if base_tail is not None and not (0.0 <= base_tail - soa <= args.tol_ms):
                        failures.append(
                            f"method_{mid:02d}/{family}: baseline trailing audio {base_tail:.3f} ms "
                            f"is not one SOA ({soa:.0f}); the baseline is not what we think it is")

            if tails:
                shown = f"{min(tails):.3f}" if len(set(round(t, 3) for t in tails)) == 1 \
                        else f"{min(tails):.3f}-{max(tails):.3f}"
                print(f"{'method_%02d' % mid:>12} {family:<9} {soa:>7.0f} {expected:>9.0f} "
                      f"{shown:>10} {min(t - expected for t in tails):>+7.3f} "
                      f"{('%.3f' % base_tail) if base_tail is not None else '(unchanged)':>14}")

    print()
    for family, deltas in per_family_delta.items():
        if deltas:
            print(f"onset bias {family:<9}: delta in [{min(deltas):+.3f}, {max(deltas):+.3f}] ms "
                  f"over {len(deltas)} clips (detector reads ~0.875 ms early by construction)")

    print()
    print("=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS -- 12 ids byte-identical, 12 rebuilt, trailing audio is max(floor, SOA) everywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
