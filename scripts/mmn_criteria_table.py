"""Per-target (single electrode or single parcel) C0-S6 criteria table, for embedding under the
Row A (Fz/FCz) and Row C (frontal/central/temporal) figure panels in insilico_mmn*.py.

Pure orchestration over the existing criteria machinery -- no shape/recovery/envelope logic is
reimplemented here. `analyze_mmn_criteria.py`'s C0-S3 (`trace_stats`/`decide`) and
`analyze_mmn_criteria_s5_s6.py`'s S4/S5/S6 (`dip_recovery_scan`/`deadline_bound_recovery`) already
operate on an arbitrary 1-D z-scored trace; both scripts just only ever call them on the ROI-MEAN
trace. `_criteria_for_trace` below is that same per-run block, generalized to run once per
individual target column instead of once on the ROI mean, and `compute_criteria_table` loops it
over every requested target name.

Usage (called from insilico_mmn.py/insilico_mmn_electrodes.py/insilico_mmn_attn.py at plot time,
on the in-memory `res["rel_ms"]`/`res["z_diff"]` arrays from finalize_method -- no h5 re-read):

    duration_map = analyze_mmn_criteria_s5_s6.load_duration_map(DURATION_CSV)
    table = compute_criteria_table(res["rel_ms"], res["z_diff"], ["Fz", "FCz"], method,
                                    duration_map)
    # table == {"Fz": {"C0": 1, "S1": 0, ..., "S6": 0}, "FCz": {...}}
"""

import numpy as np

import analyze_mmn_criteria as amc
import analyze_mmn_criteria_s5_s6 as s5s6

CRITERIA_COLUMNS = ["C0", "S1", "S2", "S3", "S4", "S5", "S6"]


def _criteria_for_trace(t, z, method_id, duration_map, window, envelope, recovery_ms,
                         recovery_frac, ctrl_window, edge_guard_bins):
    """C0-S6 for one 1-D z_diff trace. Generalizes the per-run block in
    analyze_mmn_criteria_s5_s6.py main() (lines ~218-264) from "the ROI-mean trace" to "any
    single trace" -- same functions, same formulas, just called on a different input.
    """
    cur_lo, cur_hi = window
    env_lo, env_hi = envelope

    # C0-S3: analyze_mmn_criteria.trace_stats/decide on the fixed 100-240 ms window.
    s_cur = amc.trace_stats(t, z, cur_lo, cur_hi, recovery_ms, recovery_frac,
                             ctrl_window[0], ctrl_window[1], edge_guard_bins)
    v_cur = amc.decide(s_cur)

    # S4: redefined tone-end-relative dip+recovery scan (NOT decide()'s retired S4_specificity).
    tone_end_ms = s5s6.tone_end_ms_for(method_id, duration_map)
    s4 = s5s6.dip_recovery_scan(t, z, tone_end_ms, recovery_frac=recovery_frac)

    # S5/S6: unbound trough search over [0, t.max()] + deadline-bound recovery + envelope guard.
    tmax = float(t.max())
    s_glob = amc.trace_stats(t, z, 0.0, tmax, recovery_ms, recovery_frac,
                              ctrl_window[0], ctrl_window[1], edge_guard_bins)
    global_argmin_ms = s_glob.get("argmin_ms", float("nan"))
    global_peak = s_glob.get("peak", 0.0)
    recovery_deadline = min(tone_end_ms + s5s6.RECOVERY_DEADLINE_SPAN_MS, tmax)
    recovered5, _ = s5s6.deadline_bound_recovery(t, z, global_argmin_ms, global_peak,
                                                  recovery_deadline, recovery_frac)
    s5 = bool(global_peak < 0.0 and recovered5)
    s6 = bool(s5 and not np.isnan(global_argmin_ms) and env_lo <= global_argmin_ms <= env_hi)

    return {
        "C0": int(bool(v_cur["C0_current"])),
        "S1": int(bool(v_cur["S1_interior"])),
        "S2": int(bool(v_cur["S2_recovery"])),
        "S3": int(bool(v_cur["S3_interior_recovery"])),
        "S4": int(bool(s4["qualifies"])),
        "S5": int(s5),
        "S6": int(s6),
    }


def compute_criteria_table(rel_ms, z_diff, target_names, method, duration_map,
                            window=(100.0, 240.0), envelope=(90.0, 250.0),
                            recovery_ms=120.0, recovery_frac=0.5,
                            ctrl_window=(300.0, 440.0), edge_guard_bins=1):
    """{target_name: {"C0": 0/1, "S1": 0/1, ..., "S6": 0/1}} for every name in `target_names`,
    each computed from its own column of an already baseline-z-scored z_diff [n_t, n_target].

    `method` is the stimulus-dir name (e.g. "method_27"), used to look up the per-method tone
    duration for S4/S5 via `duration_map` (from analyze_mmn_criteria_s5_s6.load_duration_map).
    """
    method_id = s5s6.parse_method_id(method)
    out = {}
    for i, name in enumerate(target_names):
        out[name] = _criteria_for_trace(rel_ms, z_diff[:, i], method_id, duration_map,
                                         window, envelope, recovery_ms, recovery_frac,
                                         ctrl_window, edge_guard_bins)
    return out
