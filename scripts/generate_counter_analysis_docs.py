#!/usr/bin/env python3
"""
Generate results_analysis_with_counter.md and
MMN_pipeline_analysis_decisions_notes_062226_with_counter.md
from CSVs in outputs/results_with_counter/.

Usage (from repo root):
    python scripts/generate_counter_analysis_docs.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("outputs/results_with_counter")
OUT_DIR = Path("aux/analysis_with_counter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["tiny", "base", "small", "medium"]
MODEL_LABELS = {m: f"whisper-{m}" for m in MODELS}

# The 7-electrode ROI used for electrodes (FC1 absent from this montage)
ELEC_ROI_COLS = ["Fz_peak", "FCz_peak", "Cz_peak", "FC2_peak", "F1_peak", "F2_peak"]
PARCEL_ROI_COLS = ["frontal_peak", "central_peak"]
CRITERIA = ["C0", "S1", "S2", "S3", "S4", "S5", "S6"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name, required=True):
    p = RESULTS_DIR / name
    if not p.exists():
        if required:
            sys.exit(f"Missing: {p}\nRun Step 6 on the cluster and rsync first.")
        return None
    return pd.read_csv(p)


def fmt(val, decimals=2):
    if pd.isna(val):
        return "—"
    return f"{val:+.{decimals}f}" if val != 0 else f"{val:.{decimals}f}"


def pct(n, d):
    return f"{n}/{d}"


def md_table(headers, rows):
    sep = ["-" * max(len(h), 6) for h in headers]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def roi_mean(sub, cols):
    available = [c for c in cols if c in sub.columns]
    if not available:
        return pd.Series(np.nan, index=sub.index)
    return sub[available].mean(axis=1)


# ---------------------------------------------------------------------------
# Load main results table
# ---------------------------------------------------------------------------
df = _load("mmn_results_table.csv")
df["is_counter"] = df["method"].str.endswith("_counter")
df["method_base"] = df["method"].str.replace("_counter", "", regex=False)

# Compute ROI means
df["elec_roi_peak"] = roi_mean(df, ELEC_ROI_COLS)
df["parcel_roi_peak"] = roi_mean(df, PARCEL_ROI_COLS)

# Per-run ROI mean peak (depends on level)
df["roi_peak"] = np.where(df["level"] == "electrodes", df["elec_roi_peak"], df["parcel_roi_peak"])
df["mmn"] = df["roi_peak"] < 0

# Parcels/electrodes convenience subsets
mtrf = df[df["mapping"] == "mtrf"]
enc  = df[df["mapping"] == "encoder"]


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def per_method_mean_table(sub, label):
    """Table 1a/2a style: mean MMN per model × method."""
    rows = []
    for m in MODELS:
        ms = sub[sub["model"] == f"whisper-{m}"]
        for _, mrow in ms.groupby("method"):
            method = mrow["method"].iloc[0]
            is_ctr = mrow["is_counter"].iloc[0]
            lbl    = mrow["label"].iloc[0]
            p_mean = mrow[mrow["level"]=="parcels"]["parcel_roi_peak"].mean()
            e_mean = mrow[mrow["level"]=="electrodes"]["elec_roi_peak"].mean()
            rows.append([m, method, lbl, "counter" if is_ctr else "regular",
                         fmt(p_mean), fmt(e_mean)])
    return md_table(["Model","Method","Stimulus","Type","MMN (parcels)","MMN (electrodes)"], rows)


def count_table(sub, n_methods, label):
    """Table 1b/2b style: per-model MMN count summary."""
    rows = []
    totals = [0, 0, 0]
    for m in MODELS:
        ms = sub[sub["model"] == f"whisper-{m}"]
        p = ms[ms["level"]=="parcels"]["mmn"].sum()
        e = ms[ms["level"]=="electrodes"]["mmn"].sum()
        t = p + e
        rows.append([m, pct(p, n_methods), pct(e, n_methods), pct(t, 2*n_methods)])
        totals[0] += p; totals[1] += e; totals[2] += t
    rows.append(["**Total**",
                 pct(totals[0], len(MODELS)*n_methods),
                 pct(totals[1], len(MODELS)*n_methods),
                 pct(totals[2], len(MODELS)*2*n_methods)])
    return md_table(["Model", f"Parcels (n/{n_methods})", f"Electrodes (n/{n_methods})",
                     f"Total (n/{2*n_methods})"], rows)


def per_method_avg_table(sub, label):
    """Table 1c/2c style: per-method average across models."""
    rows = []
    for method, grp in sub.groupby("method"):
        lbl = grp["label"].iloc[0]
        is_ctr = grp["is_counter"].iloc[0]
        avg = grp["roi_peak"].mean()
        rows.append([method, lbl, "counter" if is_ctr else "regular", fmt(avg)])
    return md_table(["Method","Stimulus","Type","Avg MMN across models"], rows)


def per_model_avg_table(sub, label):
    """Table 1d/2d style: per-model average across stimuli."""
    rows = []
    for m in MODELS:
        ms = sub[sub["model"] == f"whisper-{m}"]
        avg_p = ms[ms["level"]=="parcels"]["parcel_roi_peak"].mean()
        avg_e = ms[ms["level"]=="electrodes"]["elec_roi_peak"].mean()
        pool  = ms["roi_peak"]
        avg   = pool.mean()
        cnt   = (pool < 0).sum()
        n_methods = ms["method"].nunique()
        rows.append([m, fmt(avg), pct(cnt//2, n_methods)])
    return md_table(["Model","Avg MMN across stimuli","n stimuli showing MMN"], rows)


def per_model_level_avg(sub, label):
    """Table 1e/2e style."""
    rows = []
    for m in MODELS:
        ms = sub[sub["model"] == f"whisper-{m}"]
        p = ms[ms["level"]=="parcels"]["parcel_roi_peak"].mean()
        e = ms[ms["level"]=="electrodes"]["elec_roi_peak"].mean()
        rows.append([m, fmt(p), fmt(e)])
    return md_table(["Model","Mean MMN parcels","Mean MMN electrodes"], rows)


def regular_counter_split_table(sub, label, n_methods=10):
    """New: separate regular vs counter counts per model."""
    rows = []
    for m in MODELS:
        ms = sub[sub["model"] == f"whisper-{m}"]
        for tag, mask in [("regular", ~ms["is_counter"]), ("counter", ms["is_counter"])]:
            grp = ms[mask]
            p = grp[grp["level"]=="parcels"]["mmn"].sum()
            e = grp[grp["level"]=="electrodes"]["mmn"].sum()
            rows.append([m, tag, pct(p, n_methods), pct(e, n_methods), pct(p+e, 2*n_methods)])
    return md_table(["Model","Set",
                     f"Parcels (n/{n_methods})",f"Electrodes (n/{n_methods})",
                     f"Total (n/{2*n_methods})"], rows)


def paired_agreement_table(sub):
    """New: for each (model, level, method_base), do regular and counter agree on MMN?"""
    rows = []
    for m in MODELS:
        ms = sub[sub["model"] == f"whisper-{m}"]
        both_mmn = 0; both_no = 0; disagree = 0; total = 0
        for level in ["parcels","electrodes"]:
            ls = ms[ms["level"]==level]
            reg = ls[~ls["is_counter"]].set_index("method_base")["mmn"]
            ctr = ls[ls["is_counter"]].set_index("method_base")["mmn"]
            common = reg.index.intersection(ctr.index)
            for mb in common:
                total += 1
                r, c = reg[mb], ctr[mb]
                if r and c: both_mmn += 1
                elif not r and not c: both_no += 1
                else: disagree += 1
        rows.append([m, pct(both_mmn, total), pct(both_no, total), pct(disagree, total)])
    return md_table(["Model","Both MMN","Both no-MMN","Disagree"], rows)


def per_method_regularcounter(sub):
    """New: per method_base, regular vs counter MMN count across all model×level cells."""
    METHOD_IDS = ["27","37","43","44","53","55","60","72","74","75"]
    rows = []
    for mid in METHOD_IDS:
        reg_name = f"method_{mid}"
        ctr_name = f"method_{mid}_counter"
        rg = sub[sub["method"]==reg_name]
        ct = sub[sub["method"]==ctr_name]
        reg_lbl = rg["label"].iloc[0] if len(rg) else "—"
        ctr_lbl = ct["label"].iloc[0] if len(ct) else "—"
        r_cnt = rg["mmn"].sum()
        c_cnt = ct["mmn"].sum()
        r_n   = len(rg)
        c_n   = len(ct)
        rows.append([reg_name, reg_lbl, ctr_lbl, pct(r_cnt, r_n), pct(c_cnt, c_n)])
    return md_table(["Method","Regular (Std→Dev)","Counter (Dev→Std)",
                     f"Regular MMN (n/{r_n})","Counter MMN (n/{c_n})"], rows)


# ---------------------------------------------------------------------------
# Criteria table helpers
# ---------------------------------------------------------------------------

def load_criteria_s5s6(name):
    d = _load(name, required=False)
    if d is None:
        return None
    d["is_counter"] = d["method"].str.endswith("_counter")
    # Boolean criteria columns
    crit_map = {
        "C0": "current__C0_current",
        "S1": "current__S1_interior",
        "S2": "current__S2_recovery",
        "S3": "current__S3_interior_recovery",
        "S4": "current__S4_specificity",
        "S5": "global__S5",
        "S6": "global__S6_envelope_recovery",
    }
    for short, col in crit_map.items():
        if col in d.columns:
            d[short] = d[col].astype(bool)
        else:
            d[short] = False
    return d


def criteria_count_table(d, mapping_filter, n_per_model, title):
    """Tables 13/14/25/26/27/28 style."""
    sub = d[d["mapping"]==mapping_filter]
    rows = []
    totals = {c: 0 for c in CRITERIA}
    tot_n = 0
    for m in MODELS:
        ms = sub[sub["model"]==f"whisper-{m}"]
        n = len(ms)
        row = [m]
        for c in CRITERIA:
            cnt = ms[c].sum()
            totals[c] += cnt
            row.append(pct(cnt, n_per_model))
        rows.append(row)
        tot_n += n
    total_n = len(MODELS) * n_per_model
    tot_row = ["**Total**"] + [pct(totals[c], total_n) for c in CRITERIA]
    rows.append(tot_row)
    return md_table(["Model"] + [f"{c} (n/{n_per_model})" for c in CRITERIA], rows)


def criteria_split_table(d, mapping_filter, n_per_model):
    """Criteria counts split by regular vs counter."""
    sub = d[d["mapping"]==mapping_filter]
    rows = []
    for tag, mask_fn in [("regular", lambda ms: ms[~ms["is_counter"]]),
                          ("counter", lambda ms: ms[ms["is_counter"]])]:
        totals = {c: 0 for c in CRITERIA}
        for m in MODELS:
            ms = mask_fn(sub[sub["model"]==f"whisper-{m}"])
            n  = len(ms)
            row = [m, tag]
            for c in CRITERIA:
                cnt = ms[c].sum() if n > 0 else 0
                totals[c] += cnt
                row.append(pct(cnt, n_per_model))
            rows.append(row)
    return md_table(["Model","Set"] + [f"{c} (n/{n_per_model})" for c in CRITERIA], rows)


# ---------------------------------------------------------------------------
# ROI variant helper
# ---------------------------------------------------------------------------

def load_roi_variants():
    d = _load("mmn_roi_variants.csv", required=False)
    if d is None:
        return None
    d["is_counter"] = d["method"].str.endswith("_counter")
    d["mmn"] = d["peak"] < 0
    return d


def roi_count_table(d, mapping_filter, level_filter, n_methods):
    """Tables 15-18 style."""
    sub = d[(d["mapping"]==mapping_filter) & (d["level"]==level_filter)]
    variants = sub["roi_variant"].unique().tolist()
    rows = []
    totals = {v: 0 for v in variants}
    for m in MODELS:
        ms = sub[sub["model"]==f"whisper-{m}"]
        row = [m]
        for v in variants:
            cnt = ms[ms["roi_variant"]==v]["mmn"].sum()
            totals[v] += cnt
            row.append(pct(cnt, n_methods))
        rows.append(row)
    total_n = len(MODELS) * n_methods
    rows.append(["**Total**"] + [pct(totals[v], total_n) for v in variants])
    return md_table(["Model"] + [v for v in variants], rows)


# ---------------------------------------------------------------------------
# Build results_analysis_with_counter.md
# ---------------------------------------------------------------------------

n_reg = 10  # regular methods per model×level×mapping
n_all = 20  # regular + counter

lines = []
W = lines.append

W("# In-Silico MMN Results Analysis — With Counterbalanced Methods")
W("")
W("> **Extends `aux/results_analysis.md`** by adding 10 counterbalanced stimulus pairs")
W("> (standard/deviant frequencies swapped) to every analysis. Scope: 4 Whisper models")
W("> × 2 levels (parcels, electrodes) × 20 methods (10 regular + 10 counter) × 2 mappings")
W("> (mTRF, encoder) = **320 (model, level, mapping, method)** combinations.")
W("> All definitions, metric formulas, and section structure are identical to")
W("> `aux/results_analysis.md`; only counts and tables are updated.")
W("")
W("## Key motivation for counterbalancing")
W("")
W("Adding counterbalanced pairs (where the former deviant frequency becomes the new standard")
W("and vice versa) provides a stringent control: if the in-silico MMN signal reflects genuine")
W("deviance detection by the model, it should appear when *either* frequency is the deviant.")
W("If the signal is instead driven by a frequency-specific response bias (e.g. the model simply")
W("responds more strongly to higher or lower frequencies), then the counterbalanced direction")
W("would fail to show an MMN. The new Section 0b below reports this key comparison directly.")
W("")
W("---")
W("")

# ---- Section 1: mTRF ----
W("## Section 1 — Method A (mTRF), all 20 methods")
W("")
W("**Table 1a. Mean MMN per model × method (mTRF)**")
W("")
W(per_method_mean_table(mtrf, "mTRF"))
W("")
W("**Table 1b. Per-model MMN count summary — mTRF** (n/20 per level)")
W("")
W(count_table(mtrf, n_all, "mTRF"))
W("")
W("**Table 1c. Per-method average across models — mTRF**")
W("")
W(per_method_avg_table(mtrf, "mTRF"))
W("")
W("**Table 1d. Per-model average across stimuli — mTRF**")
W("")
W(per_model_avg_table(mtrf, "mTRF"))
W("")
W("**Table 1e. Mean MMN per model, level separated — mTRF**")
W("")
W(per_model_level_avg(mtrf, "mTRF"))
W("")
W("---")
W("")

# ---- Section 2: Encoder ----
W("## Section 2 — Method B (encoder), all 20 methods")
W("")
W("**Table 2a. Mean MMN per model × method (encoder)**")
W("")
W(per_method_mean_table(enc, "encoder"))
W("")
W("**Table 2b. Per-model MMN count summary — encoder** (n/20 per level)")
W("")
W(count_table(enc, n_all, "encoder"))
W("")
W("**Table 2c. Per-method average across models — encoder**")
W("")
W(per_method_avg_table(enc, "encoder"))
W("")
W("**Table 2d. Per-model average across stimuli — encoder**")
W("")
W(per_model_avg_table(enc, "encoder"))
W("")
W("**Table 2e. Mean MMN per model, level separated — encoder**")
W("")
W(per_model_level_avg(enc, "encoder"))
W("")
W("---")
W("")

# ---- Section 0b: Counterbalanced analysis ----
W("## Section 0b — Counterbalanced analysis (key new results)")
W("")
W("### Regular vs counter split — mTRF")
W("")
W("**Table CB-1a. mTRF MMN counts: regular vs counterbalanced** (n/10 per set)")
W("")
W(regular_counter_split_table(mtrf, "mTRF", n_reg))
W("")
W("### Regular vs counter split — encoder")
W("")
W("**Table CB-1b. Encoder MMN counts: regular vs counterbalanced** (n/10 per set)")
W("")
W(regular_counter_split_table(enc, "encoder", n_reg))
W("")
W("### Agreement between regular and counter versions")
W("")
W("**Table CB-2a. Regular ↔ counter agreement — mTRF**")
W("(n = 20 pairs per model = 10 method_bases × 2 levels)")
W("")
W(paired_agreement_table(mtrf))
W("")
W("**Table CB-2b. Regular ↔ counter agreement — encoder**")
W("")
W(paired_agreement_table(enc))
W("")
W("### Per-method-base: regular vs counter MMN count (mTRF, across all model×level)")
W("")
W("**Table CB-3a. mTRF: per-method-base regular vs counter** (n/8 per cell = 4 models × 2 levels)")
W("")
W(per_method_regularcounter(mtrf))
W("")
W("**Table CB-3b. Encoder: per-method-base regular vs counter** (n/8 per cell)")
W("")
W(per_method_regularcounter(enc))
W("")
W("### Interpretation guide")
W("")
W("- **Both MMN (bidirectional)**: the model detects deviance regardless of which")
W("  frequency plays standard vs deviant — strongest evidence for genuine deviance detection.")
W("- **Regular only**: MMN appears only when the frequency from the literature is the deviant.")
W("  Could reflect a frequency-specific response bias rather than deviance per se.")
W("- **Counter only**: MMN appears only in the swapped direction; suggests the 'deviant'")
W("  frequency produces a weaker response regardless of sequence context.")
W("- **Neither**: no MMN signal in either direction.")
W("")
W("---")
W("")

# ---- Cross-method comparisons ----
W("## Cross-Method Comparisons (updated, 20 methods)")
W("")
W("**Table 3. mTRF vs. encoder agreement** (per model × level, over 20 methods)")
W("")
agree_rows = []
for m in MODELS:
    for level in ["parcels","electrodes"]:
        mdf = df[(df["model"]==f"whisper-{m}") & (df["level"]==level)]
        both_mmn = 0; both_no = 0; agree = 0; disagree = 0
        for method in mdf["method"].unique():
            rm = mdf[(mdf["method"]==method) & (mdf["mapping"]=="mtrf")]["mmn"]
            em = mdf[(mdf["method"]==method) & (mdf["mapping"]=="encoder")]["mmn"]
            if len(rm) == 0 or len(em) == 0:
                continue
            r, e = bool(rm.iloc[0]), bool(em.iloc[0])
            if r and e: both_mmn += 1; agree += 1
            elif not r and not e: both_no += 1; agree += 1
            else: disagree += 1
        n = both_mmn + both_no + disagree
        agree_rows.append([m, level, both_mmn, both_no, pct(agree, n), pct(disagree, n)])
W(md_table(["Model","Level","Both MMN","Both no-MMN","Agree","Disagree"], agree_rows))
W("")
W("**Table 4. Stimulus-method consistency** (MMN count across all 32 model×level×mapping combinations)")
W("")
METHOD_IDS_ALL = [
    ("method_75","1000→1200 Hz (Karger_2014)"),
    ("method_75_counter","1200→1000 Hz (Karger_2014)"),
    ("method_74","1000→1500 Hz (Domjan_2012)"),
    ("method_74_counter","1500→1000 Hz (Domjan_2012)"),
    ("method_72","1000→1200 Hz (Bodatsch_2011)"),
    ("method_72_counter","1200→1000 Hz (Bodatsch_2011)"),
    ("method_60","1000→1500 Hz (Umbricht_2003a)"),
    ("method_60_counter","1500→1000 Hz (Umbricht_2003a)"),
    ("method_53","1000→1200 Hz (Salisbury_2002a)"),
    ("method_53_counter","1200→1000 Hz (Salisbury_2002a)"),
    ("method_55","1000→2000 Hz (Shinozaki_2002a)"),
    ("method_55_counter","2000→1000 Hz (Shinozaki_2002a)"),
    ("method_37","1000→1050 Hz (Javitt_2000a)"),
    ("method_37_counter","1050→1000 Hz (Javitt_2000a)"),
    ("method_43","633→700 Hz (Michie_2000b)"),
    ("method_43_counter","700→633 Hz (Michie_2000b)"),
    ("method_44","633→1000 Hz (Michie_2000c)"),
    ("method_44_counter","1000→633 Hz (Michie_2000c)"),
    ("method_27","1000→1064 Hz (Schall_1999a)"),
    ("method_27_counter","1064→1000 Hz (Schall_1999a)"),
]
t4_rows = []
for method, stim in METHOD_IDS_ALL:
    sub = df[df["method"]==method]
    mt = sub[sub["mapping"]=="mtrf"]["mmn"].sum()
    en = sub[sub["mapping"]=="encoder"]["mmn"].sum()
    n_mt = len(sub[sub["mapping"]=="mtrf"])
    n_en = len(sub[sub["mapping"]=="encoder"])
    t4_rows.append([method, stim, pct(mt, n_mt), pct(en, n_en), pct(mt+en, n_mt+n_en)])
W(md_table(["Method","Stimulus (source)",
            f"mTRF (n/{n_mt})","Encoder (n/{n_en})",
            f"Total (n/{n_mt+n_en})"], t4_rows))
W("")
W("---")
W("")

# ---- Section 4: Shape criteria ----
crit_s5s6 = load_criteria_s5s6("mmn_criteria_s5_s6.csv")
W("## Section 4 — Shape metrics C0–S6, updated (20 methods, all ROIs)")
W("")
if crit_s5s6 is not None:
    W("### Results — mTRF (n/40 per model = 20 methods × 2 levels)")
    W("")
    W("**Table 13. MMN-present counts per criterion, by model — mTRF**")
    W("")
    W(criteria_count_table(crit_s5s6, "mtrf", 40, "mTRF"))
    W("")
    W("**Table 13b. Broken down by regular vs counter — mTRF**")
    W("")
    W(criteria_split_table(crit_s5s6, "mtrf", 20))
    W("")
    W("### Results — Encoder")
    W("")
    W("**Table 14. MMN-present counts per criterion, by model — Encoder**")
    W("")
    W(criteria_count_table(crit_s5s6, "encoder", 40, "encoder"))
    W("")
    W("**Table 14b. Broken down by regular vs counter — encoder**")
    W("")
    W(criteria_split_table(crit_s5s6, "encoder", 20))
else:
    W("*mmn_criteria_s5_s6.csv not found — run Step 6 analysis scripts first.*")
W("")
W("---")
W("")

# ---- Section 5: ROI sensitivity ----
roi_df = load_roi_variants()
W("## Section 5 — ROI sensitivity (updated, n/20 methods per model)")
W("")
if roi_df is not None:
    for mapping, label in [("mtrf","mTRF"), ("encoder","Encoder")]:
        for level, lvl_label in [("electrodes","Electrodes"), ("parcels","Parcels")]:
            tnum = {"mtrf_electrodes":"15","mtrf_parcels":"17",
                    "encoder_electrodes":"16","encoder_parcels":"18"}[f"{mapping}_{level}"]
            W(f"**Table {tnum}. {lvl_label}, {label}** (n/20 methods per model)")
            W("")
            W(roi_count_table(roi_df, mapping, level, n_all))
            W("")
else:
    W("*mmn_roi_variants.csv not found.*")
W("")
W("---")
W("")

# ---- Section 6: Fz/central criteria ----
crit_fzc = load_criteria_s5s6("mmn_criteria_s5_s6_fz_central.csv")
W("## Section 6 — Shape criteria under Fz/central ROI (updated, 20 methods)")
W("")
if crit_fzc is not None:
    for mapping, label, tnum_elec, tnum_parc in [
            ("mtrf","mTRF","25","26"), ("encoder","encoder","27","28")]:
        for level, lvl_label, tnum in [("electrodes","Electrodes — Fz only", tnum_elec),
                                        ("parcels","Parcels — central only", tnum_parc)]:
            sub = crit_fzc[(crit_fzc["mapping"]==mapping) & (crit_fzc["level"]==level)]
            rows = []
            totals = {c: 0 for c in CRITERIA}
            for m in MODELS:
                ms = sub[sub["model"]==f"whisper-{m}"]
                row = [m]
                for c in CRITERIA:
                    cnt = ms[c].sum()
                    totals[c] += cnt
                    row.append(pct(cnt, n_all))
                rows.append(row)
            total_n = len(MODELS) * n_all
            rows.append(["**Total**"] + [pct(totals[c], total_n) for c in CRITERIA])
            W(f"**Table {tnum}. {lvl_label} — {label}** (n/{n_all} per model)")
            W("")
            W(md_table(["Model"]+[f"{c} (n/{n_all})" for c in CRITERIA], rows))
            W("")

    W("### Combined (parcels + electrodes, Fz/central)")
    W("")
    for mapping, label, t_m, t_e in [("mtrf","mTRF","29","30"),
                                       ("encoder","encoder","30b","31")]:
        rows = []
        totals = {c: 0 for c in CRITERIA}
        for m in MODELS:
            ms = crit_fzc[(crit_fzc["mapping"]==mapping) & (crit_fzc["model"]==f"whisper-{m}")]
            row = [m]
            for c in CRITERIA:
                cnt = ms[c].sum()
                totals[c] += cnt
                row.append(pct(cnt, 2*n_all))
            rows.append(row)
        rows.append(["**Total**"] + [pct(totals[c], len(MODELS)*2*n_all) for c in CRITERIA])
        W(f"**Table {t_m}. {label}, combined (Fz + central)** (n/{2*n_all} per model)")
        W("")
        W(md_table(["Model"]+[f"{c} (n/{2*n_all})" for c in CRITERIA], rows))
        W("")
else:
    W("*mmn_criteria_s5_s6_fz_central.csv not found.*")
W("")
W("---")
W("")
W("*Generated by `scripts/generate_counter_analysis_docs.py`.*")

# Write file
out_path = OUT_DIR / "results_analysis_with_counter.md"
out_path.write_text("\n".join(lines))
print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Build MMN_pipeline_analysis_decisions_notes_with_counter.md
# ---------------------------------------------------------------------------

lines2 = []
W2 = lines2.append

W2("# MMN Pipeline Analysis Decisions Notes — With Counterbalanced Methods")
W2("")
W2("> **Extends `aux/MMN_pipeline_analysis_decisions_notes_062226.md`.**")
W2("> All five decisions remain the same; Tables 29/30 and all per-model counts are")
W2("> updated to reflect 20 methods per model (10 regular + 10 counter).")
W2("> A new Decision 0 (counterbalancing interpretation) is added.")
W2("")
W2("---")
W2("")
W2("## Decision 0 (new) — Interpreting counterbalanced results")
W2("")
W2("The counterbalanced pairs provide a within-stimulus control:")
W2("")
W2("- If a model shows **bidirectional MMN** (both regular and counter positive),")
W2("  this is strong evidence the model detects deviance independent of absolute frequency.")
W2("- If a model shows **regular-only MMN**, the result is consistent with a frequency")
W2("  preference artifact — the 'deviant' frequency may simply evoke a stronger response")
W2("  in the model regardless of context.")
W2("- If a model shows **counter-only MMN**, the original result was likely a false positive")
W2("  driven by a suppressed response to what was the 'standard' frequency.")
W2("")
W2("See Section 0b of `results_analysis_with_counter.md` for the full per-model,")
W2("per-method breakdown.")
W2("")
W2("---")
W2("")
W2("## 1. Stimulus Set")
W2("")
W2("Unchanged — see `MMN_pipeline_analysis_decisions_notes_062226.md` §1.")
W2("The 10 original and 10 counterbalanced pairs all use Definition-2-style stimuli")
W2("built from Definition-1-sourced frequency pairs.")
W2("")
W2("---")
W2("")
W2("## 2. MMN Metric (C0–S6) — updated tables (n/40 per model)")
W2("")
W2("Tables 29 and 30 are now over **n/40 per model** (20 methods × 2 levels).")
W2("")

if crit_fzc is not None:
    for mapping, label, tnum in [("mtrf","mTRF","29"), ("encoder","encoder","30")]:
        rows = []
        totals = {c: 0 for c in CRITERIA}
        for m in MODELS:
            ms = crit_fzc[crit_fzc["model"]==f"whisper-{m}"]
            ms = ms[ms["mapping"]==mapping]
            row = [m]
            for c in CRITERIA:
                cnt = ms[c].sum()
                totals[c] += cnt
                row.append(pct(cnt, 2*n_all))
            rows.append(row)
        rows.append(["**Total**"] + [pct(totals[c], len(MODELS)*2*n_all) for c in CRITERIA])
        W2(f"**Table {tnum}. {label}, combined (Fz + central)** (n/{2*n_all} per model)")
        W2("")
        W2(md_table(["Model"]+[f"{c} (n/{2*n_all})" for c in CRITERIA], rows))
        W2("")
    W2("### What changed vs. the original tables (n/20)")
    W2("")
    W2("- **Denominator doubles** from n/20 to n/40.")
    W2("- If the counterbalanced results closely track the regular results, the")
    W2("  *fraction* (n/40) stays similar to the old (n/20), indicating robustness.")
    W2("- If regular and counter diverge substantially, the fraction changes —")
    W2("  see the regular/counter split tables in `results_analysis_with_counter.md`")
    W2("  Section 0b for the breakdown.")
    W2("")
else:
    W2("*mmn_criteria_s5_s6_fz_central.csv not found — run Step 6 first.*")
    W2("")

W2("---")
W2("")
W2("## 3. Mapping Method (Encoder vs. mTRF)")
W2("")
W2("Fit-quality table unchanged — see `MMN_pipeline_analysis_decisions_notes_062226.md` §3.")
W2("The mTRF mapping is fit only once on naturalistic speech EEG (D2); adding counter")
W2("methods adds new stimuli to apply the *same* fitted mapping to, so the mTRF vs.")
W2("encoder fit-quality comparison is unaffected.")
W2("")
W2("**Updated MMN-positive counts (C0, mTRF combined, n/40 per model):**")
W2("")
if crit_fzc is not None:
    for mapping, label in [("mtrf","mTRF"), ("encoder","encoder")]:
        W2(f"*{label}:*")
        W2("")
        sub = crit_fzc[crit_fzc["mapping"]==mapping]
        row_data = []
        for m in MODELS:
            ms = sub[sub["model"]==f"whisper-{m}"]
            cnt = ms["C0"].sum()
            row_data.append([m, pct(cnt, 2*n_all)])
        W2(md_table(["Model", f"C0 MMN (n/{2*n_all})"], row_data))
        W2("")
W2("")
W2("---")
W2("")
W2("## 4. Analysis Level (Parcels vs. Electrodes)")
W2("")
W2("Unchanged — see `MMN_pipeline_analysis_decisions_notes_062226.md` §4.")
W2("")
W2("---")
W2("")
W2("## 5. ROI Definition")
W2("")
W2("Unchanged recommendation: Fz for electrodes, central for parcels.")
W2("Updated combined ROI tables below use n/20 per model.")
W2("")

if roi_df is not None:
    for level, variants in [
        ("electrodes", ["Fz","FCz","Fz_FCz","current7"]),
        ("parcels",    ["frontal","temporal","central","current2"])
    ]:
        tnum = "23" if level=="electrodes" else "24"
        combined = roi_df[roi_df["level"]==level]
        rows = []
        totals = {v: 0 for v in variants}
        for m in MODELS:
            ms = combined[combined["model"]==f"whisper-{m}"]
            row = [m]
            for v in variants:
                cnt = ms[ms["roi_variant"]==v]["mmn"].sum()
                totals[v] += cnt
                row.append(pct(cnt, 2*n_all))
            rows.append(row)
        rows.append(["**Total**"]+[pct(totals[v], len(MODELS)*2*n_all) for v in variants])
        W2(f"**Table {tnum}. {level.capitalize()}, combined (mTRF + Encoder)** (n/{2*n_all} per model)")
        W2("")
        W2(md_table(["Model"]+variants, rows))
        W2("")
else:
    W2("*mmn_roi_variants.csv not found.*")
    W2("")

W2("")
W2("---")
W2("")
W2("*Generated by `scripts/generate_counter_analysis_docs.py`.*")

out_path2 = OUT_DIR / "MMN_pipeline_analysis_decisions_notes_062226_with_counter.md"
out_path2.write_text("\n".join(lines2))
print(f"Wrote {out_path2}")
print("Done. Files written to aux/analysis_with_counter/")
