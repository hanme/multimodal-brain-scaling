"""Cross-model C0-S6 criteria summary tables under the single-site Fz/central ROI -- reproduces
aux/results_analysis.md Section 6 Tables 25-28 (rows = model, columns = C0..S6, cells = count/10).

WHY THIS EXISTS
---------------
scripts/analyze_mmn_criteria_fz_central.py and scripts/analyze_mmn_criteria_s5_s6.py
(--roi_variant fz_central) already compute every run's C0-S6 verdict under the Fz (electrodes) /
central (parcels) single-site ROI and write it to outputs/results/mmn_criteria_s5_s6_fz_central.csv
(which carries C0-S3 alongside its own S4/S5/S6 columns, so it alone has everything needed --
no recomputation happens here, this is pure aggregation):

    C0 = current__C0_current        S4 = s4__dip_recovery      (tone-end-relative, NOT
    S1 = current__S1_interior       S5 = global__S5             current__S4_specificity,
    S2 = current__S2_recovery       S6 = global__S6_envelope_recovery  which is retired)
    S3 = current__S3_interior_recovery

This script just groups those booleans by (model, level, mapping), counts how many of the 10
methods are True per criterion, and lays the result out as the 4 tables from Section 6, with
the same Total row/column convention, in both CSV and Markdown.

Usage:
    python scripts/build_mmn_criteria_summary_tables.py \
        --in_csv outputs/results/mmn_criteria_s5_s6_fz_central.csv \
        --out_dir outputs/results
"""

from pathlib import Path
import argparse
import csv

MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]
MODEL_LABEL = {"whisper-tiny": "tiny", "whisper-base": "base",
               "whisper-small": "small", "whisper-medium": "medium"}
CRITERIA_COLUMNS = ["C0", "S1", "S2", "S3", "S4", "S5", "S6"]

# {criterion: column name in mmn_criteria_s5_s6_fz_central.csv}
SOURCE_COLUMN = {
    "C0": "current__C0_current",
    "S1": "current__S1_interior",
    "S2": "current__S2_recovery",
    "S3": "current__S3_interior_recovery",
    "S4": "s4__dip_recovery",
    "S5": "global__S5",
    "S6": "global__S6_envelope_recovery",
}

# (mapping, level) -> (output name, comparator table in results_analysis.md)
TABLES = [
    ("mtrf", "electrodes", "mmn_criteria_summary__mtrf_electrodes", "Table 25"),
    ("mtrf", "parcels", "mmn_criteria_summary__mtrf_parcels", "Table 26"),
    ("encoder", "electrodes", "mmn_criteria_summary__encoder_electrodes", "Table 27"),
    ("encoder", "parcels", "mmn_criteria_summary__encoder_parcels", "Table 28"),
]


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1")


def load_rows(in_csv):
    with open(in_csv, newline="") as f:
        return list(csv.DictReader(f))


def build_table(rows, mapping, level):
    """{model: {criterion: count_true_out_of_10}} for this (mapping, level)."""
    sub = [r for r in rows if r["mapping"] == mapping and r["level"] == level]
    out = {}
    for model in MODEL_ORDER:
        model_rows = [r for r in sub if r["model"] == model]
        assert len(model_rows) == 10, (
            f"{model}/{level}/{mapping}: expected 10 methods, found {len(model_rows)}")
        out[model] = {c: sum(_to_bool(r[SOURCE_COLUMN[c]]) for r in model_rows)
                      for c in CRITERIA_COLUMNS}
    return out


def write_csv(table, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + CRITERIA_COLUMNS + ["Total"])
        col_totals = {c: 0 for c in CRITERIA_COLUMNS}
        grand_total = 0
        for model in MODEL_ORDER:
            counts = table[model]
            row_total = sum(counts.values())
            w.writerow([MODEL_LABEL[model]] + [f"{counts[c]}/10" for c in CRITERIA_COLUMNS]
                      + [f"{row_total}/70"])
            for c in CRITERIA_COLUMNS:
                col_totals[c] += counts[c]
            grand_total += row_total
        w.writerow(["Total"] + [f"{col_totals[c]}/40" for c in CRITERIA_COLUMNS]
                  + [f"{grand_total}/280"])


def render_markdown(title, comparator, table):
    lines = [f"**{title}** (comparator: `aux/results_analysis.md` {comparator})", ""]
    header = ["Model"] + [f"{c} (n/10)" for c in CRITERIA_COLUMNS] + ["Total (n/70)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    col_totals = {c: 0 for c in CRITERIA_COLUMNS}
    grand_total = 0
    for model in MODEL_ORDER:
        counts = table[model]
        row_total = sum(counts.values())
        cells = [MODEL_LABEL[model]] + [f"{counts[c]}/10" for c in CRITERIA_COLUMNS] + [f"{row_total}/70"]
        lines.append("| " + " | ".join(cells) + " |")
        for c in CRITERIA_COLUMNS:
            col_totals[c] += counts[c]
        grand_total += row_total
    total_cells = ["**Total (n/40)**"] + [f"**{col_totals[c]}/40**" for c in CRITERIA_COLUMNS] \
        + [f"**{grand_total}/280**"]
    lines.append("| " + " | ".join(total_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_csv", default="outputs/results/mmn_criteria_s5_s6_fz_central.csv")
    p.add_argument("--out_dir", default="outputs/results")
    args = p.parse_args()

    rows = load_rows(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_sections = []
    for mapping, level, out_name, comparator in TABLES:
        table = build_table(rows, mapping, level)
        write_csv(table, out_dir / f"{out_name}.csv")
        title = f"{level.capitalize()}, {'mTRF' if mapping == 'mtrf' else 'Encoder'}" + \
                (" — Fz only" if level == "electrodes" else " — central only")
        md_sections.append(render_markdown(title, comparator, table))
        print(f"Wrote {out_dir / f'{out_name}.csv'}")

    md_path = out_dir / "mmn_criteria_summary_tables.md"
    md_path.write_text(
        "# MMN criteria summary tables (Fz / central single-site ROI)\n\n"
        "Reproduces `aux/results_analysis.md` Section 6 Tables 25-28. "
        "Source: `outputs/results/mmn_criteria_s5_s6_fz_central.csv`.\n\n"
        + "\n".join(md_sections))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
