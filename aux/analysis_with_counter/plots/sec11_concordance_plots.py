#!/usr/bin/env python
"""Section 11 — cross-model stimulus concordance on the 24-method / 7-model mTRF screen.

Question: do the SAME stimuli drive high responses across all 7 models, and the SAME stimuli drive
low ones? I.e. is the in-silico MMN a property of the *stimulus* (models agree on which oddballs are
easy/hard) or of the *model* (each has its own idiosyncratic favourites)?

Reads outputs/results_24freq_7models/mmn_s7_roi.csv (mTRF only, dip_uv_threshold == 0.25 — `s2` and
`trough_uv` are X-independent, so that filter yields one row per stimulus pair). 24 frequency methods
× {regular, counter} = 48 stimulus pairs per model per site; 7 models. Sites: parcel=frontal,
electrode=FCz.

TERMINOLOGY: the 48 are STIMULUS PAIRS — ordered (standard -> deviant), so 1000->1200 and 1200->1000
are two distinct pairs. The 24 are METHODS; each method contributes one regular and one counter pair.
That split is why 11e/11i can pair the two directions of a method against each other.

SIGN TRAP: trough_uv is deviant-standard µV at the S2 trough latency. NEGATIVE = deeper = HIGHER
response. Rank 1 = most negative = highest response. Asserted at load time, not assumed.

THE CENTRAL TRAP: trough_uv is NOT comparable across models — whisper-large's predicted µV run ~40×
the others (Section 7, caveat 2), wav2vec2-medium ≈ -3.45 µV vs whisper-small ≈ -0.99 µV. Any
cross-model comparison of raw µV measures feature-norm scale, not response. So EVERY cross-model
statistic here is rank-based or within-model-normalised: stimulus pairs are ranked WITHIN each model ×
site, and only the rankings are compared. Raw µV is never pooled, correlated across models, or averaged.

Deviance size = 12·|log2(f_dev/f_std)| semitones and SOA are read from the CANONICAL metadata
(data/metadata/literature_frequency_intensity_duration_metadata.csv, change_type == Frequency,
24 rows) — never a hardcoded frequency dict.

Outputs (to this directory):
  sec11_rank_heatmap.png       — 48 stimulus pairs (rows, sorted by mean rank) × 7 models; cell = that
                                 model's within-model percentile rank. Sequential single hue.
  sec11_pairwise_spearman.png  — 7×7 rank-correlation matrix, diverging palette, neutral gray at 0,
                                 models blocked by family.
  sec11_agreement_histogram.png— "how many of 7 models call it present" (0-7) for S2, observed vs a
                                 chance null that preserves each model's own base rate.
  sec11_floor_agreement.png    — 11g: stimulus pairs with AT LEAST k of 7 models calling the criterion
                                 present, vs the amplitude floor. One line per k (ordinal one-hue ramp).
                                 The visual of Table 50; the k>=7 line dies at the first floor.
  sec11_z_by_method.png        — 11h: per stimulus pair, a box over the 7 models' within-model z,
                                 each model overplotted. Wide boxes = the models disagree.
  sec11_regular_vs_counter.png — 11i: counter z vs regular z, one point per method × model, identity
                                 line. Off the diagonal = the swap changes the response.
  sec11_stats.csv              — every scalar reported in Section 11's tables.
  sec11_stimulus_pairs.csv     — per-pair mean rank / SD / S2 count, both sites.
  sec11_per_pair_agreement.csv — 11f: per-pair model count (/7) for S2 and every S7 floor. This CSV is
                                 now the ONLY home of the full 48-row breakdown (Tables 48a/48b in the
                                 doc drop the SOA and deviance columns).

The second normalisation (11h/11i) is the within-model z: z = (trough_uv - mean)/SD over that model's
OWN 48 stimulus pairs at that site. Like ranks it cancels the cross-model scale, but it keeps magnitude.
NOTE its one statistical trap: z sums to zero within each model, which mechanically induces a small
NEGATIVE regular<->counter correlation (~ -0.06 here). So 11i tests the observed r against a RE-PAIRING
null (randomly re-matching each model's regular pairs to its counter pairs), not against r = 0 — that
distinction flips the FCz verdict from "significant" to n.s.

Colour (dataviz skill; node unavailable, so scripts/validate_palette.js was ported to Python and run):
  * Sequential  = palette.md blue ramp 100->700, light->dark. Validated: lightness monotonic, min
    step dL 0.093 (>= 0.06), hue spread 4.1 deg. The light-end contrast WARN is the ORDINAL floor;
    this is a CONTINUOUS heatmap, where palette.md explicitly allows the lightest step to recede.
  * Diverging   = palette.md "blue <-> red" with the neutral gray midpoint #f0efec (OKLCh C=0.004).
    The red arm is generated to MIRROR the blue arm's OKLCh lightness at the palette's red hue
    (#e34948, hue 24.9 deg): per-step |dL| <= 0.001, hue spread 0.7 deg, poles dE protan 14.2 /
    deutan 15.9 / normal 18.5 — well clear of the dE >= 8 target.
  * Fig 3 pair  = observed #2a78d6 (blue slot 1) vs chance null #898781 (muted ink). CVD dE 15.9
    protan, normal-vision dE 17.8 — both pass. The gray trips the chroma floor BY DESIGN: the null
    is a recessive reference, not a categorical series, so it wears muted ink rather than a hue.
MODEL_STYLE / MLABEL are reused verbatim from sec8b_mtrf_plots.py so model identity is stable
across Sections 8b/8c/10/11.
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = "/Users/sophiesigfstead/Documents/multimodal-brain-scaling-2"
OUT  = f"{REPO}/aux/analysis_with_counter/plots"
CSV  = f"{REPO}/outputs/results_24freq_7models/mmn_s7_roi.csv"
META = f"{REPO}/data/metadata/literature_frequency_intensity_duration_metadata.csv"

# ── model identity: reused verbatim from sec8b_mtrf_plots.py ──────────────────────────────────────
MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large",
               "wav2vec2-medium", "wav2vec2-large"]
MODEL_STYLE = {
    "whisper-tiny":    dict(color="#56B4E9", marker="o"),
    "whisper-base":    dict(color="#0072B2", marker="s"),
    "whisper-small":   dict(color="#009E73", marker="^"),
    "whisper-medium":  dict(color="#E69F00", marker="D"),
    "whisper-large":   dict(color="#D55E00", marker="v"),
    "wav2vec2-medium": dict(color="#CC79A7", marker="P"),
    "wav2vec2-large":  dict(color="#000000", marker="X"),
}
MLABEL = {"whisper-tiny": "whisper tiny", "whisper-base": "whisper base",
          "whisper-small": "whisper small", "whisper-medium": "whisper medium",
          "whisper-large": "whisper large", "wav2vec2-medium": "wav2vec2 medium",
          "wav2vec2-large": "wav2vec2 large"}
WHISPER  = [m for m in MODEL_ORDER if m.startswith("whisper")]
WAV2VEC2 = [m for m in MODEL_ORDER if m.startswith("wav2vec2")]

SITES   = [("parcel", "frontal"), ("electrode", "FCz")]
SITE_T  = {"frontal": "parcel — frontal", "FCz": "electrode — FCz"}
N_COND, N_MODELS = 48, 7
HEADLINE_X = 0.5
# The amplitude floors swept in 11f/11g. Same list as Section 8b; includes the 0.25 and 2.5 µV
# bookends (house rule). S7@X = S2 AND trough_uv <= -X, so the sets are nested:
# S7@2.5 ⊆ S7@1.5 ⊆ … ⊆ S7@0.25 ⊆ S2.
FLOORS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.5]
NPERM = 5000
SEED  = 11


def fx(x):
    """Floor label carrying at least one decimal, so 1.0 reads 'S7@1.0' and never 'S7@1'."""
    s = f"{x:g}"
    return f"{x:.1f}" if "." not in s else s

# ── chart chrome (palette.md) ─────────────────────────────────────────────────────────────────────
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_RED  = ["#fad6d2", "#f1aea8", "#e4857e", "#d75853", "#b13f3c", "#892b2a", "#621b1a"]
GRAY_MID = "#f0efec"
OBS_C, NULL_C = "#2a78d6", "#898781"
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("div_rb", DIV_RED[::-1] + [GRAY_MID] + SEQ_BLUE)

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


# ══ load ══════════════════════════════════════════════════════════════════════════════════════════
def load_metadata():
    """method_id -> deviance (semitones), SOA, std/dev Hz — from the CANONICAL metadata CSV."""
    md = pd.read_csv(META)
    md = md[md.change_type == "Frequency"].copy()
    assert len(md) == 24, f"expected 24 Frequency methods in canonical metadata, got {len(md)}"
    md["semitones"] = 12.0 * np.abs(np.log2(md.deviant_freq / md.standard_freq))
    k = md.method_id.astype(int)
    return (dict(zip(k, md.semitones)), dict(zip(k, md.standard_soa)),
            dict(zip(k, md.standard_freq)), dict(zip(k, md.deviant_freq)))


def load_screen(ST):
    """Per-site {model × stimulus pair} wide frames of trough_uv and s2. Verifies 48 × 7 before returning."""
    d = pd.read_csv(CSV)
    d = d[(d.mapping == "mtrf") & (d.dip_uv_threshold == 0.25)].copy()
    d["method_id"] = d.method.str.extract(r"method_(\d+)").astype(int)
    d["semitones"] = d.method_id.map(ST)
    assert not d.semitones.isna().any(), "a screened method is missing from the canonical metadata"
    frames = {}
    for kind, roi in SITES:
        s = d[(d.roi_kind == kind) & (d.roi == roi)].copy()
        assert len(s) == N_COND * N_MODELS, f"{kind}/{roi}: expected 336 rows, got {len(s)}"
        assert not s.duplicated(["model", "method"]).any(), f"{kind}/{roi}: duplicate model×pair"
        uv = s.pivot(index="method", columns="model", values="trough_uv")[MODEL_ORDER]
        s2 = s.pivot(index="method", columns="model", values="s2")[MODEL_ORDER].astype(bool)
        assert uv.shape == (N_COND, N_MODELS), f"{kind}/{roi}: {uv.shape} != (48, 7)"
        assert not uv.isna().any().any(), f"{kind}/{roi}: missing trough_uv"
        assert set(uv.columns) == set(MODEL_ORDER) and len(set(uv.index)) == 48
        frames[roi] = dict(uv=uv, s2=s2)
        print(f"  VERIFIED {kind:9s} {roi:7s}: {uv.shape[0]} stimulus pairs × {uv.shape[1]} models")
    return frames


def depth_ranks(uv):
    """Rank the stimulus pairs WITHIN each model. Rank 1 = most negative trough = HIGHEST response."""
    r = uv.rank(axis=0, method="average", ascending=True)
    for m in uv.columns:                       # the sign trap, asserted rather than assumed
        assert uv[m].idxmin() == r[m].idxmin(), "rank 1 must be the most NEGATIVE (deepest) trough"
        assert uv[m].idxmax() == r[m].idxmax(), "rank 48 must be the least negative (shallowest)"
    return r


def response_pct(uv):
    """Within-model percentile of RESPONSE HEIGHT: 100 = deepest = highest response, 0 = shallowest."""
    r = (-uv).rank(axis=0, method="average", ascending=True)
    return (r - 1) / (len(r) - 1) * 100.0


def within_model_z(uv):
    """z = (trough_uv − mean) / SD of that model's OWN 48-pair trough distribution at that site.

    The other legitimate normalisation alongside ranks: it puts every model on its own scale, so
    whisper-large's ~40× µV cancels. Sign is inherited from trough_uv — z < 0 = deeper than that
    model's average = HIGHER response. Standardised over all 48 stimulus pairs (not the S2-passing
    subset), so every model contributes to every method's box.
    """
    z = (uv - uv.mean(axis=0)) / uv.std(axis=0)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9) and np.allclose(z.std(axis=0), 1, atol=1e-9)
    for m in uv.columns:                       # sign trap again: deepest µV must be the lowest z
        assert uv[m].idxmin() == z[m].idxmin(), "z < 0 must mean deeper (higher response)"
    return z


def criterion_frames(uv, s2):
    """The 7 binary criteria of 11f/11g: S2, then S7@X for each floor. Returns {name: 48×7 bool}."""
    out = {"S2": s2.astype(bool)}
    for x in FLOORS:
        out[f"S7@{fx(x)}"] = (s2 & (uv <= -x)).astype(bool)
    return out


# ══ statistics ════════════════════════════════════════════════════════════════════════════════════
def kendall_w(R):
    """Kendall's W over an items × raters rank matrix, with the tie correction."""
    n, m = R.shape
    Ri = R.sum(axis=1)
    S = ((Ri - Ri.mean()) ** 2).sum()
    T = 0.0
    for j in range(m):
        _, cnt = np.unique(R[:, j], return_counts=True)
        T += (cnt ** 3 - cnt).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n) - m * T)


def kendall_w_perm(R, nperm=NPERM, seed=SEED):
    """W + permutation null: each model's ranking shuffled independently (H0 = unrelated orderings)."""
    rng = np.random.default_rng(seed)
    obs = kendall_w(R)
    null = np.array([kendall_w(np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])]))
                     for _ in range(nperm)])
    return obs, (1 + (null >= obs).sum()) / (nperm + 1), null


def fleiss_kappa(cnt, m=N_MODELS):
    """Chance-corrected agreement over binary calls. cnt = per-pair count of 'present' (0..m)."""
    M = np.column_stack([cnt, m - cnt])
    n = M.shape[0]
    p_j = M.sum(axis=0) / (n * m)
    P_i = ((M ** 2).sum(axis=1) - m) / (m * (m - 1))
    Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
    return (Pbar - Pe) / (1 - Pe), Pbar, Pe


def agreement_null(A, nperm=NPERM, seed=SEED):
    """Null that PRESERVES each model's own base rate: shuffle its calls across stimulus pairs."""
    rng = np.random.default_rng(seed)
    hist = np.zeros((nperm, N_MODELS + 1)); kap = np.empty(nperm); unan = np.empty(nperm)
    for t in range(nperm):
        P = np.column_stack([rng.permutation(A[:, j]) for j in range(A.shape[1])])
        c = P.sum(axis=1)
        hist[t] = np.bincount(c, minlength=N_MODELS + 1)
        kap[t] = fleiss_kappa(c)[0]
        unan[t] = ((c == 0) | (c == N_MODELS)).sum()
    return hist, kap, unan


# ══ figures ═══════════════════════════════════════════════════════════════════════════════════════
def cond_label(cond, ST, FSTD, FDEV):
    mid = int(cond.split("_")[1]); counter = cond.endswith("_counter")
    lo, hi = FSTD[mid], FDEV[mid]
    hz = f"{hi:.0f}→{lo:.0f}" if counter else f"{lo:.0f}→{hi:.0f}"
    return f"m{mid:02d}{'c' if counter else ' '} {hz:>11s} {ST[mid]:5.2f}st"


def fig_rank_heatmap(frames, ST, FSTD, FDEV, path):
    """Stimulus pairs × models; cell = within-model percentile of response height (100 = deepest)."""
    # each panel carries its OWN row order (sorted by that site's mean rank), so both panels need
    # their own row labels — hence the wide gutter rather than a shared axis.
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 11.6), gridspec_kw=dict(wspace=0.60))
    for ax, (kind, roi) in zip(axes, SITES):
        pct = response_pct(frames[roi]["uv"])
        order = pct.mean(axis=1).sort_values(ascending=False).index      # consensus-high at the top
        P = pct.loc[order]
        im = ax.imshow(P.to_numpy(), cmap=CMAP_SEQ, vmin=0, vmax=100, aspect="auto",
                       interpolation="nearest")
        # 2px surface gap between cells (marks-and-anatomy spacer rule)
        ax.set_xticks(np.arange(-.5, N_MODELS, 1), minor=True)
        ax.set_yticks(np.arange(-.5, N_COND, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(N_MODELS))
        ax.set_xticklabels([MLABEL[m] for m in P.columns], rotation=42, ha="right", fontsize=8)
        for t, m in zip(ax.get_xticklabels(), P.columns):
            t.set_color(MODEL_STYLE[m]["color"])
            if m == "wav2vec2-large":
                t.set_color(INK)
        ax.set_yticks(range(N_COND))
        ax.set_yticklabels([cond_label(c, ST, FSTD, FDEV) for c in order],
                           fontsize=5.6, fontfamily="monospace", color=INK2)
        ax.set_title(SITE_T[roi], fontsize=10, color=INK, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02, shrink=0.55)
    cb.set_label("within-model percentile of response height\n(100 = deepest trough = highest response)",
                 fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=BASELINE)
    fig.suptitle("Do the same stimuli drive high responses in every model?\n"
                 "Stimulus pairs ranked WITHIN each model (raw µV is not cross-model comparable); "
                 "rows sorted by mean rank.\nConsistent rows ⇒ stimulus-driven · noisy rows ⇒ model-driven",
                 fontsize=11, color=INK, y=0.985)
    fig.text(0.5, 0.008, "m## = method id · 'c' = counter (std/dev swapped) · st = deviance size "
                         "(12·|log₂(f_dev/f_std)|).  mTRF, 24 methods × {regular, counter}.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_pairwise_spearman(frames, path):
    """7×7 Spearman of within-model ranks. Diverging, neutral gray at 0, models blocked by family."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.6), gridspec_kw=dict(wspace=0.06))
    # full -1..1 range on purpose: rescaling to the observed span would visually inflate what are
    # in fact weak correlations. The printed cell values carry the precision.
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for k, (ax, (kind, roi)) in enumerate(zip(axes, SITES)):
        C = depth_ranks(frames[roi]["uv"]).corr(method="spearman").loc[MODEL_ORDER, MODEL_ORDER]
        A = C.to_numpy().copy()
        np.fill_diagonal(A, np.nan)                       # the diagonal is 1 by construction, not data
        im = ax.imshow(A, cmap=CMAP_DIV, norm=norm, aspect="equal", interpolation="nearest")
        for i in range(N_MODELS):
            for j in range(N_MODELS):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=MUTED)
                    continue
                v = A[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.6,
                        color="#ffffff" if abs(v) > 0.45 else INK2)
        ax.set_xticks(np.arange(-.5, N_MODELS, 1), minor=True)
        ax.set_yticks(np.arange(-.5, N_MODELS, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(N_MODELS))
        ax.set_xticklabels([MLABEL[m] for m in MODEL_ORDER], rotation=42, ha="right", fontsize=8)
        for tl, m in zip(ax.get_xticklabels(), MODEL_ORDER):
            tl.set_color(INK if m == "wav2vec2-large" else MODEL_STYLE[m]["color"])
        ax.set_yticks(range(N_MODELS))
        if k == 0:                       # shared categorical axis: label it once, on the left panel
            ax.set_yticklabels([MLABEL[m] for m in MODEL_ORDER], fontsize=8)
            for tl, m in zip(ax.get_yticklabels(), MODEL_ORDER):
                tl.set_color(INK if m == "wav2vec2-large" else MODEL_STYLE[m]["color"])
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        # family block: whisper = first 5, wav2vec2 = last 2
        for (a, b) in [(-0.5, 4.5), (4.5, 6.5)]:
            ax.add_patch(plt.Rectangle((a, a), b - a, b - a, fill=False, ec=INK, lw=1.6, zorder=5))
        ax.text(2.0, -0.92, "whisper", ha="center", fontsize=8, color=MUTED, style="italic")
        ax.text(5.5, -0.92, "wav2vec2", ha="center", fontsize=8, color=MUTED, style="italic")
        ax.set_title(SITE_T[roi], fontsize=10, color=INK, pad=30)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.02, shrink=0.72,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Spearman ρ of within-model stimulus-pair ranks", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=BASELINE)
    fig.suptitle("Do any two models rank the 48 stimuli the same way?\n"
                 "Pairwise Spearman of within-model ranks · gray ≈ 0 = no relationship · "
                 "boxes group the two model families", fontsize=11, color=INK, y=1.06)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_agreement_histogram(binary, path):
    """How many of 7 models call each stimulus pair S2-present — observed vs the base-rate-preserving null."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True)
    ks = np.arange(N_MODELS + 1)
    for ax, (kind, roi) in zip(axes, SITES):
        b = binary[(roi, "S2")]
        ax.bar(ks, b["obs_hist"], width=0.66, color=OBS_C, zorder=3,
               label="observed", edgecolor=SURFACE, linewidth=1.4)
        ax.plot(ks, b["null_mean"], color=NULL_C, lw=2, marker="o", ms=5, zorder=5,
                label="chance null (mean)", mec=SURFACE, mew=1.2)
        ax.fill_between(ks, b["null_lo"], b["null_hi"], color=NULL_C, alpha=0.22, zorder=2, lw=0,
                        label="chance null (95% interval)")
        ax.set_xticks(ks)
        ax.set_xlabel("models calling S2 present  (of 7)", fontsize=9, color=INK2)
        ax.set_title(f"{SITE_T[roi]}   ·   Fleiss κ = {b['kappa']:+.3f} "
                     f"({'n.s.' if b['p_kappa'] >= .05 else f'p = {b_fmt(b)}'})",
                     fontsize=9.5, color=INK, pad=7)
        ax.grid(axis="y", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.annotate(f"unanimous (0 or 7): {b['obs_unan']}/48 observed\n"
                    f"vs {b['null_unan']:.1f} by chance  (p = {b['p_unan']:.2f})",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.8, color=INK2,
                    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, lw=0.8))
    axes[0].set_ylabel("stimulus pairs  (of 48)", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    fig.suptitle("“Most models agree” is what chance already predicts\n"
                 "S2 base rates are high (~78% frontal / ~79% FCz), so the null puts most stimulus pairs "
                 "at 5–7 of 7 too", fontsize=11, color=INK, y=1.04)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def b_fmt(b):
    p = b["p_kappa"]
    return f"{p:.4f}" if p >= 1e-4 else "<1e-4"


def cumulative_agreement(uv, s2):
    """Stimulus pairs with AT LEAST k of 7 models calling each criterion present, k = 7..1.

    The {>=7} set nests inside {>=6} inside … inside {>=1}, so each row is non-decreasing left to
    right; it is the reverse cumulative sum of the Table-49 distribution. Both are asserted.
    """
    crits = criterion_frames(uv, s2)
    rows = {}
    for nm, B in crits.items():
        cnt = B[MODEL_ORDER].to_numpy().astype(bool).sum(axis=1)
        cum = [int((cnt >= k).sum()) for k in range(N_MODELS, 0, -1)]
        assert all(a <= b for a, b in zip(cum, cum[1:])), f"{nm}: nesting violated"
        obs = np.bincount(cnt, minlength=N_MODELS + 1)
        assert cum == [int(obs[k:].sum()) for k in range(N_MODELS, 0, -1)], \
            f"{nm}: cumulative != reverse cumsum of the distribution"
        rows[nm] = cum
    return pd.DataFrame(rows, index=[f">={k}" for k in range(N_MODELS, 0, -1)]).T


def fig_floor_agreement(frames, path):
    """11g — how fast agreement dies as the amplitude floor rises. Survival curves, one per k.

    k (">=1" … ">=7") is ORDINAL — the tiers have an order — so per the dataviz form rule it takes a
    one-hue ramp rather than categorical hues, and the ramp is validated with the ORDINAL rules
    (lightest step must clear 2:1 vs surface; min step dL >= 0.06). Steps run from ramp fraction 2/6
    (#6da7ec, 2.44:1) to the 700 end: dL_min = 0.062, hue spread 3.3 deg — passes.
    """
    ks = list(range(N_MODELS, 0, -1))                       # 7 … 1
    ramp = [CMAP_SEQ(t) for t in np.linspace(2 / 6, 1.0, N_MODELS)]
    kcol = {k: ramp[i] for i, k in enumerate(ks[::-1])}      # >=1 lightest … >=7 darkest
    marks = {7: "X", 6: "P", 5: "v", 4: "D", 3: "^", 2: "s", 1: "o"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)
    xs = np.arange(len(FLOORS) + 1)
    xlab = ["S2\n(no floor)"] + [f"{fx(x)}" for x in FLOORS]
    for ax, (kind, roi) in zip(axes, SITES):
        C = cumulative_agreement(frames[roi]["uv"], frames[roi]["s2"])
        for k in ks:
            y = C[f">={k}"].to_numpy()
            ax.plot(xs, y, color=kcol[k], marker=marks[k], ms=5.5, lw=2, zorder=3 + k,
                    mec=SURFACE, mew=0.7, label=f"≥ {k} model" + ("s" if k > 1 else ""))
        # x is ORDINAL (a criterion sweep, incl. the floorless S2), so the ticks are evenly spaced
        # rather than placed on a µV scale — same convention as Section 8b.
        ax.set_xticks(xs); ax.set_xticklabels(xlab, fontsize=8)
        ax.set_xlabel("amplitude floor X  (µV)   —   S7@X = S2 AND trough ≤ −X", fontsize=8.5, color=INK2)
        ax.set_ylim(-2, 50)
        ax.set_title(SITE_T[roi], fontsize=10, color=INK, pad=6)
        ax.grid(axis="y", lw=0.6, zorder=0); ax.set_axisbelow(True)
    axes[0].set_ylabel("stimulus pairs  (of 48)", fontsize=9, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=7, frameon=False,
               fontsize=8.3, handletextpad=0.35, columnspacing=1.4)
    fig.suptitle("Agreement dies as soon as an amplitude floor is applied\n"
                 "Stimulus pairs on which AT LEAST k of 7 models call the criterion present · "
                 "the {≥7} set nests inside {≥6} inside … inside {≥1}",
                 fontsize=11, color=INK, y=1.03)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_z_by_method(frames, ST, FSTD, FDEV, path):
    """11h — per stimulus pair, the spread of within-model z across the 7 models. Box + the 7 points."""
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 12.2), gridspec_kw=dict(wspace=0.52))
    for ax, (kind, roi) in zip(axes, SITES):
        z = within_model_z(frames[roi]["uv"])
        order = z.median(axis=1).sort_values().index          # deepest (most negative) median at top
        Z = z.loc[order]
        pos = np.arange(len(order))
        bp = ax.boxplot([Z.loc[c].to_numpy() for c in order], positions=pos, vert=False,
                        widths=0.62, showfliers=False, patch_artist=True, zorder=2)
        for box in bp["boxes"]:
            box.set(facecolor="#eef4fd", edgecolor=BASELINE, linewidth=0.9)
        for part in ("whiskers", "caps"):
            for ln in bp[part]:
                ln.set(color=BASELINE, linewidth=0.9)
        for med in bp["medians"]:
            med.set(color=INK, linewidth=1.8)
        for j, m in enumerate(Z.columns):                     # the 7 models, identity by colour+marker
            st = MODEL_STYLE[m]
            ax.scatter(Z[m].to_numpy(), pos, s=26, c=st["color"], marker=st["marker"],
                       edgecolors=SURFACE, linewidths=0.6, zorder=4, alpha=0.95)
        ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(pos)
        ax.set_yticklabels([cond_label(c, ST, FSTD, FDEV) for c in order],
                           fontsize=5.8, fontfamily="monospace", color=INK2)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.invert_yaxis()
        ax.set_xlabel("within-model z of the S2/S7 trough      ←  deeper = higher response",
                      fontsize=8.5, color=INK2)
        ax.set_title(SITE_T[roi], fontsize=10, color=INK, pad=8)
        ax.grid(axis="x", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                      ls="", ms=6, mec=SURFACE, mew=0.6, label=MLABEL[m]) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=7,
               frameon=False, fontsize=8.5, handletextpad=0.35, columnspacing=1.5)
    fig.suptitle("Each model on its OWN scale: z of the trough vs that model's 48-pair mean/SD\n"
                 "Box = spread across the 7 models for one stimulus pair · a tight box would mean the "
                 "models agree about that stimulus", fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.055, "m## = method id · 'c' = counter (std/dev swapped) · st = deviance size.  "
                         "z < 0 = deeper than that model's own average.  mTRF, 24 methods × "
                         "{regular, counter}.", ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_regular_vs_counter(frames, path):
    """11i — is 1000→1500 the same response as 1500→1000? One point per method × model."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9))
    stats_out = {}
    for ax, (kind, roi) in zip(axes, SITES):
        z = within_model_z(frames[roi]["uv"])
        xs, ys = [], []
        for m in MODEL_ORDER:
            reg = {int(c.split("_")[1]): z[m][c] for c in z.index if not c.endswith("_counter")}
            ctr = {int(c.split("_")[1]): z[m][c] for c in z.index if c.endswith("_counter")}
            ks = sorted(reg)
            assert len(ks) == 24 and set(ks) == set(ctr), "regular/counter pairing incomplete"
            x = np.array([reg[k] for k in ks]); y = np.array([ctr[k] for k in ks])
            st = MODEL_STYLE[m]
            ax.scatter(x, y, s=30, c=st["color"], marker=st["marker"], edgecolors=SURFACE,
                       linewidths=0.6, zorder=3, alpha=0.9, label=MLABEL[m])
            xs.append(x); ys.append(y)
        X, Y = np.concatenate(xs), np.concatenate(ys)
        lim = float(max(np.abs(np.r_[X, Y]))) * 1.08
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1.4, ls=(0, (5, 3)), zorder=2,
                label="identity (same response)")
        ax.axhline(0, color=GRID, lw=0.8, zorder=1); ax.axvline(0, color=GRID, lw=0.8, zorder=1)
        r, pr = stats.pearsonr(X, Y)
        rho, prho = stats.spearmanr(X, Y)
        stats_out[roi] = dict(r=r, p_r=pr, rho=rho, p_rho=prho, n=len(X))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("regular  z  (std → dev)", fontsize=9, color=INK2)
        ax.set_ylabel("counter  z  (dev → std)", fontsize=9, color=INK2)
        pf = lambda v: f"{v:.3f}" if v >= 1e-3 else f"{v:.1e}"
        ax.set_title(f"{SITE_T[roi]}   ·   n = {len(X)} (24 methods × 7 models)\n"
                     f"Pearson r = {r:+.3f} (p = {pf(pr)})  ·  Spearman ρ = {rho:+.3f} "
                     f"(p = {pf(prho)})", fontsize=9, color=INK, pad=8)
        ax.grid(lw=0.6, zorder=0); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=8, frameon=False,
               fontsize=8.3, handletextpad=0.35, columnspacing=1.3)
    fig.suptitle("Is 1000→1500 the same response as 1500→1000?\n"
                 "Each point is one method in one model, on that model's own z scale · "
                 "on the diagonal ⇒ the swap changes nothing",
                 fontsize=11, color=INK, y=1.04)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return stats_out


# ══ main ══════════════════════════════════════════════════════════════════════════════════════════
def main():
    print("Section 11 — cross-model stimulus concordance (24-method / 7-model mTRF screen)")
    ST, SOA, FSTD, FDEV = load_metadata()
    print(f"  canonical metadata: 24 Frequency methods, {len(set(np.round(list(ST.values()), 2)))} distinct deviance sizes")
    frames = load_screen(ST)
    rows = []
    R = lambda roi: depth_ranks(frames[roi]["uv"])

    # ── 1. continuous concordance ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("1. CONTINUOUS CONCORDANCE — Kendall's W over the 48 stimulus pairs (permutation null, "
          f"{NPERM} shuffles)")
    print("=" * 92)
    print(f"{'site':10s} {'W (7 models)':>13s} {'p':>9s} {'null mean':>10s} {'W (6, no wh-large)':>19s} "
          f"{'p':>9s} {'W (5 whisper)':>14s} {'p':>9s}")
    for kind, roi in SITES:
        uv = frames[roi]["uv"]
        W7, p7, null7 = kendall_w_perm(R(roi).to_numpy())
        W6, p6, _ = kendall_w_perm(depth_ranks(uv.drop(columns=["whisper-large"])).to_numpy())
        W5, p5, _ = kendall_w_perm(depth_ranks(uv[WHISPER]).to_numpy())
        print(f"{roi:10s} {W7:13.3f} {p7:9.4g} {null7.mean():10.3f} {W6:19.3f} {p6:9.4g} "
              f"{W5:14.3f} {p5:9.4g}")
        rows += [dict(site=roi, stat="kendall_W_7", value=W7, p=p7, extra=f"null_mean={null7.mean():.3f}"),
                 dict(site=roi, stat="kendall_W_6_no_wlarge", value=W6, p=p6, extra=""),
                 dict(site=roi, stat="kendall_W_5_whisper", value=W5, p=p5, extra=""),
                 dict(site=roi, stat="kendall_W_null_p95", value=float(np.quantile(null7, .95)), p=np.nan, extra="")]

    print("\n" + "=" * 92)
    print("   PAIRWISE SPEARMAN of within-model ranks (21 pairs)")
    print("=" * 92)
    for kind, roi in SITES:
        C = R(roi).corr(method="spearman")
        allp = [C.loc[a, b] for i, a in enumerate(MODEL_ORDER) for b in MODEL_ORDER[i + 1:]]
        wi_w = [C.loc[a, b] for i, a in enumerate(WHISPER) for b in WHISPER[i + 1:]]
        cross = [C.loc[a, b] for a in WHISPER for b in WAV2VEC2]
        wi_2 = [C.loc[WAV2VEC2[0], WAV2VEC2[1]]]
        print(f"\n{SITE_T[roi]}")
        for nm, v in [("all pairs (21)", allp), ("within-whisper (10)", wi_w),
                      ("whisper↔wav2vec2 (10)", cross), ("within-wav2vec2 (1)", wi_2)]:
            print(f"   {nm:24s} mean={np.mean(v):+.3f}  min={np.min(v):+.3f}  max={np.max(v):+.3f}")
            rows.append(dict(site=roi, stat=f"spearman_{nm.split(' (')[0].replace(' ', '_')}_mean",
                             value=float(np.mean(v)), p=np.nan,
                             extra=f"min={np.min(v):.3f};max={np.max(v):.3f};n={len(v)}"))
        print(C.round(3).to_string())

    # ── 2. binary agreement ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"2. BINARY AGREEMENT — S2 and S7@{HEADLINE_X} vs a null preserving each model's OWN base rate")
    print("=" * 92)
    binary = {}
    for kind, roi in SITES:
        uv, s2 = frames[roi]["uv"], frames[roi]["s2"]
        for crit, B in [("S2", s2), (f"S7@{HEADLINE_X}", s2 & (uv <= -HEADLINE_X))]:
            A = B[MODEL_ORDER].to_numpy().astype(bool)
            cnt = A.sum(axis=1)
            kap, Pbar, Pe = fleiss_kappa(cnt)
            hist, nk, nu = agreement_null(A)
            obs_hist = np.bincount(cnt, minlength=8)
            obs_unan = int(((cnt == 0) | (cnt == N_MODELS)).sum())
            p_kap = (1 + (nk >= kap).sum()) / (NPERM + 1)
            p_unan = (1 + (nu >= obs_unan).sum()) / (NPERM + 1)
            print(f"\n--- {SITE_T[roi]} · {crit} ---")
            print("   base rate/model: " + "  ".join(f"{MLABEL[m].split()[0][0]}{MLABEL[m].split()[1][:3]}"
                                                     f"={A[:, i].mean():.2f}" for i, m in enumerate(MODEL_ORDER)))
            print(f"   pooled base rate = {A.mean():.3f}")
            print(f"   Fleiss κ = {kap:+.3f}  (P̄={Pbar:.3f}, Pe={Pe:.3f})   perm p = {p_kap:.4g}")
            print(f"   unanimous (0/7 or 7/7): observed {obs_unan}/48 vs null mean {nu.mean():.1f} "
                  f"(null p95={np.quantile(nu, .95):.0f})  p = {p_unan:.3g}")
            print("   k        : " + " ".join(f"{k:5d}" for k in range(8)))
            print("   observed : " + " ".join(f"{v:5d}" for v in obs_hist))
            print("   null mean: " + " ".join(f"{v:5.1f}" for v in hist.mean(axis=0)))
            binary[(roi, crit)] = dict(obs_hist=obs_hist, null_mean=hist.mean(axis=0),
                                       null_lo=np.quantile(hist, .025, axis=0),
                                       null_hi=np.quantile(hist, .975, axis=0),
                                       kappa=kap, p_kappa=p_kap, obs_unan=obs_unan,
                                       null_unan=nu.mean(), p_unan=p_unan, base=A.mean())
            rows += [dict(site=roi, stat=f"fleiss_kappa_{crit}", value=kap, p=p_kap,
                          extra=f"base_rate={A.mean():.3f}"),
                     dict(site=roi, stat=f"unanimous_{crit}", value=obs_unan, p=p_unan,
                          extra=f"null_mean={nu.mean():.2f}")]

    # ── 3. consensus stimuli ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("3. CONSENSUS STIMULI — mean within-model percentile of response height (100 = deepest)")
    print("=" * 92)
    cons = {}
    for kind, roi in SITES:
        pct = response_pct(frames[roi]["uv"])
        t = pd.DataFrame({"mean_pct": pct.mean(axis=1), "sd_pct": pct.std(axis=1),
                          "n_s2": frames[roi]["s2"].sum(axis=1)})
        t["method_id"] = [int(c.split("_")[1]) for c in t.index]
        t["dirn"] = ["counter" if c.endswith("_counter") else "regular" for c in t.index]
        t["st"] = t.method_id.map(ST); t["soa"] = t.method_id.map(SOA)
        t["hz"] = [f"{FDEV[m]:.0f}→{FSTD[m]:.0f}" if d == "counter" else f"{FSTD[m]:.0f}→{FDEV[m]:.0f}"
                   for m, d in zip(t.method_id, t.dirn)]
        cons[roi] = t.sort_values("mean_pct", ascending=False)
        cols = ["method_id", "dirn", "st", "hz", "soa", "mean_pct", "sd_pct", "n_s2"]
        print(f"\n### {SITE_T[roi]} — CONSENSUS HIGH (top 6 by mean rank)")
        print(cons[roi].head(6)[cols].to_string(float_format=lambda v: f"{v:.1f}"))
        print(f"\n### {SITE_T[roi]} — CONSENSUS LOW (bottom 6)")
        print(cons[roi].tail(6)[cols].to_string(float_format=lambda v: f"{v:.1f}"))
    j = cons["frontal"][["mean_pct"]].join(cons["FCz"][["mean_pct"]], lsuffix="_fr", rsuffix="_fcz")
    rho, p = stats.spearmanr(j.mean_pct_fr, j.mean_pct_fcz)
    print(f"\nfrontal ↔ FCz agreement on the mean ranking (48 stimulus pairs): ρ = {rho:+.3f}, p = {p:.3g}")
    for k in (6, 12):
        o_hi = len(set(cons["frontal"].head(k).index) & set(cons["FCz"].head(k).index))
        o_lo = len(set(cons["frontal"].tail(k).index) & set(cons["FCz"].tail(k).index))
        print(f"   top-{k} overlap = {o_hi}/{k}   bottom-{k} overlap = {o_lo}/{k}")
        rows += [dict(site="both", stat=f"top{k}_overlap", value=o_hi, p=np.nan, extra=f"/{k}"),
                 dict(site="both", stat=f"bottom{k}_overlap", value=o_lo, p=np.nan, extra=f"/{k}")]
    rows.append(dict(site="both", stat="frontal_vs_fcz_meanrank_rho", value=rho, p=p, extra="n=48"))
    pd.concat({r: cons[r] for r in cons}, names=["site"]).to_csv(f"{OUT}/sec11_stimulus_pairs.csv")
    print(f"\n  wrote {OUT}/sec11_stimulus_pairs.csv")

    # ── 4. is it just deviance size? ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("4. CONTROLLING FOR DEVIANCE SIZE")
    print("=" * 92)
    ids316 = sorted(m for m in ST if abs(ST[m] - 3.16) < 0.01)
    print(f"   3.16 st cluster: {len(ids316)} methods -> {2 * len(ids316)} stimulus pairs (largest balanced block)")
    for kind, roi in SITES:
        uv = frames[roi]["uv"]
        sel = [c for c in uv.index if int(c.split("_")[1]) in ids316]
        assert len(sel) == 20, f"3.16 st block should be 20 stimulus pairs, got {len(sel)}"
        W, p, nullb = kendall_w_perm(depth_ranks(uv.loc[sel]).to_numpy())
        W6, p6, _ = kendall_w_perm(depth_ranks(uv.loc[sel].drop(columns=["whisper-large"])).to_numpy())
        # deviance-residualised ranks over all 48
        st_r = stats.rankdata([ST[int(c.split("_")[1])] for c in uv.index])
        res = {}
        for m in MODEL_ORDER:
            y = stats.rankdata(uv[m].to_numpy())
            sl, ic, *_ = stats.linregress(st_r, y)
            res[m] = y - (ic + sl * st_r)
        Wr, pr, _ = kendall_w_perm(pd.DataFrame(res).rank(axis=0, method="average").to_numpy())
        print(f"\n{SITE_T[roi]}")
        print(f"   W within 3.16 st block (n=20):  W(7) = {W:.3f}, p = {p:.4g}   "
              f"[null mean {nullb.mean():.3f}, p95 {np.quantile(nullb, .95):.3f}]")
        print(f"                                   W(6, no wh-large) = {W6:.3f}, p = {p6:.4g}")
        print(f"   W on deviance-residualised ranks (n=48): W(7) = {Wr:.3f}, p = {pr:.4g}")
        rows += [dict(site=roi, stat="kendall_W_316st_block", value=W, p=p,
                      extra=f"n=20;null_mean={nullb.mean():.3f};null_p95={np.quantile(nullb, .95):.3f}"),
                 dict(site=roi, stat="kendall_W_316st_block_no_wlarge", value=W6, p=p6, extra="n=20"),
                 dict(site=roi, stat="kendall_W_deviance_residualised", value=Wr, p=pr, extra="n=48")]

    # ── 5. direction check ───────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("5. DIRECTION CHECK — regular ↔ counter rank correlation (paired on 24 methods, within model)")
    print("=" * 92)
    print(f"{'model':17s} {'frontal ρ':>10s} {'p':>8s} {'FCz ρ':>8s} {'p':>8s}")
    dirn = {}
    for m in MODEL_ORDER:
        vals = []
        for kind, roi in SITES:
            r = depth_ranks(frames[roi]["uv"])[m]
            reg = {int(c.split("_")[1]): r[c] for c in r.index if not c.endswith("_counter")}
            ctr = {int(c.split("_")[1]): r[c] for c in r.index if c.endswith("_counter")}
            ks = sorted(reg)
            assert len(ks) == 24 and set(ks) == set(ctr), "regular/counter pairing incomplete"
            rho_, p_ = stats.spearmanr([reg[k] for k in ks], [ctr[k] for k in ks])
            vals += [rho_, p_]
            rows.append(dict(site=roi, stat=f"dir_rho_{m}", value=rho_, p=p_, extra="n=24 methods"))
        dirn[m] = vals
        print(f"{MLABEL[m]:17s} {vals[0]:+10.3f} {vals[1]:8.3f} {vals[2]:+8.3f} {vals[3]:8.3f}")
    for i, (kind, roi) in enumerate(SITES):
        v = [dirn[m][2 * i] for m in MODEL_ORDER]
        n_sig = sum(1 for m in MODEL_ORDER if dirn[m][2 * i + 1] < 0.05)
        print(f"   {roi:8s} mean ρ = {np.mean(v):+.3f}   median = {np.median(v):+.3f}   "
              f"n significant (p<0.05) = {n_sig}/7   n positive = {sum(1 for x in v if x > 0)}/7")
        rows += [dict(site=roi, stat="dir_rho_mean", value=float(np.mean(v)), p=np.nan,
                      extra=f"n_sig={n_sig};n_pos={sum(1 for x in v if x > 0)}")]

    # ── 6. per-pair agreement across the amplitude floor (11f) ──────────────────────────────
    print("\n" + "=" * 92)
    print("6. PER-CONDITION AGREEMENT ACROSS THE AMPLITUDE FLOOR — count of models (/7) per criterion")
    print("=" * 92)
    percond = {}
    for kind, roi in SITES:
        crits = criterion_frames(frames[roi]["uv"], frames[roi]["s2"])
        t = pd.DataFrame({c: B[MODEL_ORDER].sum(axis=1) for c, B in crits.items()})
        # nesting sanity: S2 >= S7@0.25 >= … >= S7@2.5 for every stimulus pair
        names = ["S2"] + [f"S7@{fx(x)}" for x in FLOORS]
        for a, b in zip(names, names[1:]):
            assert (t[a] >= t[b]).all(), f"{roi}: {b} is not nested inside {a}"
        t["method_id"] = [int(c.split("_")[1]) for c in t.index]
        t["dirn"] = ["counter" if c.endswith("_counter") else "regular" for c in t.index]
        t["st"] = t.method_id.map(ST); t["soa"] = t.method_id.map(SOA)
        t["hz"] = [f"{FDEV[m]:.0f}→{FSTD[m]:.0f}" if d == "counter" else f"{FSTD[m]:.0f}→{FDEV[m]:.0f}"
                   for m, d in zip(t.method_id, t.dirn)]
        percond[roi] = t.sort_values(["method_id", "dirn"])
        print(f"\n### {SITE_T[roi]} — per-pair model count (/7)")
        print(percond[roi][["method_id", "dirn", "st", "hz", "soa"] + names].to_string())
        print(f"   column means (/7): " + "  ".join(f"{c}={t[c].mean():.2f}" for c in names))
    pd.concat({r: percond[r] for r in percond}, names=["site"]).to_csv(f"{OUT}/sec11_per_pair_agreement.csv")
    print(f"\n  wrote {OUT}/sec11_per_pair_agreement.csv")

    # ── 7. agreement-count distribution across the floor, vs the chance null (11g) ───────────────
    print("\n" + "=" * 92)
    print("7. AGREEMENT-COUNT DISTRIBUTION ACROSS THE FLOOR — observed vs base-rate-preserving null")
    print("=" * 92)
    dist = {}
    for kind, roi in SITES:
        crits = criterion_frames(frames[roi]["uv"], frames[roi]["s2"])
        print(f"\n### {SITE_T[roi]}")
        print(f"{'criterion':10s} {'base':>5s} {'κ':>7s} {'p(κ)':>9s} " +
              " ".join(f"{'k=' + str(k):>11s}" for k in range(8)))
        for cname, B in crits.items():
            A = B[MODEL_ORDER].to_numpy().astype(bool)
            cnt = A.sum(axis=1)
            obs = np.bincount(cnt, minlength=8)
            hist, nk, nu = agreement_null(A)
            kap = fleiss_kappa(cnt)[0]
            p_kap = (1 + (nk >= kap).sum()) / (NPERM + 1)
            nullm = hist.mean(axis=0)
            print(f"{cname:10s} {A.mean():5.2f} {kap:+7.3f} {p_kap:9.4g} " +
                  " ".join(f"{obs[k]:4d} ({nullm[k]:4.1f})" for k in range(8)))
            dist[(roi, cname)] = dict(obs=obs, null=nullm, kappa=kap, p_kappa=p_kap,
                                      base=float(A.mean()), obs_unan=int(((cnt == 0) | (cnt == 7)).sum()),
                                      null_unan=float(nu.mean()),
                                      p_unan=(1 + (nu >= ((cnt == 0) | (cnt == 7)).sum()).sum()) / (NPERM + 1))
            rows += [dict(site=roi, stat=f"fleiss_kappa_floor_{cname}", value=kap, p=p_kap,
                          extra=f"base_rate={A.mean():.3f}")]
            for k in range(8):
                rows.append(dict(site=roi, stat=f"agree_k{k}_{cname}", value=int(obs[k]), p=np.nan,
                                 extra=f"null={nullm[k]:.2f}"))

    # ── 7b. cumulative agreement: pairs with AT LEAST k models (Table 50) ────────────────────────
    print("\n" + "=" * 92)
    print("7b. CUMULATIVE AGREEMENT — stimulus pairs with AT LEAST k of 7 models (Table 50)")
    print("=" * 92)
    for kind, roi in SITES:
        C = cumulative_agreement(frames[roi]["uv"], frames[roi]["s2"])
        print(f"\n### {SITE_T[roi]}")
        print(C.to_string())
        for nm in C.index:
            for col in C.columns:
                rows.append(dict(site=roi, stat=f"cum_{nm}_{col.replace('>=', 'ge')}",
                                 value=int(C.loc[nm, col]), p=np.nan, extra="of 48"))
    print("\n   nesting ({>=7} ⊆ {>=6} ⊆ … ⊆ {>=1}) and reverse-cumsum consistency with the")
    print("   Table-49 distribution: asserted for all 14 rows.")

    # ── 8. within-model z: regular vs counter (11i numbers) ──────────────────────────────────────
    print("\n" + "=" * 92)
    print("8. REGULAR vs COUNTER on within-model z (24 methods × 7 models, pooled per site)")
    print("=" * 92)
    print("   Re-pairing null: within each model, regular stimulus pairs are randomly re-matched to")
    print("   counter stimulus pairs. This preserves every marginal AND the within-model sum-to-zero")
    print("   constraint of z (which by itself induces a small negative r), so it isolates the")
    print("   effect of the TRUE method pairing.")
    rng = np.random.default_rng(SEED)
    for kind, roi in SITES:
        z = within_model_z(frames[roi]["uv"])
        X, Y = [], []
        for m in MODEL_ORDER:
            reg = {int(c.split("_")[1]): z[m][c] for c in z.index if not c.endswith("_counter")}
            ctr = {int(c.split("_")[1]): z[m][c] for c in z.index if c.endswith("_counter")}
            ks = sorted(reg)
            X.append(np.array([reg[k] for k in ks])); Y.append(np.array([ctr[k] for k in ks]))
        obs_r = stats.pearsonr(np.concatenate(X), np.concatenate(Y))[0]
        null = np.empty(NPERM)
        for t in range(NPERM):
            null[t] = stats.pearsonr(np.concatenate(X),
                                     np.concatenate([rng.permutation(y) for y in Y]))[0]
        p = (1 + (null <= obs_r).sum()) / (NPERM + 1)   # one-sided: is the TRUE pairing more negative?
        print(f"   {roi:8s} observed r = {obs_r:+.3f}   re-pairing null: mean {null.mean():+.3f}, "
              f"2.5th pct {np.quantile(null, .025):+.3f}   p(true pairing more negative) = {p:.4g}")
        rows.append(dict(site=roi, stat="regular_vs_counter_repair_null_p", value=obs_r, p=p,
                         extra=f"null_mean={null.mean():.4f};null_p2.5={np.quantile(null, .025):.3f}"))

    # ── figures ──────────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("FIGURES")
    print("=" * 92)
    fig_rank_heatmap(frames, ST, FSTD, FDEV, f"{OUT}/sec11_rank_heatmap.png")
    fig_pairwise_spearman(frames, f"{OUT}/sec11_pairwise_spearman.png")
    fig_agreement_histogram(binary, f"{OUT}/sec11_agreement_histogram.png")
    fig_floor_agreement(frames, f"{OUT}/sec11_floor_agreement.png")
    fig_z_by_method(frames, ST, FSTD, FDEV, f"{OUT}/sec11_z_by_method.png")
    rc = fig_regular_vs_counter(frames, f"{OUT}/sec11_regular_vs_counter.png")
    for roi, d in rc.items():
        print(f"   {roi:8s} Pearson r = {d['r']:+.3f} (p = {d['p_r']:.3g})   "
              f"Spearman ρ = {d['rho']:+.3f} (p = {d['p_rho']:.3g})   n = {d['n']}")
        rows += [dict(site=roi, stat="regular_vs_counter_z_pearson", value=d["r"], p=d["p_r"],
                      extra=f"n={d['n']}"),
                 dict(site=roi, stat="regular_vs_counter_z_spearman", value=d["rho"], p=d["p_rho"],
                      extra=f"n={d['n']}")]
    pd.DataFrame(rows).to_csv(f"{OUT}/sec11_stats.csv", index=False)
    print(f"  wrote {OUT}/sec11_stats.csv  ({len(rows)} statistics)")


if __name__ == "__main__":
    main()
