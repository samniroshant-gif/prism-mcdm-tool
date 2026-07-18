"""
PRISM — Sustainability MCDM Assessment Tool
Streamlit implementation

Levels:
  1. System definition (processes, categories, custom indicators, units, indicator values)
  1.5 Correlation check (within-category Spearman correlation review, informational)
  2. Indicator processing (MEREC normalisation/weights, N2 normalisation, category scores)
  3. Decision aggregation (Equal/Entropy/CRITIC weighting + RCW consolidation,
                            TOPSIS/VIKOR/ELECTRE I/MULTIMOORA/WASPAS, PSI compromise ranking,
                            category-combination rank-stability scatter)
  4. Optional validation: weighting-method sensitivity, benefit/cost indicator
     sensitivity, Monte Carlo Dirichlet uncertainty.

Run with:  streamlit run app.py
"""

import itertools
import os
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from scipy.stats import spearmanr

st.set_page_config(page_title="PRISM - Performance Ranking via Integrated Sustainability Metrics", page_icon="🧭", layout="wide")

# Cloud-safe Times New Roman lookalike (Tinos) for matplotlib + CSS @font-face
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_TINO_FILES = [
    ("Tinos-Regular.ttf", "normal", 400),
    ("Tinos-Bold.ttf", "normal", 700),
    ("Tinos-Italic.ttf", "italic", 400),
    ("Tinos-BoldItalic.ttf", "italic", 700),
]
_FONT_FACE_CSS = []
for _fname, _style, _weight in _TINO_FILES:
    _fpath = os.path.join(_FONT_DIR, _fname)
    if os.path.isfile(_fpath):
        try:
            fm.fontManager.addfont(_fpath)
        except Exception:
            pass
        try:
            with open(_fpath, "rb") as _fh:
                _b64 = base64.b64encode(_fh.read()).decode("ascii")
            _FONT_FACE_CSS.append(
                f"@font-face{{font-family:'Tinos';font-style:{_style};font-weight:{_weight};"
                f"src:url(data:font/ttf;base64,{_b64}) format('truetype');"
                f"font-display:swap;}}"
            )
        except Exception:
            pass

_FONT_CSS = '"Times New Roman", "Tinos", Times, serif'
_FONT_FACES = "\n".join(_FONT_FACE_CSS)

def tnr_label(text, *, color="#1A202C", size="12pt", weight="400", strike=False):
    """Render plain UI text in Times/Tinos (st.text uses a fixed sans font)."""
    deco = "text-decoration:line-through;" if strike else ""
    st.markdown(
        f"<div style='font-family:{_FONT_CSS};font-size:{size};font-weight:{weight};"
        f"color:{color};margin:4px 0;line-height:1.4;{deco}'>{text}</div>",
        unsafe_allow_html=True,
    )

st.markdown(f"""
<style>
{_FONT_FACES}
@import url('https://fonts.googleapis.com/css2?family=Tinos:ital,wght@0,400;0,700;1,400;1,700&display=swap');

/* ── Global Times / Tinos (no blanket span — preserves Material Icons) ── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"],
[data-testid="stText"],
[data-testid="stText"] *,
.stMarkdown, .stText, .stCaption, .stAlert, .stTooltipContent,
.stCheckbox, .stRadio, .stSelectbox, .stMultiSelect,
.stNumberInput, .stTextInput, .stTextArea, .stSlider,
.stDownloadButton > button,
[data-baseweb="select"], [data-baseweb="input"],
[data-baseweb="checkbox"], [data-baseweb="radio"],
[data-baseweb="tag"], [data-baseweb="button"],
[data-testid="stDataFrame"] *,
[data-testid="stTable"] *,
[data-testid="stDataEditor"] *,
table, th, td, .stDataFrame, .stTable, .stDataEditor,
div, p, label, input, textarea, select, button, pre {{
    font-family: {_FONT_CSS} !important;
    font-size: 12pt !important;
}}

/* Force Streamlit fixed/monospace text widgets onto Tinos */
[data-testid="stText"],
[data-testid="stText"] pre,
.stText, .stText pre,
pre, code, .stCodeBlock, .stCode {{
    font-family: {_FONT_CSS} !important;
}}

/* ── Checkbox / radio / widget labels ── */
.stCheckbox label,
.stCheckbox p,
.stRadio label,
.stRadio p,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
.stCaption {{
    font-family: {_FONT_CSS} !important;
    font-size: 12pt !important;
}}

/* ── Headings — dark blue, bold ── */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    font-family: {_FONT_CSS} !important;
    font-weight: 700 !important;
    color: #0D2B5E !important;
}}

/* ── Sidebar professional styling ── */
[data-testid="stSidebar"] {{
    background: #F7F9FC !important;
    border-right: 1px solid #DCE3EF !important;
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    font-size: 12pt !important;
    color: #4A5568 !important;
}}

/* ── Hide default Streamlit alert box colours — use neutral cards ── */
.stAlert {{
    border-radius: 4px !important;
    border-left: 3px solid #0D2B5E !important;
    background: #F7F9FC !important;
    color: #1A202C !important;
}}

/* ── Dataframe / table styling ── */
[data-testid="stDataFrame"] th,
[data-testid="stTable"] th {{
    background-color: #0D2B5E !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-family: {_FONT_CSS} !important;
    font-size: 12pt !important;
}}
[data-testid="stDataFrame"] td,
[data-testid="stTable"] td,
[data-testid="stDataFrame"] *,
[data-testid="stTable"] *,
[data-testid="stDataEditor"] * {{
    font-family: {_FONT_CSS} !important;
    color: #1A202C !important;
}}

/* ── Button styling ── */
.stButton > button {{
    font-family: {_FONT_CSS} !important;
    font-size: 12pt !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
}}
.stButton > button[kind="primary"] {{
    background-color: #0D2B5E !important;
    color: #FFFFFF !important;
    border: none !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: #1A3A6E !important;
}}

/* ── Expander header label (not icon spans) ── */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {{
    font-family: {_FONT_CSS} !important;
    font-size: 12pt !important;
    font-weight: 600 !important;
    color: #0D2B5E !important;
}}

/* ── Material icons — sidebar, expanders, header buttons ── */
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] span,
[data-testid="stExpander"] summary > span,
[data-testid="stExpander"] summary span[class*="material"],
[data-testid="stExpander"] .material-icons,
[data-testid="stExpander"] [class*="material-symbols"],
span.material-icons,
span[class*="material-symbols"],
.material-icons,
[class*="material-symbols"] {{
    font-family: "Material Symbols Rounded", "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{
    background-color: #0D2B5E !important;
}}

/* ── Divider ── */
hr {{
    border-color: #DCE3EF !important;
}}
</style>
""", unsafe_allow_html=True)



MPL_STYLE = {
    "font.family": "Tinos",
    "font.serif": ["Times New Roman", "Tinos", "Times", "Liberation Serif", "DejaVu Serif"],
    "axes.titlecolor": "#0D2B5E",
    "axes.labelcolor": "#0D2B5E",
    "xtick.color": "#0D2B5E",
    "ytick.color": "#0D2B5E",
    "axes.edgecolor": "#0D2B5E",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

def apply_mpl_style():
    plt.rcParams.update(MPL_STYLE)

apply_mpl_style()

def mpl_show(fig):
    """Display matplotlib figure in Streamlit and close it."""
    apply_mpl_style()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


CAT_SHORT = {
    "env": "E",
    "eco": "C",
    "soc": "S",
    "qua": "Q",
    "pro": "P",
    "cir": "O",
}

PROC_COLORS = ["#2563EB", "#16A34A", "#EA580C", "#9333EA", "#0891B2",
               "#CA8A04", "#DB2777", "#4F46E5", "#65A30D", "#DC2626"]

CATEGORY_ORDER = ["env", "eco", "soc", "qua", "pro", "cir"]

CATS = {
    "env": {
        "label": "Environmental", "color": "#0F6E56", "bg": "#E1F5EE",
        "indicators": ["Cumulative energy demand", "CO2 emissions", "Water consumption"],
        "default_units": ["MJ", "kg CO2-eq", "L"],
        "unit_options": [
            ["MJ", "GJ", "kWh", "MJ/kg"],
            ["kg CO2-eq", "t CO2-eq", "g CO2-eq"],
            ["L", "m3", "mL", "kg"],
        ],
        "benefit": [False, False, False],
    },
    "eco": {
        "label": "Economic", "color": "#185FA5", "bg": "#E6F1FB",
        "indicators": ["Material cost", "Machine cost", "Labour cost", "Consumables cost"],
        "default_units": ["GBP/part", "GBP/part", "GBP/part", "GBP/part"],
        "unit_options": [
            ["GBP/part", "USD/part", "EUR/part", "GBP/kg", "USD/kg"],
            ["GBP/part", "USD/part", "EUR/part", "GBP/hr", "USD/hr"],
            ["GBP/part", "USD/part", "EUR/part", "GBP/hr", "USD/hr"],
            ["GBP/part", "USD/part", "EUR/part"],
        ],
        "benefit": [False, False, False, False],
    },
    "soc": {
        "label": "Social", "color": "#534AB7", "bg": "#EEEDFE",
        "indicators": ["Recordable injury rate", "Job satisfaction", "Toxicity potential"],
        "default_units": ["per 100 workers", "GBP/year", "kg-1,4-DCB"],
        "unit_options": [
            ["per 100 workers", "per 200,000 hrs", "TRIR"],
            ["GBP/year", "USD/year", "EUR/year", "score (1-10)", "score (1-5)"],
            ["kg-1,4-DCB", "CTUh", "DALYs", "cases/yr"],
        ],
        "benefit": [False, True, False],
    },
    "qua": {
        "label": "Quality", "color": "#854F0B", "bg": "#FAEEDA",
        "indicators": ["Tensile strength", "Yield strength", "% elongation"],
        "default_units": ["MPa", "MPa", "%"],
        "unit_options": [
            ["MPa", "GPa", "ksi", "N/mm2"],
            ["MPa", "GPa", "ksi", "N/mm2"],
            ["%", "ratio"],
        ],
        "benefit": [True, True, True],
    },
    "pro": {
        "label": "Productivity", "color": "#993C1D", "bg": "#FAECE7",
        "indicators": ["Total production time", "Material utilisation rate"],
        "default_units": ["hrs", "%"],
        "unit_options": [["hrs", "min", "days", "s"], ["%", "ratio", "g/g"]],
        "benefit": [False, True],
    },
    "cir": {
        "label": "Circularity", "color": "#0F6E56", "bg": "#E1F5EE",
        "indicators": [
            "Recycled input fraction",
            "Recoverable output fraction",
            "Unrecoverable loss fraction",
            "Virgin material intensity",
            "Recycled content of product",
        ],
        "default_units": ["%", "%", "%", "kg/kg", "%"],
        "unit_options": [
            ["%", "ratio", "Custom..."],
            ["%", "ratio", "Custom..."],
            ["%", "ratio", "Custom..."],
            ["kg/kg", "g/g", "ratio", "Custom..."],
            ["%", "ratio", "Custom..."],
        ],
        "benefit": [True, True, False, False, True],
    },
}

CUSTOM_SENTINEL = "Custom..."


# ============================================================================
# CORE MATH
# ============================================================================

def merec_norm(vals, benefit):
    vals = np.asarray(vals, dtype=float)
    if benefit:
        mn = vals.min()
        return np.where(vals > 0, mn / vals, 0.0)
    else:
        mx = vals.max() or 1.0
        return vals / mx


def n2_norm(vals, benefit):
    vals = np.asarray(vals, dtype=float)
    if benefit:
        s = vals.sum() or 1.0
        return vals / s
    else:
        inv = np.where(vals > 0, 1.0 / vals, 0.0)
        s = inv.sum() or 1.0
        return inv / s


def merec_weights(norm_matrix):
    n_crit, n_alt = norm_matrix.shape
    safe = np.clip(norm_matrix, 1e-15, None)
    abs_ln = np.abs(np.log(safe))
    S = np.log(1 + abs_ln.sum(axis=0) / n_crit)
    E = np.zeros(n_crit)
    for j in range(n_crit):
        mask = np.ones(n_crit, dtype=bool)
        mask[j] = False
        s_prime = np.log(1 + abs_ln[mask, :].sum(axis=0) / n_crit)
        E[j] = np.sum(np.abs(s_prime - S))
    total = E.sum() or 1.0
    return E / total


def entropy_weights(mat):
    k, n = mat.shape
    E = np.zeros(k)
    for j in range(k):
        row = mat[j]
        s = row.sum() or 1.0
        p = row / s
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(p > 0, p * np.log(p), 0.0)
        e = -term.sum() / (np.log(n) or 1.0)
        E[j] = min(max(e, 0.0), 1.0)
    d = 1 - E
    total = d.sum() or 1.0
    return d / total


def critic_weights(mat):
    k, n = mat.shape
    rescaled = np.zeros_like(mat)
    for j in range(k):
        row = mat[j]
        mx, mn = row.max(), row.min()
        rng = (mx - mn) or 1.0
        rescaled[j] = (row - mn) / rng
    means = rescaled.mean(axis=1)
    stds = rescaled.std(axis=1, ddof=0)
    corr = np.ones((k, k))
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            da = rescaled[a] - means[a]
            db = rescaled[b] - means[b]
            num = np.sum(da * db)
            den = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2)) or 1.0
            corr[a, b] = num / den
    C = np.array([stds[j] * np.sum(1 - corr[j, :]) for j in range(k)])
    total = C.sum() or 1.0
    return C / total


def rcw_consolidate(weight_sets):
    k = len(weight_sets[0])
    harmonics = np.zeros(k)
    for i in range(k):
        s_inv = sum(1.0 / (ws[i] or 1e-9) for ws in weight_sets)
        harmonics[i] = 1.0 / s_inv
    total = harmonics.sum() or 1.0
    return harmonics / total


def rank_with_ties(values, ascending, eps=1e-6):
    n = len(values)
    order = np.argsort(values if ascending else -values, kind="stable")
    ranks = np.zeros(n, dtype=int)
    cur_rank = 1
    for pos in range(n):
        if pos > 0:
            prev_val = values[order[pos - 1]]
            cur_val = values[order[pos]]
            if abs(cur_val - prev_val) > eps:
                cur_rank = pos + 1
        ranks[order[pos]] = cur_rank
    return ranks


def topsis(weighted_mat):
    ideal = weighted_mat.max(axis=1)
    anti = weighted_mat.min(axis=1)
    Dp = np.sqrt(((weighted_mat - ideal[:, None]) ** 2).sum(axis=0))
    Dm = np.sqrt(((weighted_mat - anti[:, None]) ** 2).sum(axis=0))
    C = np.where((Dp + Dm) > 0, Dm / (Dp + Dm), 0.0)
    return rank_with_ties(C, ascending=False)


def vikor(weighted_mat, weights, v=0.5):
    f_star = weighted_mat.max(axis=1)
    f_minus = weighted_mat.min(axis=1)
    denom = np.where((f_star - f_minus) != 0, f_star - f_minus, 1.0)
    S = np.sum(weights[:, None] * (f_star[:, None] - weighted_mat) / denom[:, None], axis=0)
    R = np.max(weights[:, None] * (f_star[:, None] - weighted_mat) / denom[:, None], axis=0)
    Sm, Sx = S.min(), S.max() or 1.0
    Rm, Rx = R.min(), R.max() or 1.0
    Q = v * (S - Sm) / ((Sx - Sm) or 1.0) + (1 - v) * (R - Rm) / ((Rx - Rm) or 1.0)
    return rank_with_ties(Q, ascending=True)


def electre1(weighted_mat, weights, c_thresh=0.6, d_thresh=0.4):
    n_alt = weighted_mat.shape[1]
    total_w = weights.sum() or 1.0
    outranks = np.zeros((n_alt, n_alt), dtype=bool)
    ranges = weighted_mat.max(axis=1) - weighted_mat.min(axis=1)
    max_range = ranges.max() or 1.0
    for a in range(n_alt):
        for b in range(n_alt):
            if a == b:
                continue
            concordant = weighted_mat[:, a] >= weighted_mat[:, b]
            C = np.sum(weights[concordant]) / total_w
            D = np.max(np.maximum(weighted_mat[:, b] - weighted_mat[:, a], 0)) / max_range
            outranks[a, b] = (C >= c_thresh) and (D <= d_thresh)
    scores = outranks.sum(axis=1) - outranks.sum(axis=0)
    return rank_with_ties(scores, ascending=False)


def multimoora(weighted_mat):
    RS = weighted_mat.sum(axis=0)
    ref = weighted_mat.max(axis=1)
    RP = np.max(np.abs(ref[:, None] - weighted_mat), axis=0)
    FMF = np.prod(np.maximum(weighted_mat, 1e-9), axis=0)
    rs_rank = rank_with_ties(RS, ascending=False)
    rp_rank = rank_with_ties(RP, ascending=True)
    fmf_rank = rank_with_ties(FMF, ascending=False)
    combined = rs_rank + rp_rank + fmf_rank
    return rank_with_ties(combined.astype(float), ascending=True)


def waspas(weighted_mat, lam=0.5):
    # weighted_mat shape: (n_crit, n_alt) — already normalised x weights
    # WSM = sum of weighted normalised values (already in weighted_mat)
    WSM = weighted_mat.sum(axis=0)
    # WPM = product of (normalised_value ^ weight) per criterion
    # Since weighted_mat[j,i] = norm[j,i] * w[j], and we need norm[j,i]^w[j],
    # we use exp(w[j] * log(norm[j,i])) = exp(log(weighted_mat[j,i] / w[j] + 1e-12) * w[j])
    # Simpler: WPM = prod(weighted_mat^1) is wrong; use log space
    # Actually for WASPAS with pre-weighted matrix: WSM is correct already
    # WPM needs raw norm values — approximate via: norm = weighted_mat / weights (row-wise)
    # We pass weights separately via the lambda in MCDM_FUNCS
    Q = WSM  # fallback to WSM only if weights unavailable
    return rank_with_ties(Q, ascending=False)


def waspas_full(weighted_mat, weights, lam=0.5):
    """Correct WASPAS: recovers normalised matrix from weighted_mat and weights."""
    wm = np.asarray(weighted_mat, dtype=float)
    w = np.asarray(weights, dtype=float)
    # Recover normalised matrix: norm[j,i] = wm[j,i] / w[j]
    norm = np.where(w[:, None] > 1e-12, wm / w[:, None], 0.0)
    # WSM = sum of weighted normalised values (= wm.sum(axis=0))
    WSM = wm.sum(axis=0)
    # WPM = product of norm[j,i]^w[j]
    WPM = np.prod(np.maximum(norm, 1e-9) ** w[:, None], axis=0)
    Q = lam * WSM + (1 - lam) * WPM
    return rank_with_ties(Q, ascending=False)



MCDM_FUNCS = {
    "topsis": lambda wm, w: topsis(wm),
    "vikor": lambda wm, w: vikor(wm, w),
    "electre": lambda wm, w: electre1(wm, w),
    "multimoora": lambda wm, w: multimoora(wm),
    "waspas": lambda wm, w: waspas_full(wm, w),
}
METHOD_LABELS = {
    "topsis": "TOPSIS", "vikor": "VIKOR", "electre": "ELECTRE I",
    "multimoora": "MULTIMOORA", "waspas": "WASPAS",
}
ALL_MCDM_KEYS = ["topsis", "vikor", "electre", "multimoora", "waspas"]
WEIGHT_COMBO_SETS = [
    (("equal",), "Equal"),
    (("entropy",), "Entropy"),
    (("critic",), "CRITIC"),
    (("equal", "entropy"), "RCW(Eq+En)"),
    (("equal", "critic"), "RCW(Eq+Cr)"),
    (("entropy", "critic"), "RCW(En+Cr)"),
]


def calc_psi(method_ranks, methods, p):
    n_alt = len(next(iter(method_ranks.values())))
    psi = np.zeros(n_alt)
    for i in range(n_alt):
        ranks = np.array([method_ranks[m][i] for m in methods], dtype=float)
        r_bar = ranks.mean()
        cv = ranks.std(ddof=0) / r_bar if (len(ranks) > 1 and r_bar > 0) else 0.0
        M = 1.0 / (r_bar or 1.0)
        A = 1.0 / (1 + cv)
        psi[i] = (M ** p) * (A ** (1 - p))
    return psi


def get_category_weights(mat, weight_methods):
    k = mat.shape[0]
    sets = []
    if "equal" in weight_methods:
        sets.append(np.full(k, 1.0 / k))
    if "entropy" in weight_methods:
        sets.append(entropy_weights(mat))
    if "critic" in weight_methods:
        sets.append(critic_weights(mat))
    if not sets:
        return np.full(k, 1.0 / k)
    return rcw_consolidate(sets) if len(sets) > 1 else sets[0]


def run_mcdm_suite(weighted_mat, weights, methods):
    ranks = {}
    for m in methods:
        ranks[m] = MCDM_FUNCS[m](weighted_mat, weights)
    return ranks




def get_combinations(cats_list):
    order_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    cats_sorted = sorted(cats_list, key=lambda c: order_index[c])
    combos = []
    for r in range(1, len(cats_sorted) + 1):
        combos.extend(itertools.combinations(cats_sorted, r))
    combos.sort(key=lambda c: (len(c), [order_index[x] for x in c]))
    return [list(c) for c in combos]


def compute_dirichlet_k(mat):
    k_cat = mat.shape[0]
    w_eq = np.full(k_cat, 1.0 / k_cat)
    w_en = entropy_weights(mat)
    w_cr = critic_weights(mat)
    W = np.array([w_eq, w_en, w_cr])
    n_methods = 3
    var_per_cat = W.var(axis=0, ddof=0)
    mean_var = var_per_cat.mean()
    max_var = ((n_methods - 1) * (1 / n_methods) ** 2 + (1 - 1 / n_methods) ** 2) / n_methods
    dispersion_ratio = min(mean_var / max_var, 1.0) if max_var > 0 else 0.0
    agreement = 1 - dispersion_ratio
    return agreement * 100, w_eq, w_en, w_cr


# ============================================================================
# SESSION STATE
# ============================================================================

def init_state():
    defaults = {
        "step": 0,
        "n_proc": 3,
        "proc_names": [],
        "sel_cats": set(),
        "sel_units": {},
        "indicator_values": {},
        "use_custom_indicators": None,
        "custom_indicator_counts": {},
        "custom_indicators": {},
        "l3_cats": set(),
        "sel_weight_methods": set(),
        "sel_mcdm_methods": set(),
        "computed": False,
        "corr_acknowledged": False,
        "validation_choice": "None - skip validation",
        "disabled_indicators": set(),  # set of (ckey, j) tuples
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


init_state()


def ordered_sel_cats():
    return [c for c in CATEGORY_ORDER if c in st.session_state.sel_cats]


def ordered_l3_cats():
    return [c for c in CATEGORY_ORDER if c in st.session_state.l3_cats]


def get_full_indicators(ckey):
    cat = CATS[ckey]
    disabled = st.session_state.get("disabled_indicators", set())
    names, units, benefits = [], [], []
    for j, ind in enumerate(cat["indicators"]):
        if (ckey, j) not in disabled:
            names.append(ind)
            units.append(st.session_state.sel_units.get(f"{ckey}_{j}", cat["default_units"][j]))
            benefits.append(cat["benefit"][j])
    n_custom = st.session_state.custom_indicator_counts.get(ckey, 0)
    for ci in range(n_custom):
        info = st.session_state.custom_indicators.get((ckey, ci), {})
        names.append(info.get("name") or f"Custom indicator {ci+1}")
        units.append(info.get("unit") or "unit")
        benefits.append(info.get("benefit", True))
    return names, units, benefits


def get_raw_matrix(ckey):
    names, units, benefits = get_full_indicators(ckey)
    n_proc = st.session_state.n_proc
    raw = np.zeros((len(names), n_proc))
    for j in range(len(names)):
        for pi in range(n_proc):
            raw[j, pi] = st.session_state.indicator_values.get((ckey, j, pi), 0.0)
    return raw, benefits


def compute_category_score_from_raw(ckey, raw_override=None):
    names, units, benefits = get_full_indicators(ckey)
    n_ind = len(names)
    if raw_override is not None:
        raw = raw_override
    else:
        raw, _ = get_raw_matrix(ckey)
    nm = np.zeros_like(raw)
    n2 = np.zeros_like(raw)
    for j in range(n_ind):
        nm[j] = merec_norm(raw[j], benefits[j])
        n2[j] = n2_norm(raw[j], benefits[j])
    w = merec_weights(nm)
    score = (n2 * w[:, None]).sum(axis=0)
    return score


STEP_LABELS = [
    "1. Processes", "2. Categories", "3. Custom indicators", "4. Units",
    "5. Indicators", "6. Correlation check", "7. MEREC weights",
    "8. Category scores", "9. Level 3 categories", "10. Category weights",
    "11. MCDM methods", "12. Results", "13. Validation (optional)",
    "14. Analytics (optional)",
]

with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0;font-size:22px;color:#0D2B5E;"
        "font-family:Times New Roman,Tinos,Times,serif;font-weight:700;'>PRISM</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:12pt;color:#6B7A99;margin-top:2px;"
        "font-family:Times New Roman,Tinos,Times,serif;'>"
        "Performance Ranking via Integrated Sustainability Metrics</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    SECTIONS = {
        "Problem Definition": [1, 2, 3, 4, 5],
        "Indicator Processing": [6, 7, 8],
        "Decision Aggregation": [9, 10, 11, 12],
        "Validation": [13],
        "Analytics": [14],
    }

    step = st.session_state.step
    for section, steps in SECTIONS.items():
        st.markdown(
            f"<p style='font-size:12pt;font-weight:700;color:#0D2B5E;"
            f"text-transform:uppercase;letter-spacing:0.05em;"
            f"font-family:Times New Roman,Tinos,Times,serif;margin:8px 0 2px 0;'>"
            f"{section}</p>",
            unsafe_allow_html=True,
        )
        for s in steps:
            if s >= len(STEP_LABELS) + 1:
                continue
            label = STEP_LABELS[s - 1]
            short = label.split(". ", 1)[1] if ". " in label else label
            if s < step:
                st.markdown(
                    f"<p style='font-size:12pt;color:#16A34A;margin:1px 0;"
                    f"font-family:Times New Roman,Tinos,Times,serif;'>✓ {short}</p>",
                    unsafe_allow_html=True,
                )
            elif s == step:
                st.markdown(
                    f"<p style='font-size:12pt;font-weight:700;color:#0D2B5E;"
                    f"background:#E8EEF7;padding:3px 8px;border-radius:4px;"
                    f"border-left:3px solid #0D2B5E;margin:1px 0;"
                    f"font-family:Times New Roman,Tinos,Times,serif;'>▶ {short}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p style='font-size:12pt;color:#9CA3AF;margin:1px 0;"
                    f"font-family:Times New Roman,Tinos,Times,serif;'>○ {short}</p>",
                    unsafe_allow_html=True,
                )

    st.divider()
    if st.button("Reset", use_container_width=True):
        reset_all()
        st.rerun()


# ============================================================================
# STEP 1 - PROCESSES (FIX 1: no example names pre-filled)
# ============================================================================

def landing_page():
    st.markdown(
        "<h1 style='font-size:32px;font-weight:700;color:#0D2B5E;"
        "font-family:Times New Roman,Tinos,Times,serif;margin-bottom:4px;'>"
        "PRISM</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:16px;color:#4A5568;font-style:italic;"
        "font-family:Times New Roman,Tinos,Times,serif;margin-bottom:24px;'>"
        "Performance Ranking via Integrated Sustainability Metrics</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            "<h3 style='color:#0D2B5E;font-family:Times New Roman,Tinos,Times,serif;"
            "font-weight:700;'>About PRISM</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:14px;line-height:1.8;color:#1A202C;"
            "font-family:Times New Roman,Tinos,Times,serif;text-align:justify;'>"
            "PRISM is an integrated multi-criteria decision-making framework "
            "developed for the comparative sustainability assessment of manufacturing "
            "processes. It combines indicator-level weighting through the Method Based "
            "on the Removal Effects of Criteria (MEREC), cross-category weight "
            "consolidation via the Reciprocal Composite Weighting (RCW) method, and "
            "a multi-method MCDM aggregation suite, producing a final compromise "
            "ranking through the Performance Stability Index (PSI).</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:14px;line-height:1.8;color:#1A202C;"
            "font-family:Times New Roman,Tinos,Times,serif;text-align:justify;'>"
            "The framework evaluates alternatives across six sustainability dimensions: "
            "Environmental, Economic, Social, Quality, Productivity, and Circularity. "
            "An integrated validation suite and analytics layer provide robustness "
            "evidence and stakeholder-oriented sensitivity analysis.</p>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            "<h3 style='color:#0D2B5E;font-family:Times New Roman,Tinos,Times,serif;"
            "font-weight:700;'>Framework</h3>",
            unsafe_allow_html=True,
        )
        framework_items = [
            ("System Definition", "Processes, categories, indicators"),
            ("Indicator Processing", "MEREC weighting, N2 normalisation"),
            ("Decision Aggregation", "RCW, MCDM suite, PSI ranking"),
            ("Validation", "5 robustness checks"),
            ("Analytics", "Contribution, sensitivity, stakeholder"),
        ]
        for title, desc in framework_items:
            st.markdown(
                f"<div style='background:#F7F9FC;border-left:3px solid #0D2B5E;"
                f"padding:8px 12px;border-radius:4px;margin:6px 0;'>"
                f"<span style='font-weight:700;color:#0D2B5E;font-size:13px;"
                f"font-family:Times New Roman,Tinos,Times,serif;'>{title}</span>"
                f"<br><span style='font-size:12px;color:#6B7A99;"
                f"font-family:Times New Roman,Tinos,Times,serif;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Begin Assessment →", type="primary",
                     use_container_width=True, key="start_btn"):
            st.session_state.step = 1
            st.rerun()



def step1():
    st.header("Step 1 - Define processes")

    n = st.slider("Number of processes", min_value=2, max_value=10,
                   value=st.session_state.n_proc, key="n_proc_slider")
    st.session_state.n_proc = n

    names = st.session_state.proc_names
    if len(names) < n:
        names = names + [""] * (n - len(names))
    elif len(names) > n:
        names = names[:n]
    st.session_state.proc_names = names

    st.subheader("Name each process")
    cols = st.columns(min(n, 5))
    new_names = []
    for i in range(n):
        with cols[i % len(cols)]:
            val = st.text_input(f"Process {i+1}", value=names[i], key=f"pname_{i}",
                                 placeholder="enter name")
            new_names.append(val.strip())
    st.session_state.proc_names = new_names

    st.divider()
    if st.button("Next ->", type="primary"):
        final_names = [n.strip() if n.strip() else f"P{i+1}" for i, n in enumerate(new_names)]
        st.session_state.proc_names = final_names
        st.session_state.step = 2
        st.rerun()


def step2():
    st.header("Step 2 - Select assessment categories")

    cols = st.columns(len(CATEGORY_ORDER))
    for i, key in enumerate(CATEGORY_ORDER):
        cat = CATS[key]
        with cols[i]:
            checked = key in st.session_state.sel_cats
            new_val = st.checkbox(cat["label"], value=checked, key=f"catchk_{key}")
            if new_val:
                st.session_state.sel_cats.add(key)
            else:
                st.session_state.sel_cats.discard(key)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("Next ->", type="primary"):
            if not st.session_state.sel_cats:
                st.error("Select at least one category.")
            else:
                st.session_state.step = 3
                st.rerun()


def step3():
    st.header("Step 3 - Add custom indicators (optional)")

    use_custom = st.radio(
        "Do you want to add any indicators beyond the predefined list?",
        ["No, use the predefined indicators only", "Yes, I want to add custom indicators"],
        index=0 if st.session_state.use_custom_indicators in (None, False) else 1,
        key="use_custom_radio",
    )
    st.session_state.use_custom_indicators = use_custom.startswith("Yes")

    if st.session_state.use_custom_indicators:
        st.divider()
        st.subheader("How many custom indicators per category?")
        cats = ordered_sel_cats()
        cols = st.columns(len(cats))
        for i, ckey in enumerate(cats):
            cat = CATS[ckey]
            with cols[i]:
                cnt = st.number_input(
                    cat["label"], min_value=0, max_value=5,
                    value=st.session_state.custom_indicator_counts.get(ckey, 0),
                    key=f"customcnt_{ckey}",
                )
                st.session_state.custom_indicator_counts[ckey] = int(cnt)

        any_custom = any(v > 0 for v in st.session_state.custom_indicator_counts.values())
        if any_custom:
            st.divider()
            st.subheader("Define each custom indicator")
            for ckey in cats:
                n_custom = st.session_state.custom_indicator_counts.get(ckey, 0)
                if n_custom == 0:
                    continue
                cat = CATS[ckey]
                st.markdown(f"**{cat['label']}**")
                for ci in range(n_custom):
                    info = st.session_state.custom_indicators.get((ckey, ci), {})
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        name = st.text_input(
                            f"Indicator name #{ci+1}", value=info.get("name", ""),
                            key=f"custname_{ckey}_{ci}", placeholder="e.g. Noise level",
                        )
                    with c2:
                        unit = st.text_input(
                            "Unit", value=info.get("unit", ""),
                            key=f"custunit_{ckey}_{ci}", placeholder="e.g. dB",
                        )
                    with c3:
                        benefit = st.selectbox(
                            "Direction", ["Cost (lower better)", "Benefit (higher better)"],
                            index=1 if info.get("benefit", False) else 0,
                            key=f"custben_{ckey}_{ci}",
                        )
                    st.session_state.custom_indicators[(ckey, ci)] = {
                        "name": name.strip(), "unit": unit.strip() or "unit",
                        "benefit": benefit.startswith("Benefit"),
                    }
                st.write("")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("Next ->", type="primary"):
            st.session_state.step = 4
            st.rerun()


def step4():
    st.header("Step 4 - Select units and enable indicators")

    disabled = st.session_state.get("disabled_indicators", set())

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )
        st.write("")

        # Column headers
        hc1, hc2, hc3, hc4 = st.columns([0.4, 2, 1.5, 1.5])
        with hc1: st.markdown("**Use**")
        with hc2: st.markdown("**Indicator**")
        with hc3: st.markdown("**Unit**")
        with hc4: st.markdown("**Custom unit**")

        for j, ind in enumerate(cat["indicators"]):
            key = f"{ckey}_{j}"
            is_enabled = (ckey, j) not in disabled
            current = st.session_state.sel_units.get(key, cat["default_units"][j])
            options = list(cat["unit_options"][j])
            is_preset = current in options
            display_options = options + [CUSTOM_SENTINEL]
            default_index = options.index(current) if is_preset else len(options)

            c1, c2, c3, c4 = st.columns([0.4, 2, 1.5, 1.5])
            with c1:
                enabled = st.checkbox(
                    "use", value=is_enabled,
                    key=f"ind_enabled_{ckey}_{j}",
                    label_visibility="collapsed",
                )
                if enabled:
                    disabled.discard((ckey, j))
                else:
                    disabled.add((ckey, j))
            with c2:
                if not enabled:
                    tnr_label(ind, color="#aaaaaa", strike=True)
                else:
                    tnr_label(ind)
            with c3:
                chosen = st.selectbox(
                    "unit", display_options, index=default_index,
                    key=f"unitsel_{key}", label_visibility="collapsed",
                    disabled=not enabled,
                )
            with c4:
                if chosen == CUSTOM_SENTINEL and enabled:
                    custom_val = st.text_input(
                        "custom unit", value=current if not is_preset else "",
                        key=f"unitcustom_{key}", label_visibility="collapsed",
                        placeholder="type unit",
                    )
                    final_unit = custom_val.strip() or "unit"
                else:
                    final_unit = chosen if chosen != CUSTOM_SENTINEL else current
                    st.caption(" ")
            st.session_state.sel_units[key] = final_unit

        st.session_state.disabled_indicators = disabled
        st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("Next ->", type="primary"):
            # Validate at least 1 indicator per selected category
            errors = []
            disabled = st.session_state.get("disabled_indicators", set())
            for ckey in ordered_sel_cats():
                cat = CATS[ckey]
                enabled_count = sum(
                    1 for j in range(len(cat["indicators"]))
                    if (ckey, j) not in disabled
                ) + st.session_state.custom_indicator_counts.get(ckey, 0)
                if enabled_count < 1:
                    errors.append(f"{cat['label']}: at least one indicator must be enabled.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                # Clear any stale seed keys for step5 so editor rebuilds
                for k in [k for k in st.session_state if k.startswith("editor_seed_")]:
                    del st.session_state[k]
                st.session_state.step = 5
                st.rerun()


# ============================================================================
# STEP 5 - INDICATOR VALUES
# FIX 2: values no longer reset to zero. The data_editor's own widget state
# (keyed per category, in st.session_state[editor_key]) is the single
# source of truth across reruns. We only build a seed DataFrame the FIRST
# time the key appears in session_state; on every subsequent rerun
# Streamlit reuses the live widget state instead of overwriting it.
# ============================================================================

def step5():
    st.header("Step 5 - Enter indicator values")

    names = st.session_state.proc_names

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        rows = [f"{ind_names[j]} ({ind_units[j]})" for j in range(len(ind_names))]
        editor_key = f"editor_{ckey}"
        seed_key = f"editor_seed_{ckey}"

        if seed_key not in st.session_state:
            seed = [[st.session_state.indicator_values.get((ckey, j, pi), 0.0)
                      for pi in range(len(names))] for j in range(len(ind_names))]
            st.session_state[seed_key] = pd.DataFrame(seed, index=rows, columns=names)

        edited = st.data_editor(st.session_state[seed_key], key=editor_key, use_container_width=True)

        for j in range(len(ind_names)):
            for pi in range(len(names)):
                st.session_state.indicator_values[(ckey, j, pi)] = float(edited.iloc[j, pi])

        st.write("")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 4
            st.rerun()
    with c2:
        if st.button("Next ->", type="primary"):
            has_values = any(v != 0 for v in st.session_state.indicator_values.values())
            if not has_values:
                st.error("Enter at least some values before proceeding.")
            else:
                st.session_state.corr_acknowledged = False
                st.session_state.step = 6
                st.rerun()


def step6():
    st.header("Step 6 - Within-category correlation check")

    names = st.session_state.proc_names
    n_proc = len(names)
    flagged_any = False

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        ind_names, _, _ = get_full_indicators(ckey)
        n_ind = len(ind_names)

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        if n_ind < 2:
            st.write("")
            continue
        if n_proc < 3:
            st.write("")
            continue

        raw = np.zeros((n_ind, n_proc))
        for j in range(n_ind):
            for pi in range(n_proc):
                raw[j, pi] = st.session_state.indicator_values.get((ckey, j, pi), 0.0)

        corr_mat = np.eye(n_ind)
        for a in range(n_ind):
            for b in range(n_ind):
                if a == b:
                    continue
                rho, _ = spearmanr(raw[a], raw[b])
                corr_mat[a, b] = rho if not np.isnan(rho) else 0.0

        df_corr = pd.DataFrame(corr_mat, index=ind_names, columns=ind_names).round(3)

        st.dataframe(df_corr, use_container_width=True)

        pairs_flagged = []
        for a in range(n_ind):
            for b in range(a + 1, n_ind):
                if abs(corr_mat[a, b]) > 0.8:
                    pairs_flagged.append((ind_names[a], ind_names[b], corr_mat[a, b]))

        if pairs_flagged:
            flagged_any = True
            for a, b, r in pairs_flagged:
                st.warning(f"**{a}** and **{b}** are highly correlated (rho = {r:.3f}).")

        st.write("")

    if flagged_any:
        st.info(
            "Highly correlated indicator pairs were found. This is shown for your "
            "awareness - both indicators remain in the analysis."
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 5
            st.rerun()
    with c2:
        if st.button("Accept and proceed ->", type="primary"):
            st.session_state.corr_acknowledged = True
            compute_level2()
            st.session_state.step = 7
            st.rerun()


def compute_level2():
    n_proc = st.session_state.n_proc
    nm_data, n2_data, merec_w, cat_scores = {}, {}, {}, {}

    for ckey in ordered_sel_cats():
        ind_names, ind_units, benefits = get_full_indicators(ckey)
        n_ind = len(ind_names)

        raw = np.zeros((n_ind, n_proc))
        for j in range(n_ind):
            for pi in range(n_proc):
                raw[j, pi] = st.session_state.indicator_values.get((ckey, j, pi), 0.0)

        nm = np.zeros((n_ind, n_proc))
        n2 = np.zeros((n_ind, n_proc))
        for j in range(n_ind):
            nm[j] = merec_norm(raw[j], benefits[j])
            n2[j] = n2_norm(raw[j], benefits[j])

        w = merec_weights(nm)
        scores = (n2 * w[:, None]).sum(axis=0)

        nm_data[ckey] = nm
        n2_data[ckey] = n2
        merec_w[ckey] = w
        cat_scores[ckey] = scores

    st.session_state.nm_data = nm_data
    st.session_state.n2_data = n2_data
    st.session_state.merec_w = merec_w
    st.session_state.cat_scores = cat_scores
    st.session_state.computed = True


# ============================================================================
# STEP 7 - MEREC WEIGHTS
# FIX 4: only the single, final MEREC weight column is shown; per-process
# intermediate normalisation columns are no longer displayed.
# ============================================================================

def step7():
    st.header("Step 7 - MEREC weights")

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)
        w = st.session_state.merec_w[ckey]

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        rows = [[ind, round(w[j], 4)] for j, ind in enumerate(ind_names)]
        st.dataframe(pd.DataFrame(rows, columns=["Indicator", "MEREC weight"]),
                     use_container_width=True, hide_index=True)
        st.write("")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 6
            st.rerun()
    with c2:
        if st.button("View category scores ->", type="primary"):
            st.session_state.step = 8
            st.rerun()


def step8():
    st.header("Step 8 - Category scores")

    names = st.session_state.proc_names
    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        scores = st.session_state.cat_scores[ckey]

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        apply_mpl_style()
        fig, ax = plt.subplots(figsize=(6, max(1.2, 0.45 * len(names))))
        bars = ax.barh(names, scores, color=cat["color"], edgecolor="white")
        for bar, val in zip(bars, scores):
            ax.text(bar.get_width() + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9, color="#0D2B5E")
        ax.set_xlim(0, max(scores) * 1.18)
        ax.tick_params(axis="both", labelsize=9)
        plt.tight_layout()
        mpl_show(fig)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 7
            st.rerun()
    with c2:
        if st.button("Select for Level 3 ->", type="primary"):
            st.session_state.step = 9
            st.rerun()


def step9():
    st.header("Step 9 - Select categories for Level 3")

    cats = ordered_sel_cats()
    cols = st.columns(len(cats))
    for i, ckey in enumerate(cats):
        cat = CATS[ckey]
        with cols[i]:
            checked = ckey in st.session_state.l3_cats
            new_val = st.checkbox(cat["label"], value=checked, key=f"l3chk_{ckey}")
            if new_val:
                st.session_state.l3_cats.add(ckey)
            else:
                st.session_state.l3_cats.discard(ckey)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 8
            st.rerun()
    with c2:
        if st.button("Level 3 ->", type="primary"):
            if not st.session_state.l3_cats:
                st.error("Select at least one category.")
            else:
                st.session_state.step = 10
                st.rerun()


def step10():
    st.header("Step 10 - Category weighting methods")

    options = {"equal": "Equal weights", "entropy": "Entropy weights", "critic": "CRITIC weights"}
    for key, label in options.items():
        checked = key in st.session_state.sel_weight_methods
        new_val = st.checkbox(label, value=checked, key=f"wmchk_{key}")
        if new_val:
            st.session_state.sel_weight_methods.add(key)
        else:
            st.session_state.sel_weight_methods.discard(key)

    if st.session_state.sel_weight_methods:
        l3_cats = ordered_l3_cats()
        mat = np.array([st.session_state.cat_scores[c] for c in l3_cats])
        k = len(l3_cats)

        sets, labels = [], []
        if "equal" in st.session_state.sel_weight_methods:
            sets.append(np.full(k, 1.0 / k)); labels.append("Equal")
        if "entropy" in st.session_state.sel_weight_methods:
            sets.append(entropy_weights(mat)); labels.append("Entropy")
        if "critic" in st.session_state.sel_weight_methods:
            sets.append(critic_weights(mat)); labels.append("CRITIC")

        final_w = rcw_consolidate(sets) if len(sets) > 1 else sets[0]
        st.session_state.final_cat_weights = final_w

        rows = []
        for i, ckey in enumerate(l3_cats):
            row = [CATS[ckey]["label"]]
            for s in sets:
                row.append(round(s[i], 4))
            row.append(round(final_w[i], 4))
            rows.append(row)
        cols = ["Category"] + labels + (["RCW"] if len(sets) > 1 else ["Final"])
        st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 9
            st.rerun()
    with c2:
        if st.button("Select MCDM ->", type="primary"):
            if not st.session_state.sel_weight_methods:
                st.error("Select at least one weighting method.")
            else:
                st.session_state.step = 11
                st.rerun()


def step11():
    st.header("Step 11 - Select MCDM methods")


    MCDM_FAMILIES = {
        "Distance-based": {
            "description": "Evaluate alternatives by distance from ideal or reference solutions.",
            "methods": {
                "topsis": ("TOPSIS", "Technique for Order of Preference by Similarity to Ideal Solution"),
            },
        },
        "Compromise-based": {
            "description": "Seek a compromise balancing group utility and individual regret.",
            "methods": {
                "vikor": ("VIKOR", "VIšekriterijumsko KOmpromisno Rangiranje"),
            },
        },
        "Outranking": {
            "description": "Compare alternatives pairwise using concordance and discordance.",
            "methods": {
                "electre": ("ELECTRE I", "ELimination Et Choix Traduisant la REalité"),
            },
        },
        "Aggregation": {
            "description": "Combine weighted scores through mathematical aggregation functions.",
            "methods": {
                "waspas": ("WASPAS", "Weighted Aggregated Sum Product ASsessment"),
            },
        },
        "Multi-subordinate": {
            "description": "Aggregate multiple subordinate ranking models into one.",
            "methods": {
                "multimoora": ("MULTIMOORA", "Multi-Objective Optimisation by Ratio Analysis"),
            },
        },
    }

    families_selected = set()
    for family_name, family_data in MCDM_FAMILIES.items():
        with st.expander(family_name, expanded=True):
            st.caption(family_data["description"])
            for key, (short, full) in family_data["methods"].items():
                checked = key in st.session_state.sel_mcdm_methods
                new_val = st.checkbox(
                    f"{short} — {full}",
                    value=checked,
                    key=f"mmchk_{key}",
                )
                if new_val:
                    st.session_state.sel_mcdm_methods.add(key)
                    families_selected.add(family_name)
                else:
                    st.session_state.sel_mcdm_methods.discard(key)

    # Recount families after loop
    families_selected = set()
    for family_name, family_data in MCDM_FAMILIES.items():
        for key in family_data["methods"]:
            if key in st.session_state.sel_mcdm_methods:
                families_selected.add(family_name)

    n_selected = len(st.session_state.sel_mcdm_methods)
    n_families = len(families_selected)



    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 10
            st.rerun()
    with c2:
        if st.button("Calculate results ->", type="primary"):
            if n_selected == 0:
                st.error("Select at least one MCDM method.")
            else:
                st.session_state.step = 12
                st.rerun()



def step12():
    st.header("Step 12 - Results")

    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    final_w = st.session_state.final_cat_weights
    cat_scores = st.session_state.cat_scores
    methods = list(st.session_state.sel_mcdm_methods)
    multi = len(methods) > 1

    weighted_mat = np.array([cat_scores[c] * final_w[i] for i, c in enumerate(l3_cats)])

    method_ranks = run_mcdm_suite(weighted_mat, final_w, methods)
    st.session_state.last_method_ranks = method_ranks

    st.subheader("MCDM rankings")

    cols = ["Process"] + [METHOD_LABELS[m] for m in methods]
    if multi:
        cols.append("PSI Rank (p=0.50)")
        psi_05 = calc_psi(method_ranks, methods, 0.5)
        psi_rank_05 = rank_with_ties(psi_05, ascending=False)

    rows = []
    for pi, name in enumerate(names):
        row = [name] + [int(method_ranks[m][pi]) for m in methods]
        if multi:
            row.append(int(psi_rank_05[pi]))
        rows.append(row)

    st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True, hide_index=True)

    if multi:
        st.divider()
        st.subheader("PSI curve")
        st.caption("PSI_i = M_i^p x A_i^(1-p)  ;  M_i = 1/Rbar  ;  A_i = 1/(1+CV)  ;  p in (0,1)")

        p_val = st.slider("p (stability <-> performance)", min_value=0.01, max_value=0.99,
                           value=0.5, step=0.01, key="psi_p_slider")

        psi_vals = calc_psi(method_ranks, methods, p_val)
        psi_ranks = rank_with_ties(psi_vals, ascending=False)

        bar_cols = st.columns(len(names))
        for i, name in enumerate(names):
            with bar_cols[i]:
                st.metric(name, f"{psi_vals[i]:.4f}", f"rank {psi_ranks[i]}")

        p_range = np.linspace(0.01, 0.99, 99)
        apply_mpl_style()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for i, name in enumerate(names):
            series = [calc_psi(method_ranks, methods, p)[i] for p in p_range]
            ax.plot(p_range, series, label=name, linewidth=2,
                    color=PROC_COLORS[i % len(PROC_COLORS)])
        ax.axvline(x=p_val, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("p", fontsize=10)
        ax.set_ylabel("PSI", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  ncol=len(names), fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        mpl_show(fig)

        st.divider()
        st.subheader("Ranking across category combinations")
        n_combo = 2 ** len(l3_cats) - 1

        combo_p = st.slider("p for combination view", min_value=0.0, max_value=1.0,
                             value=0.5, step=0.01, key="combo_p_slider")

        combos = get_combinations(l3_cats)
        cat_initial = {c: CAT_SHORT.get(c, CATS[c]["label"][:1]) for c in l3_cats}
        combo_labels = ["+".join(cat_initial[c] for c in combo) for combo in combos]

        rank_grid = np.zeros((n_proc, len(combos)), dtype=int)
        for ci, combo in enumerate(combos):
            if len(combo) == 1:
                rank_grid[:, ci] = rank_with_ties(cat_scores[combo[0]], ascending=False)
            else:
                sub_mat = np.array([cat_scores[c] for c in combo])
                sub_w = get_category_weights(sub_mat, st.session_state.sel_weight_methods)
                sub_weighted = sub_mat * sub_w[:, None]
                sub_ranks = run_mcdm_suite(sub_weighted, sub_w, methods)
                psi_combo = calc_psi(sub_ranks, methods, combo_p)
                rank_grid[:, ci] = rank_with_ties(psi_combo, ascending=False)

        apply_mpl_style()
        n_combos_b = len(combos)
        fig, ax = plt.subplots(figsize=(max(8, n_combos_b * 0.22), 4.2))
        for pi, name in enumerate(names):
            ax.scatter(range(n_combos_b), rank_grid[pi, :],
                       label=name, s=40, marker="s",
                       color=PROC_COLORS[pi % len(PROC_COLORS)], zorder=3)
        ax.set_xticks(range(n_combos_b))
        ax.set_xticklabels(combo_labels, rotation=90, fontsize=7)
        ax.set_ylabel("Rank", fontsize=10)
        ax.set_xlabel("Category combination", fontsize=10)
        ax.set_ylim(n_proc + 0.5, 0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1),
                  ncol=len(names), fontsize=9, frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        mpl_show(fig)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 11
            st.rerun()
    with c2:
        if st.button("Validation (optional) ->", type="primary"):
            st.session_state.step = 13
            st.rerun()
    with c3:
        if st.button("Analytics ->", type="primary"):
            st.session_state.step = 14
            st.rerun()
    with c4:
        if st.button("Reset all"):
            reset_all()
            st.rerun()


# ============================================================================
# STEP 13 - VALIDATION (optional)
# ============================================================================

def validation_rank_reversal():
    st.subheader("4. Rank-reversal test")
    
    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})

    if n_proc < 3:
        st.warning("Rank-reversal test requires at least 3 alternatives (you currently have 2).")
        return

    p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                       value=0.5, step=0.01, key="rr_p_slider")

    excluded = st.multiselect(
        "Select alternative(s) to temporarily exclude",
        options=names,
        default=[],
        key="rr_exclude",
    )

    if not excluded:
        return

    keep_idx = [i for i, n in enumerate(names) if n not in excluded]
    keep_names = [names[i] for i in keep_idx]

    if len(keep_idx) < 2:
        st.warning("At least 2 alternatives must remain after exclusion.")
        return

    # --- Baseline ranks for the kept alternatives ---
    multi = len(sel_mcdm_methods) > 1
    base_method_ranks_kept = {}
    for m in sel_mcdm_methods:
        if m in method_ranks:
            # Re-rank among kept subset only (to get correct relative positions)
            scores_kept = np.array([method_ranks[m][i] for i in keep_idx], dtype=float)
            base_method_ranks_kept[m] = rank_with_ties(scores_kept, ascending=True)

    if multi and base_method_ranks_kept:
        base_psi_kept = calc_psi(base_method_ranks_kept, sel_mcdm_methods, p_val)
        base_psi_rank_kept = rank_with_ties(base_psi_kept, ascending=False)

    # --- Perturbed pipeline: re-run full PRISM on kept alternatives only ---
    # Rebuild category scores from raw indicator values, restricted to kept_idx
    cat_scores_pert = {}
    for ckey in l3_cats:
        ind_names, ind_units, benefits = get_full_indicators(ckey)
        n_ind = len(ind_names)
        raw_full = np.zeros((n_ind, n_proc))
        for j in range(n_ind):
            for pi in range(n_proc):
                raw_full[j, pi] = st.session_state.indicator_values.get((ckey, j, pi), 0.0)
        raw_kept = raw_full[:, keep_idx]
        # MEREC norm + weights on reduced set
        nm = np.zeros_like(raw_kept)
        n2 = np.zeros_like(raw_kept)
        for j in range(n_ind):
            nm[j] = merec_norm(raw_kept[j], benefits[j])
            n2[j] = n2_norm(raw_kept[j], benefits[j])
        w_ind = merec_weights(nm)
        cat_scores_pert[ckey] = (n2 * w_ind[:, None]).sum(axis=0)

    mat_pert = np.array([cat_scores_pert[c] for c in l3_cats])
    w_cat_pert = get_category_weights(mat_pert, sel_weight_methods)
    weighted_mat_pert = mat_pert * w_cat_pert[:, None]
    ranks_pert = run_mcdm_suite(weighted_mat_pert, w_cat_pert, sel_mcdm_methods)

    if multi:
        psi_pert = calc_psi(ranks_pert, sel_mcdm_methods, p_val)
        psi_rank_pert = rank_with_ties(psi_pert, ascending=False)

    # --- Build comparison table ---
    cols_table = ["Alternative"]
    for m in sel_mcdm_methods:
        cols_table += [f"{METHOD_LABELS[m]} (baseline)", f"{METHOD_LABELS[m]} (reduced)"]
    if multi:
        cols_table += [f"PSI Rank (baseline, p={p_val:.2f})", f"PSI Rank (reduced, p={p_val:.2f})"]

    rows_table = []
    reversal_found = False
    for idx_out, i in enumerate(keep_idx):
        row = [keep_names[idx_out]]
        method_reversal = False
        for m in sel_mcdm_methods:
            base_r = int(base_method_ranks_kept[m][idx_out]) if m in base_method_ranks_kept else "—"
            pert_r = int(ranks_pert[m][idx_out])
            row += [base_r, pert_r]
            if base_r != pert_r:
                method_reversal = True
        if multi:
            base_psi_r = int(base_psi_rank_kept[idx_out]) if base_method_ranks_kept else "—"
            pert_psi_r = int(psi_rank_pert[idx_out])
            row += [base_psi_r, pert_psi_r]
            if base_psi_r != pert_psi_r:
                method_reversal = True
        if method_reversal:
            reversal_found = True
        rows_table.append(row)

    df_compare = pd.DataFrame(rows_table, columns=cols_table)

    st.markdown(
        f"**Excluded:** {', '.join(excluded)}  |  "
        f"**Remaining:** {', '.join(keep_names)}  |  p = {p_val:.2f}"
    )
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    if reversal_found:
        st.markdown('<div style="background:#FFF8E1;border-left:3px solid #D97706;padding:8px 12px;border-radius:4px;margin:6px 0;font-family:Times New Roman,Tinos,Times,serif;color:#7B5800;font-size:13px;">Rank-reversal detected: one or more alternatives changed position after exclusion.</div>', unsafe_allow_html=True)
    else:
        st.success(
            "✅ No rank-reversal detected: the relative ranking of the remaining alternatives "
            "is identical to their baseline positions. The result is robust to the exclusion "
            f"of {', '.join(excluded)}."
        )

    



def validation_normalisation_sensitivity():
    st.subheader("5. Normalisation sensitivity")
    
    names = st.session_state.proc_names
    l3_cats = ordered_l3_cats()
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    multi = len(sel_mcdm_methods) > 1

    # --- Alternative normalisation functions ---
    def minmax_norm(vals, benefit):
        vals = np.asarray(vals, dtype=float)
        mn, mx = vals.min(), vals.max()
        rng = (mx - mn) or 1.0
        if benefit:
            return (vals - mn) / rng
        else:
            return (mx - vals) / rng

    def vector_norm(vals, benefit):
        vals = np.asarray(vals, dtype=float)
        denom = np.sqrt(np.sum(vals ** 2)) or 1.0
        normed = vals / denom
        if benefit:
            return normed
        else:
            mx = normed.max() or 1.0
            return mx - normed

    NORM_OPTIONS = {
        "Min-max": minmax_norm,
        "Vector": vector_norm,
        "N2 (baseline)": n2_norm,
    }

    alt_norm_label = st.selectbox(
        "Alternative normalisation method",
        [k for k in NORM_OPTIONS if k != "N2 (baseline)"],
        key="norm_sens_select",
    )
    p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                       value=0.5, step=0.01, key="norm_sens_p_slider")

    alt_norm_func = NORM_OPTIONS[alt_norm_label]

    # --- Baseline ranks (from session state) ---
    base_psi_rank = None
    if multi and method_ranks:
        base_psi = calc_psi(method_ranks, sel_mcdm_methods, p_val)
        base_psi_rank = rank_with_ties(base_psi, ascending=False)

    # --- Alternative pipeline ---
    cat_scores_alt = {}
    for ckey in l3_cats:
        ind_names, ind_units, benefits = get_full_indicators(ckey)
        n_ind = len(ind_names)
        raw, _ = get_raw_matrix(ckey)
        # MEREC-norm + weights unchanged
        nm = np.zeros_like(raw)
        for j in range(n_ind):
            nm[j] = merec_norm(raw[j], benefits[j])
        w_ind = merec_weights(nm)
        # Swap N2 for chosen normalisation
        alt_n = np.zeros_like(raw)
        for j in range(n_ind):
            alt_n[j] = alt_norm_func(raw[j], benefits[j])
        cat_scores_alt[ckey] = (alt_n * w_ind[:, None]).sum(axis=0)

    mat_alt = np.array([cat_scores_alt[c] for c in l3_cats])
    w_cat_alt = get_category_weights(mat_alt, sel_weight_methods)
    weighted_mat_alt = mat_alt * w_cat_alt[:, None]
    ranks_alt = run_mcdm_suite(weighted_mat_alt, w_cat_alt, sel_mcdm_methods)

    alt_psi_rank = None
    if multi:
        psi_alt = calc_psi(ranks_alt, sel_mcdm_methods, p_val)
        alt_psi_rank = rank_with_ties(psi_alt, ascending=False)

    # --- Comparison table ---
    cols_table = ["Alternative"]
    for m in sel_mcdm_methods:
        cols_table += [f"{METHOD_LABELS[m]} (N2 baseline)", f"{METHOD_LABELS[m]} ({alt_norm_label})"]
    if multi:
        cols_table += [f"PSI Rank (N2, p={p_val:.2f})", f"PSI Rank ({alt_norm_label}, p={p_val:.2f})"]

    rows_table = []
    reversal_found = False
    for pi, name in enumerate(names):
        row = [name]
        for m in sel_mcdm_methods:
            base_r = int(method_ranks[m][pi]) if m in method_ranks else "—"
            alt_r  = int(ranks_alt[m][pi])
            row += [base_r, alt_r]
            if base_r != "—" and base_r != alt_r:
                reversal_found = True
        if multi:
            base_psi_r = int(base_psi_rank[pi]) if base_psi_rank is not None else "—"
            alt_psi_r  = int(alt_psi_rank[pi]) if alt_psi_rank is not None else "—"
            row += [base_psi_r, alt_psi_r]
            if base_psi_r != "—" and base_psi_r != alt_psi_r:
                reversal_found = True
        rows_table.append(row)

    st.markdown(
        f"**Baseline normalisation:** N2  |  "
        f"**Alternative:** {alt_norm_label}  |  p = {p_val:.2f}"
    )
    st.dataframe(pd.DataFrame(rows_table, columns=cols_table),
                 use_container_width=True, hide_index=True)

    if reversal_found:
        st.warning(
            f"⚠️ Rank change detected: switching from N2 to {alt_norm_label} normalisation "
            "changes one or more rankings. The result has some sensitivity to normalisation choice."
        )
    else:
        st.markdown(f'<div style="background:#E8F5E9;border-left:3px solid #16A34A;padding:8px 12px;border-radius:4px;margin:6px 0;font-family:Times New Roman,Tinos,Times,serif;color:#1B5E20;font-size:13px;">No rank change under {alt_norm_label} normalisation. The result is robust to normalisation method choice.</div>', unsafe_allow_html=True)

    


def validation_intro():
    st.header("Step 13 - Validation (optional)")

    choice = st.radio(
        "Choose a validation method",
        [
            "None - skip validation",
            "1. Weighting-method sensitivity",
            "2. Benefit/Cost indicator sensitivity",
            "3. Monte Carlo uncertainty (Dirichlet)",
            "4. Rank-reversal test",
            "5. Normalisation sensitivity",
        ],
        index=0, key="validation_radio",
    )
    st.session_state.validation_choice = choice

    if choice.startswith("1."):
        validation_weight_sensitivity()
    elif choice.startswith("2."):
        validation_bc_sensitivity()
    elif choice.startswith("3."):
        validation_monte_carlo()
    elif choice.startswith("4."):
        validation_rank_reversal()
    elif choice.startswith("5."):
        validation_normalisation_sensitivity()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back to results"):
            st.session_state.step = 12
            st.rerun()
    with c2:
        if st.button("Reset all"):
            reset_all()
            st.rerun()


def validation_weight_sensitivity():
    st.subheader("1. Weighting-method sensitivity")

    names = st.session_state.proc_names
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    mat = np.array([cat_scores[c] for c in l3_cats])

    rows = [{"Process": name} for name in names]
    for combo_methods, combo_label in WEIGHT_COMBO_SETS:
        w = get_category_weights(mat, set(combo_methods))
        weighted_mat = mat * w[:, None]
        ranks = run_mcdm_suite(weighted_mat, w, ALL_MCDM_KEYS)
        for pi in range(len(names)):
            for mkey in ALL_MCDM_KEYS:
                col = f"{combo_label} - {METHOD_LABELS[mkey]}"
                rows[pi][col] = int(ranks[mkey][pi])

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Overall ranking across category combinations (this weighting scheme set)**")

    wm_options = {"equal": "Equal", "entropy": "Entropy", "critic": "CRITIC"}
    if "sens1_wm" not in st.session_state:
        st.session_state.sens1_wm = set(st.session_state.sel_weight_methods) or {"equal"}
    cols = st.columns(3)
    for i, (k, lbl) in enumerate(wm_options.items()):
        with cols[i]:
            checked = k in st.session_state.sens1_wm
            new_val = st.checkbox(lbl, value=checked, key=f"sens1wm_{k}")
            if new_val:
                st.session_state.sens1_wm.add(k)
            else:
                st.session_state.sens1_wm.discard(k)
    if not st.session_state.sens1_wm:
        st.session_state.sens1_wm = {"equal"}

    sens1_p = st.slider("p for combination view", min_value=0.0, max_value=1.0,
                         value=0.5, step=0.01, key="sens1_p_slider")

    combos = get_combinations(l3_cats)
    cat_initial = {c: CAT_SHORT.get(c, CATS[c]["label"][:1]) for c in l3_cats}
    combo_labels = ["+".join(cat_initial[c] for c in combo) for combo in combos]
    n_proc = len(names)

    rank_grid = np.zeros((n_proc, len(combos)), dtype=int)
    for ci, combo in enumerate(combos):
        if len(combo) == 1:
            rank_grid[:, ci] = rank_with_ties(cat_scores[combo[0]], ascending=False)
        else:
            sub_mat = np.array([cat_scores[c] for c in combo])
            sub_w = get_category_weights(sub_mat, st.session_state.sens1_wm)
            sub_weighted = sub_mat * sub_w[:, None]
            sub_ranks = run_mcdm_suite(sub_weighted, sub_w, ALL_MCDM_KEYS)
            psi_combo = calc_psi(sub_ranks, ALL_MCDM_KEYS, sens1_p)
            rank_grid[:, ci] = rank_with_ties(psi_combo, ascending=False)

    apply_mpl_style()
    n_combos_c = len(combos)
    fig, ax = plt.subplots(figsize=(max(8, n_combos_c * 0.22), 4.2))
    for pi, name in enumerate(names):
        ax.scatter(range(n_combos_c), rank_grid[pi, :],
                   label=name, s=40, marker="s",
                   color=PROC_COLORS[pi % len(PROC_COLORS)], zorder=3)
    ax.set_xticks(range(n_combos_c))
    ax.set_xticklabels(combo_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Rank", fontsize=10)
    ax.set_xlabel("Category combination", fontsize=10)
    ax.set_ylim(n_proc + 0.5, 0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1),
              ncol=len(names), fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    mpl_show(fig)


def validation_bc_sensitivity():
    st.subheader("2. Benefit/Cost indicator sensitivity")

    l3_cats = ordered_l3_cats()
    names = st.session_state.proc_names
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})

    c1, c2 = st.columns(2)
    with c1:
        ben_pct = st.slider("Benefit-type indicators (%)", min_value=-25, max_value=25,
                             value=0, step=5, key="bc_ben_slider")
    with c2:
        cost_pct = st.slider("Cost-type indicators (%)", min_value=-25, max_value=25,
                              value=0, step=5, key="bc_cost_slider")

    p_val = st.slider("p value", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                       key="bc_p_slider")

    multi = len(sel_mcdm_methods) > 1
    cols = ["Process"] + [METHOD_LABELS[m] for m in sel_mcdm_methods]
    if multi and method_ranks:
        cols.append("PSI Rank (p=0.50)")
        base_psi = calc_psi(method_ranks, sel_mcdm_methods, 0.5)
        base_psi_rank = rank_with_ties(base_psi, ascending=False)

    base_rows = []
    for pi, name in enumerate(names):
        row = [name] + [int(method_ranks[m][pi]) for m in sel_mcdm_methods if m in method_ranks]
        if multi and method_ranks:
            row.append(int(base_psi_rank[pi]))
        base_rows.append(row)

    st.markdown("**Baseline results (Step 12, unperturbed)**")
    st.dataframe(pd.DataFrame(base_rows, columns=cols), use_container_width=True, hide_index=True)

    cat_scores_pert = {}
    for ckey in l3_cats:
        raw, benefits = get_raw_matrix(ckey)
        raw_pert = raw.copy()
        for j in range(raw.shape[0]):
            pct = ben_pct if benefits[j] else cost_pct
            raw_pert[j] = raw[j] * (1 + pct / 100.0)
        cat_scores_pert[ckey] = compute_category_score_from_raw(ckey, raw_pert)

    mat = np.array([cat_scores_pert[c] for c in l3_cats])
    w = get_category_weights(mat, sel_weight_methods)
    weighted_mat = mat * w[:, None]
    ranks = run_mcdm_suite(weighted_mat, w, sel_mcdm_methods)

    p_cols = ["Process"] + [METHOD_LABELS[m] for m in sel_mcdm_methods]
    if multi:
        p_cols.append(f"PSI Rank (p={p_val:.2f})")
        psi_vals = calc_psi(ranks, sel_mcdm_methods, p_val)
        psi_ranks = rank_with_ties(psi_vals, ascending=False)

    p_rows = []
    for pi, name in enumerate(names):
        row = [name] + [int(ranks[m][pi]) for m in sel_mcdm_methods]
        if multi:
            row.append(int(psi_ranks[pi]))
        p_rows.append(row)

    st.markdown(
        f"**Perturbed results** - Benefit indicators: **{ben_pct:+d}%**, "
        f"Cost indicators: **{cost_pct:+d}%**, p = **{p_val:.2f}**"
    )
    st.dataframe(pd.DataFrame(p_rows, columns=p_cols), use_container_width=True, hide_index=True)


def validation_monte_carlo():
    st.subheader("3. Monte Carlo uncertainty (Dirichlet)")

    l3_cats = ordered_l3_cats()
    names = st.session_state.proc_names
    n_proc = len(names)
    cat_scores = st.session_state.cat_scores
    final_w = st.session_state.final_cat_weights
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS

    mat = np.array([cat_scores[c] for c in l3_cats])
    k_value, w_eq, w_en, w_cr = compute_dirichlet_k(mat)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Data-driven k (inter-method agreement)", f"{k_value:.1f} / 100")
    with c2:
        p_val = st.slider("p value (for PSI compromise rank)", min_value=0.0, max_value=1.0,
                           value=0.5, step=0.01, key="mc_p_slider")

    with st.expander("Show Equal/Entropy/CRITIC weights used to compute k"):
        wdf = pd.DataFrame(
            [w_eq, w_en, w_cr], index=["Equal", "Entropy", "CRITIC"],
            columns=[CATS[c]["label"] for c in l3_cats],
        ).round(4)
        st.dataframe(wdf, use_container_width=True)

    run_mc = st.button("Run Monte Carlo simulation (10,000 iterations)", type="primary")

    if run_mc:
        n_iter = 10000
        alpha_scale = 2.0 + (k_value / 100.0) * 200.0
        alpha = np.maximum(final_w * alpha_scale, 0.05)

        rng = np.random.default_rng()
        draws = rng.dirichlet(alpha, size=n_iter)

        rank_counts = {m: np.zeros((n_proc, n_proc), dtype=int) for m in sel_mcdm_methods}
        psi_rank_counts = np.zeros((n_proc, n_proc), dtype=int) if len(sel_mcdm_methods) > 1 else None

        progress = st.progress(0, text="Running Monte Carlo simulation...")
        batch = max(1, n_iter // 20)

        for it in range(n_iter):
            w_draw = draws[it]
            weighted_mat = mat * w_draw[:, None]
            ranks_draw = run_mcdm_suite(weighted_mat, w_draw, sel_mcdm_methods)
            for m in sel_mcdm_methods:
                for pi in range(n_proc):
                    r = ranks_draw[m][pi]
                    rank_counts[m][pi, r - 1] += 1
            if psi_rank_counts is not None:
                psi_vals = calc_psi(ranks_draw, sel_mcdm_methods, p_val)
                psi_ranks = rank_with_ties(psi_vals, ascending=False)
                for pi in range(n_proc):
                    psi_rank_counts[pi, psi_ranks[pi] - 1] += 1
            if it % batch == 0:
                progress.progress(min(1.0, it / n_iter), text=f"Running Monte Carlo simulation... {it}/{n_iter}")
        progress.progress(1.0, text="Done.")

        st.session_state.mc_rank_counts = rank_counts
        st.session_state.mc_psi_rank_counts = psi_rank_counts
        st.session_state.mc_n_iter = n_iter
        st.session_state.mc_methods_used = sel_mcdm_methods
        st.session_state.mc_k_value = k_value

    if "mc_rank_counts" in st.session_state:
        rank_counts = st.session_state.mc_rank_counts
        psi_rank_counts = st.session_state.mc_psi_rank_counts
        n_iter = st.session_state.mc_n_iter
        methods_used = st.session_state.mc_methods_used

        st.markdown(
            f"**Results from {n_iter:,} iterations** "
            f"(k = {st.session_state.mc_k_value:.1f}, rank distribution as % of draws)"
        )

        for m in methods_used:
            st.markdown(f"**{METHOD_LABELS[m]}**")
            pct_table = (rank_counts[m] / n_iter * 100).round(1)
            df = pd.DataFrame(pct_table, index=names, columns=[f"Rank {r+1}" for r in range(n_proc)])
            st.dataframe(df, use_container_width=True)

        if psi_rank_counts is not None:
            st.markdown(f"**PSI compromise rank (p={p_val:.2f})**")
            pct_table = (psi_rank_counts / n_iter * 100).round(1)
            df = pd.DataFrame(pct_table, index=names, columns=[f"Rank {r+1}" for r in range(n_proc)])
            st.dataframe(df, use_container_width=True)



# ============================================================================
# STEP 14 - AUXILIARY ASSESSMENT (optional)
# ============================================================================





def analytics_category_weighted_contribution():
    st.subheader("A. Weighted category contribution")
    
    names = st.session_state.proc_names
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    final_w = st.session_state.final_cat_weights

    cat_labels = [CATS[c]["label"] for c in l3_cats]
    cat_colors = [CATS[c]["color"] for c in l3_cats]

    # Compute contributions: shape (n_cats, n_proc)
    contribs = np.array([cat_scores[c] * final_w[ci]
                         for ci, c in enumerate(l3_cats)])
    totals = contribs.sum(axis=0)  # total weighted score per process

    # ── Stacked bar chart ────────────────────────────────────────────────────
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(max(5, len(names)*1.5), 3.8))
    bottoms_d = np.zeros(len(names))
    for ci, ckey in enumerate(l3_cats):
        ax.bar(names, contribs[ci], bottom=bottoms_d,
               color=CATS[ckey]["color"], label=CATS[ckey]["label"],
               edgecolor="white", linewidth=0.5)
        for pi in range(len(names)):
            if contribs[ci][pi] > 0.005:
                ax.text(pi, bottoms_d[pi] + contribs[ci][pi]/2,
                        f"{contribs[ci][pi]:.4f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms_d += contribs[ci]
    ax.set_ylabel("Weighted category score", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=len(l3_cats), fontsize=8, frameon=False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    mpl_show(fig)

    # ── Percentage breakdown table ────────────────────────────────────────────
    st.markdown("**Percentage contribution per category**")
    rows = []
    for pi, name in enumerate(names):
        row = {"Process": name, "Total score": round(float(totals[pi]), 4)}
        for ci, ckey in enumerate(l3_cats):
            pct = (contribs[ci, pi] / totals[pi] * 100) if totals[pi] > 0 else 0.0
            row[f"{CATS[ckey]['label']} (%)"] = round(pct, 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Category weight table ─────────────────────────────────────────────────
    st.markdown("**RCW category weights applied**")
    w_rows = [{"Category": CATS[l3_cats[ci]]["label"],
               "RCW weight": round(float(final_w[ci]), 4),
               "Weight (%)": round(float(final_w[ci]) * 100, 1)}
              for ci in range(len(l3_cats))]
    st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)


def analytics_leave_one_out():
    st.subheader("B. Leave-one-out category analysis")
    
    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    multi = len(sel_mcdm_methods) > 1

    if len(l3_cats) < 2:
        st.warning("Need at least 2 categories selected for leave-one-out analysis.")
        return

    p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                       value=0.5, step=0.01, key="loo_p_slider")

    # ── Baseline PSI ranks ────────────────────────────────────────────────────
    if multi and method_ranks:
        base_psi = calc_psi(method_ranks, sel_mcdm_methods, p_val)
        base_psi_ranks = rank_with_ties(base_psi, ascending=False)
    else:
        st.warning("Need at least 2 MCDM methods for PSI-based leave-one-out.")
        return

    # ── Leave-one-out runs ────────────────────────────────────────────────────
    loo_results = {}  # ckey -> psi_ranks array

    for excl_ci, excl_ckey in enumerate(l3_cats):
        reduced_cats = [c for c in l3_cats if c != excl_ckey]
        mat_red = np.array([cat_scores[c] for c in reduced_cats])
        w_red = get_category_weights(mat_red, sel_weight_methods)
        wm_red = mat_red * w_red[:, None]
        ranks_red = run_mcdm_suite(wm_red, w_red, sel_mcdm_methods)
        psi_red = calc_psi(ranks_red, sel_mcdm_methods, p_val)
        loo_results[excl_ckey] = rank_with_ties(psi_red, ascending=False)

    # ── Rank change matrix ────────────────────────────────────────────────────
    st.markdown("**Rank change when each category is excluded**")

    matrix_rows = []
    any_change = False
    for excl_ckey in l3_cats:
        row = {"Category excluded": CATS[excl_ckey]["label"]}
        for pi, name in enumerate(names):
            base_r = int(base_psi_ranks[pi])
            new_r = int(loo_results[excl_ckey][pi])
            delta = new_r - base_r
            if delta != 0:
                any_change = True
            row[f"{name} (Δ rank)"] = (f"+{delta}" if delta > 0 else
                                        str(delta) if delta != 0 else "—")
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    z_vals = []
    for excl_ckey in l3_cats:
        row_z = []
        for pi in range(n_proc):
            delta = int(loo_results[excl_ckey][pi]) - int(base_psi_ranks[pi])
            row_z.append(delta)
        z_vals.append(row_z)

    apply_mpl_style()
    z_arr = np.array(z_vals, dtype=float)
    cat_lbls = [CATS[c]["label"] for c in l3_cats]
    cmap = LinearSegmentedColormap.from_list("rg", ["#16A34A", "#f8fafc", "#DC2626"])
    vext = max(abs(z_arr.min()), abs(z_arr.max()), 1)
    fig, ax = plt.subplots(figsize=(max(4, len(names)*1.2),
                                     max(2.5, len(l3_cats)*0.7)))
    im = ax.imshow(z_arr, cmap=cmap, vmin=-vext, vmax=vext, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(len(cat_lbls))); ax.set_yticklabels(cat_lbls, fontsize=9)
    for r in range(len(cat_lbls)):
        for c in range(len(names)):
            v = int(z_arr[r, c])
            txt = f"+{v}" if v > 0 else str(v) if v != 0 else "—"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=10, color="#0D2B5E", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Rank change", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("Process", fontsize=9)
    ax.set_ylabel("Category excluded", fontsize=9)
    plt.tight_layout()
    mpl_show(fig)

    # ── Within-category ranking ───────────────────────────────────────────────
    st.divider()
    st.markdown("**Within-category ranking**")

    wc_rows = []
    for ckey in l3_cats:
        scores = cat_scores[ckey]
        ranks = rank_with_ties(scores, ascending=False)
        row = {"Category": CATS[ckey]["label"]}
        for pi, name in enumerate(names):
            row[f"{name} (score)"] = round(float(scores[pi]), 4)
            row[f"{name} (rank)"] = int(ranks[pi])
        wc_rows.append(row)
    st.dataframe(pd.DataFrame(wc_rows), use_container_width=True, hide_index=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    st.divider()
    if not any_change:
        st.markdown('<div style="background:#E8F5E9;border-left:3px solid #16A34A;padding:8px 12px;border-radius:4px;margin:6px 0;font-family:Times New Roman,Tinos,Times,serif;color:#1B5E20;font-size:13px;">No rank changes detected across all leave-one-out runs. The PSI ranking is robust to the exclusion of any single category.</div>', unsafe_allow_html=True)
    else:
        critical = []
        for excl_ckey in l3_cats:
            for pi in range(n_proc):
                delta = int(loo_results[excl_ckey][pi]) - int(base_psi_ranks[pi])
                if delta != 0:
                    critical.append(CATS[excl_ckey]["label"])
                    break
        critical = list(dict.fromkeys(critical))
        st.warning(
            f"⚠️ Rank changes detected when the following "
            f"{'category is' if len(critical) == 1 else 'categories are'} excluded: "
            f"**{', '.join(critical)}**. "
            f"{'This category is' if len(critical) == 1 else 'These categories are'} "
            f"critical to the current ranking."
        )


def analytics_category_intro():
    st.subheader("1. Category-wise contribution analysis")

    sub = st.radio(
        "Select analysis",
        [
            "None - skip",
            "A. Weighted category contribution",
            "B. Leave-one-out category analysis",
        ],
        index=0, key="cat_analysis_radio",
    )

    if sub.startswith("A."):
        analytics_category_weighted_contribution()
    elif sub.startswith("B."):
        analytics_leave_one_out()



def analytics_indicator_contribution():
    st.subheader("A. Indicator contribution share")
    
    names = st.session_state.proc_names
    l3_cats = ordered_l3_cats()
    n2_data = st.session_state.n2_data
    merec_w = st.session_state.merec_w

    for ckey in l3_cats:
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)
        n_ind = len(ind_names)
        n2 = n2_data[ckey]        # shape (n_ind, n_proc)
        w_ind = merec_w[ckey]     # shape (n_ind,)

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        # Weighted scores per indicator per process
        weighted = n2 * w_ind[:, None]          # (n_ind, n_proc)
        cat_totals = weighted.sum(axis=0)        # (n_proc,)
        shares = np.where(
            cat_totals > 0,
            weighted / cat_totals * 100,
            0.0
        )                                        # (n_ind, n_proc)

        # Table
        rows = []
        for j in range(n_ind):
            row = {"Indicator": f"{ind_names[j]} ({ind_units[j]})"}
            for pi, name in enumerate(names):
                row[f"{name} (%)"] = round(float(shares[j, pi]), 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

        # Grouped bar chart — one group per indicator
        apply_mpl_style()
        ind_labels_c = [f"{ind_names[j]} ({ind_units[j]})" for j in range(n_ind)]
        x = np.arange(n_ind)
        w_bar = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(max(5, n_ind*1.4), 2.8))
        for pi, name in enumerate(names):
            offsets = x + (pi - len(names)/2 + 0.5) * w_bar
            vals = [float(shares[j, pi]) for j in range(n_ind)]
            ax.bar(offsets, vals, width=w_bar*0.9,
                   color=PROC_COLORS[pi % len(PROC_COLORS)], label=name)
            for ox, v in zip(offsets, vals):
                if v > 1:
                    ax.text(ox, v + 0.5, f"{v:.1f}%", ha="center",
                            fontsize=7, color="#0D2B5E")
        ax.set_xticks(x)
        ax.set_xticklabels(ind_labels_c, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Contribution share (%)", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  ncol=len(names), fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        mpl_show(fig)
        st.write("")


def analytics_indicator_loo():
    st.subheader("B. Leave-one-out indicator analysis")
    
    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    multi = len(sel_mcdm_methods) > 1

    if not multi:
        st.warning("Need at least 2 MCDM methods for PSI-based leave-one-out.")
        return

    p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                       value=0.5, step=0.01, key="ind_loo_p")

    # Baseline PSI ranks
    base_psi = calc_psi(method_ranks, sel_mcdm_methods, p_val)
    base_psi_ranks = rank_with_ties(base_psi, ascending=False)

    # Run leave-one-out per indicator per category
    def run_loo(excl_ckey, excl_j):
        """Rerun pipeline excluding indicator j from category excl_ckey."""
        new_cat_scores = {}
        for ckey in l3_cats:
            ind_names_, ind_units_, benefits_ = get_full_indicators(ckey)
            n_ind_ = len(ind_names_)
            raw, _ = get_raw_matrix(ckey)

            if ckey == excl_ckey and n_ind_ > 1:
                # Remove row j
                keep = [jj for jj in range(n_ind_) if jj != excl_j]
                raw_red = raw[keep, :]
                ben_red = [benefits_[jj] for jj in keep]
                nm = np.zeros_like(raw_red)
                n2 = np.zeros_like(raw_red)
                for jj in range(len(keep)):
                    nm[jj] = merec_norm(raw_red[jj], ben_red[jj])
                    n2[jj] = n2_norm(raw_red[jj], ben_red[jj])
                w = merec_weights(nm)
                new_cat_scores[ckey] = (n2 * w[:, None]).sum(axis=0)
            else:
                new_cat_scores[ckey] = cat_scores[ckey]

        mat = np.array([new_cat_scores[c] for c in l3_cats])
        w_cat = get_category_weights(mat, sel_weight_methods)
        wm = mat * w_cat[:, None]
        ranks_ = run_mcdm_suite(wm, w_cat, sel_mcdm_methods)
        psi_ = calc_psi(ranks_, sel_mcdm_methods, p_val)
        return rank_with_ties(psi_, ascending=False)

    any_change = False

    for ckey in l3_cats:
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)
        n_ind = len(ind_names)

        if n_ind < 2:
            st.markdown(
                f"<span style='background:{cat['bg']};color:{cat['color']};"
                f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
                f"font-family:Times New Roman,Tinos,Times,serif;'>"
                f"{cat['label']}</span> — only one indicator, skip.", 
                unsafe_allow_html=True,
            )
            st.write("")
            continue

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;"
            f"font-family:Times New Roman,Tinos,Times,serif;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        # Build rank change matrix for this category
        matrix_rows = []
        z_vals = []
        ind_labels = [f"{ind_names[j]} ({ind_units[j]})" for j in range(n_ind)]

        for j in range(n_ind):
            new_ranks = run_loo(ckey, j)
            row = {"Indicator excluded": ind_labels[j]}
            row_z = []
            for pi, name in enumerate(names):
                delta = int(new_ranks[pi]) - int(base_psi_ranks[pi])
                if delta != 0:
                    any_change = True
                row[f"{name} (Δ rank)"] = (
                    f"+{delta}" if delta > 0 else str(delta) if delta != 0 else "—"
                )
                row_z.append(delta)
            matrix_rows.append(row)
            z_vals.append(row_z)

        st.dataframe(pd.DataFrame(matrix_rows),
                     use_container_width=True, hide_index=True)

        # Heatmap per category
        apply_mpl_style()
        z_arr_h = np.array(z_vals, dtype=float)
        cmap_h = LinearSegmentedColormap.from_list("rg", ["#16A34A","#f8fafc","#DC2626"])
        vext_h = max(abs(z_arr_h.min()), abs(z_arr_h.max()), 1)
        fig, ax = plt.subplots(figsize=(max(4, len(names)*1.2),
                                         max(2, n_ind*0.65)))
        im = ax.imshow(z_arr_h, cmap=cmap_h,
                       vmin=-vext_h, vmax=vext_h, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticks(range(n_ind))
        ax.set_yticklabels(ind_labels, fontsize=8)
        for r in range(n_ind):
            for c in range(len(names)):
                v = int(z_arr_h[r, c])
                txt = f"+{v}" if v > 0 else str(v) if v != 0 else "—"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=10, color="#0D2B5E", fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Rank change", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xlabel("Process", fontsize=9)
        ax.set_ylabel("Indicator excluded", fontsize=9)
        plt.tight_layout()
        mpl_show(fig)
        st.write("")

    # Overall interpretation
    st.divider()
    if not any_change:
        st.markdown('<div style="background:#E8F5E9;border-left:3px solid #16A34A;padding:8px 12px;border-radius:4px;margin:6px 0;font-family:Times New Roman,Tinos,Times,serif;color:#1B5E20;font-size:13px;">No rank changes detected across all indicator leave-one-out runs. The PSI ranking is robust to the exclusion of any single indicator.</div>', unsafe_allow_html=True)
    else:
        st.warning(
            "⚠️ One or more rank changes detected. "
            "Indicators highlighted in red above are critical to the current ranking — "
            "their removal alters the final process order."
        )


def analytics_indicator_intro():
    st.subheader("2. Indicator-wise analysis")

    sub = st.radio(
        "Select analysis",
        [
            "None - skip",
            "A. Indicator contribution share",
            "B. Leave-one-out indicator analysis",
        ],
        index=0, key="ind_analysis_radio",
    )

    if sub.startswith("A."):
        analytics_indicator_contribution()
    elif sub.startswith("B."):
        analytics_indicator_loo()



def analytics_stakeholder_preference():
    st.subheader("3. Stakeholder preference simulation")
    
    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    final_w = st.session_state.final_cat_weights
    multi = len(sel_mcdm_methods) > 1

    cat_labels = [CATS[c]["label"] for c in l3_cats]
    n_cats = len(l3_cats)

    stakeholder = "Stakeholder"

    # ── p value ───────────────────────────────────────────────────────────────
    p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                       value=0.5, step=0.01, key="sh_p_slider")

    # ── Category weight sliders ───────────────────────────────────────────────
    st.markdown("**Adjust category weights (%)**")

    custom_weights = {}
    for ci, ckey in enumerate(l3_cats):
        cat = CATS[ckey]
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(
                f"<span style='background:{cat['bg']};color:{cat['color']};"
                f"padding:2px 10px;border-radius:10px;font-size:13px;font-weight:600;"
                f"font-family:Times New Roman,Tinos,Times,serif;'>"
                f"{cat['label']}</span>", unsafe_allow_html=True,
            )
        with c2:
            default_val = int(round(float(final_w[ci]) * 100))
            val = st.slider(
                cat["label"], min_value=0, max_value=100,
                value=st.session_state.get(f"sh_w_{ckey}", default_val),
                step=1, key=f"sh_w_{ckey}",
                label_visibility="collapsed",
            )
        custom_weights[ckey] = val

    total = sum(custom_weights.values())
    c1, c2 = st.columns([3, 1])
    with c1:
        st.progress(min(total / 100, 1.0))
    with c2:
        if total == 100:
            st.success(f"Total: {total}% ✅")
        else:
            st.error(f"Total: {total}%")

    if total != 100:
        st.error("Adjust weights to sum to exactly 100% before running.")
        return

    # ── Run button ────────────────────────────────────────────────────────────
    if not st.button("Run simulation →", type="primary", key="sh_run"):
            return

    # ── Original ranking ──────────────────────────────────────────────────────
    if multi and method_ranks:
        base_psi = calc_psi(method_ranks, sel_mcdm_methods, p_val)
        base_psi_ranks = rank_with_ties(base_psi, ascending=False)
    else:
        st.warning("Need at least 2 MCDM methods for PSI ranking.")
        return

    # ── Rerun pipeline with custom weights ────────────────────────────────────
    w_custom = np.array([custom_weights[c] / 100.0 for c in l3_cats])
    mat = np.array([cat_scores[c] for c in l3_cats])
    wm_custom = mat * w_custom[:, None]
    ranks_custom = run_mcdm_suite(wm_custom, w_custom, sel_mcdm_methods)
    psi_custom = calc_psi(ranks_custom, sel_mcdm_methods, p_val)
    psi_ranks_custom = rank_with_ties(psi_custom, ascending=False)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"**Ranking comparison — Original RCW vs {stakeholder}**")

    comp_rows = []
    any_change = False
    for pi, name in enumerate(names):
        orig_r = int(base_psi_ranks[pi])
        new_r = int(psi_ranks_custom[pi])
        delta = new_r - orig_r
        if delta != 0:
            any_change = True
        comp_rows.append({
            "Process": name,
            "Original PSI rank (RCW)": orig_r,
            f"New PSI rank ({stakeholder})": new_r,
            "Rank change": (f"+{delta}" if delta > 0
                            else str(delta) if delta != 0 else "—"),
        })

    st.dataframe(pd.DataFrame(comp_rows),
                 use_container_width=True, hide_index=True)

    if any_change:
        st.warning(
            f"⚠️ The ranking changes under {stakeholder}'s weights. "
            "The current RCW-based ranking is sensitive to this stakeholder's priorities."
        )
    else:
        st.markdown(f'<div style="background:#E8F5E9;border-left:3px solid #16A34A;padding:8px 12px;border-radius:4px;margin:6px 0;font-family:Times New Roman,Tinos,Times,serif;color:#1B5E20;font-size:13px;">The ranking is identical under the custom weights. The result is robust to this stakeholder profile.</div>', unsafe_allow_html=True)

    # ── Method-by-method breakdown ────────────────────────────────────────────
    st.divider()
    st.markdown(f"**Method-by-method ranking under {stakeholder}'s weights**")

    method_rows = []
    for pi, name in enumerate(names):
        row = {"Process": name,
               f"PSI rank ({stakeholder})": int(psi_ranks_custom[pi]),
               f"PSI score ({stakeholder})": round(float(psi_custom[pi]), 4)}
        for m in sel_mcdm_methods:
            row[METHOD_LABELS[m]] = int(ranks_custom[m][pi])
        method_rows.append(row)

    st.dataframe(pd.DataFrame(method_rows),
                 use_container_width=True, hide_index=True)

    # ── Weight comparison chart ───────────────────────────────────────────────
    st.divider()
    st.markdown("**Category weight comparison — RCW vs stakeholder**")

    apply_mpl_style()
    x_w = np.arange(n_cats)
    w_bar = 0.35
    fig, ax = plt.subplots(figsize=(max(5, n_cats*1.2), 3.0))
    rcw_vals = [round(float(final_w[ci])*100, 1) for ci in range(n_cats)]
    sh_vals  = [custom_weights[c] for c in l3_cats]
    ax.bar(x_w - w_bar/2, rcw_vals, width=w_bar,
           color="#185FA5", label="RCW (original)", edgecolor="white")
    ax.bar(x_w + w_bar/2, sh_vals,  width=w_bar,
           color="#EA580C", label="Stakeholder",    edgecolor="white")
    for xi, (rv, sv) in enumerate(zip(rcw_vals, sh_vals)):
        ax.text(xi - w_bar/2, rv + 0.5, f"{rv}%", ha="center", fontsize=8, color="#0D2B5E")
        ax.text(xi + w_bar/2, sv + 0.5, f"{sv}%", ha="center", fontsize=8, color="#0D2B5E")
    ax.set_xticks(x_w)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Weight (%)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=2, fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    mpl_show(fig)

    # ── PSI score comparison chart ────────────────────────────────────────────
    st.markdown("**PSI score comparison — Original vs stakeholder**")

    apply_mpl_style()
    x_p = np.arange(n_proc)
    w_bar_p = 0.35
    fig, ax = plt.subplots(figsize=(max(5, n_proc*1.2), 3.0))
    base_vals = [round(float(base_psi[pi]), 4) for pi in range(n_proc)]
    cust_vals  = [round(float(psi_custom[pi]), 4) for pi in range(n_proc)]
    ax.bar(x_p - w_bar_p/2, base_vals, width=w_bar_p,
           color="#185FA5", label="Original PSI (RCW)", edgecolor="white")
    ax.bar(x_p + w_bar_p/2, cust_vals, width=w_bar_p,
           color="#EA580C", label="Stakeholder PSI",    edgecolor="white")
    ymax = max(max(base_vals), max(cust_vals))
    for xi, (bv, cv) in enumerate(zip(base_vals, cust_vals)):
        ax.text(xi - w_bar_p/2, bv + ymax*0.01, f"{bv:.4f}",
                ha="center", fontsize=7, color="#0D2B5E")
        ax.text(xi + w_bar_p/2, cv + ymax*0.01, f"{cv:.4f}",
                ha="center", fontsize=7, color="#0D2B5E")
    ax.set_xticks(x_p)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("PSI score", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=2, fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    mpl_show(fig)


def analytics_indicator_robustness():
    st.subheader("4. Indicator robustness analysis")

    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    multi = len(sel_mcdm_methods) > 1

    if not multi:
        st.warning("Need at least 2 MCDM methods for PSI-based robustness analysis.")
        return

    # Build full indicator list
    all_indicators = []
    for ckey in l3_cats:
        ind_names, ind_units, benefits = get_full_indicators(ckey)
        for j, (iname, iunit) in enumerate(zip(ind_names, ind_units)):
            all_indicators.append({
                "ckey": ckey, "j": j,
                "label": f"{CATS[ckey]['label']} — {iname} ({iunit})",
                "benefit": benefits[j],
            })

    if not all_indicators:
        st.warning("No indicators available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_label = st.selectbox(
            "Indicator to vary", [ind["label"] for ind in all_indicators],
            key="rob_ind_sel"
        )
    with c2:
        range_pct = st.slider(
            "Sweep range (% of current value)",
            min_value=50, max_value=500, value=200, step=50,
            key="rob_range_pct"
        )
    with c3:
        p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                           value=0.5, step=0.01, key="rob_p_slider")

    sel_ind = next(ind for ind in all_indicators if ind["label"] == sel_label)
    ckey_s, j_s = sel_ind["ckey"], sel_ind["j"]
    benefit = sel_ind["benefit"]

    # Current values for this indicator across all processes
    cur_vals = np.array([
        st.session_state.indicator_values.get((ckey_s, j_s, pi), 0.0)
        for pi in range(n_proc)
    ])

    # Sweep range — vary the indicator across all processes simultaneously
    # using process-specific current values as anchors
    # We sweep a multiplier from low to high
    lo_mult = max(0.05, 1 - range_pct/100)
    hi_mult = 1 + range_pct/100
    steps = 100
    multipliers = np.linspace(lo_mult, hi_mult, steps)

    # For each multiplier, rerun full pipeline
    rank_trajectories = np.zeros((n_proc, steps), dtype=int)

    for si, mult in enumerate(multipliers):
        # Override indicator values for all processes
        new_cat_scores = {}
        for ckey in l3_cats:
            ind_names_, ind_units_, benefits_ = get_full_indicators(ckey)
            n_ind_ = len(ind_names_)
            raw, _ = get_raw_matrix(ckey)
            raw_mod = raw.copy()
            if ckey == ckey_s:
                raw_mod[j_s, :] = cur_vals * mult
            nm = np.zeros_like(raw_mod)
            n2 = np.zeros_like(raw_mod)
            for j in range(n_ind_):
                nm[j] = merec_norm(raw_mod[j], benefits_[j])
                n2[j] = n2_norm(raw_mod[j], benefits_[j])
            w_ind = merec_weights(nm)
            new_cat_scores[ckey] = (n2 * w_ind[:, None]).sum(axis=0)

        mat = np.array([new_cat_scores[c] for c in l3_cats])
        w_cat = get_category_weights(mat, sel_weight_methods)
        wm = mat * w_cat[:, None]
        ranks_ = run_mcdm_suite(wm, w_cat, sel_mcdm_methods)
        psi_ = calc_psi(ranks_, sel_mcdm_methods, p_val)
        psi_ranks = rank_with_ties(psi_, ascending=False)
        for pi in range(n_proc):
            rank_trajectories[pi, si] = int(psi_ranks[pi])

    # Current multiplier index (closest to 1.0)
    cur_idx = np.argmin(np.abs(multipliers - 1.0))

    # X axis values: actual indicator values (use process 0 as reference scale)
    # Use multiplier * mean current value as x axis label
    mean_cur = cur_vals.mean() if cur_vals.mean() > 0 else 1.0
    x_vals = multipliers * mean_cur

    # ── Matplotlib chart ──────────────────────────────────────────────────────
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))

    linestyles = ["-", "--", ":"]
    for pi, name in enumerate(names):
        ax.plot(x_vals, rank_trajectories[pi],
                label=name, linewidth=2,
                color=PROC_COLORS[pi % len(PROC_COLORS)],
                linestyle=linestyles[pi % len(linestyles)])

    # Mark current position
    ax.axvline(x=mean_cur, color="#888888", linestyle="--",
                linewidth=1, label="Current value")

    ax.set_xlabel(sel_ind["label"], fontsize=10)
    ax.set_ylabel("PSI rank", fontsize=10)
    ax.set_ylim(n_proc + 0.5, 0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"Rank {int(v)}" if v == int(v) else ""))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=n_proc + 1, fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    mpl_show(fig)

    # ── Rank stability intervals table ────────────────────────────────────────
    st.markdown("**Rank stability intervals**")

    unit = sel_ind["label"].split("(")[-1].rstrip(")")
    rows_stab = []
    for pi, name in enumerate(names):
        traj = rank_trajectories[pi]
        intervals = []
        cur_r = traj[0]
        start_x = x_vals[0]
        for si in range(1, steps):
            if traj[si] != cur_r:
                intervals.append({"Rank": int(cur_r),
                                   "From": round(float(start_x), 2),
                                   "To": round(float(x_vals[si-1]), 2)})
                cur_r = traj[si]
                start_x = x_vals[si]
        intervals.append({"Rank": int(cur_r),
                           "From": round(float(start_x), 2),
                           "To": round(float(x_vals[-1]), 2)})

        for iv in intervals:
            rows_stab.append({
                "Process": name,
                "PSI rank": iv["Rank"],
                f"From ({unit})": iv["From"],
                f"To ({unit})": iv["To"],
                "Contains current value": "✓" if iv["From"] <= mean_cur <= iv["To"] else "",
            })

    st.dataframe(pd.DataFrame(rows_stab),
                 use_container_width=True, hide_index=True)

    # ── Key insight ───────────────────────────────────────────────────────────
    st.divider()
    insights = []
    for pi, name in enumerate(names):
        cur_rank = int(rank_trajectories[pi, cur_idx])
        # Find range where current rank holds
        stable_from, stable_to = None, None
        for si in range(steps):
            if rank_trajectories[pi, si] == cur_rank:
                if stable_from is None:
                    stable_from = x_vals[si]
                stable_to = x_vals[si]
        if stable_from is not None:
            pct_range = round((stable_to - stable_from) / mean_cur * 100, 1)
            insights.append(
                f"**{name}** — rank {cur_rank} stable across "
                f"{stable_from:.2f} to {stable_to:.2f} {unit} "
                f"({pct_range}% of current value range)"
            )

    for ins in insights:
        st.markdown(ins)



def auxiliary_intro():
    st.header("Step 14 - Analytics (optional)")

    choice = st.radio(
        "Select tool",
        [
            "None - skip",
            "1. Category-wise contribution analysis",
            "2. Indicator-wise analysis",
            "3. Stakeholder preference simulation",
            "4. Indicator robustness analysis",
        ],
        index=0, key="auxiliary_radio",
    )

    if choice.startswith("1."):
        analytics_category_intro()
    elif choice.startswith("2."):
        analytics_indicator_intro()
    elif choice.startswith("3."):
        analytics_stakeholder_preference()
    elif choice.startswith("4."):
        analytics_indicator_robustness()

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("<- Back to validation"):
            st.session_state.step = 13
            st.rerun()
    with c2:
        if st.button("<- Back to results"):
            st.session_state.step = 12
            st.rerun()
    with c3:
        if st.button("Reset all"):
            reset_all()
            st.rerun()



STEPS = {
    0: landing_page, 1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6,
    7: step7, 8: step8, 9: step9, 10: step10, 11: step11, 12: step12,
    13: validation_intro, 14: auxiliary_intro,
}

STEPS[st.session_state.step]()
