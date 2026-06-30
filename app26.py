"""
PRISM — Performance Ranking via Integrated Sustainability Metrics
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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

st.set_page_config(page_title="Performance Ranking via Integrated Sustainability Metrics", page_icon="🧭", layout="wide")

PROC_COLORS = ["#2563EB", "#16A34A", "#EA580C", "#9333EA", "#0891B2",
               "#CA8A04", "#DB2777", "#4F46E5", "#65A30D", "#DC2626"]

CATEGORY_ORDER = ["env", "eco", "soc", "qua", "pro"]

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
    WSM = weighted_mat.sum(axis=0)
    WPM = np.prod(np.maximum(weighted_mat, 1e-9), axis=0)
    mw = WSM.max() or 1.0
    mp = WPM.max() or 1.0
    Q = lam * WSM / mw + (1 - lam) * WPM / mp
    return rank_with_ties(Q, ascending=False)


def spotis(weighted_mat, bounds):
    """
    SPOTIS — Stable Preference Ordering Towards Ideal Solution.
    bounds: np.array shape (n_crit, 2) — col 0 = min bound, col 1 = max bound.
    Ideal point s* = max bound for benefit (already encoded in weighted_mat
    direction), but SPOTIS works on the RAW category scores before weighting,
    so here we receive the weighted matrix and reconstruct distances using the
    passed bounds (also pre-weighted by the same category weights).
    Rank: ascending (lower distance score = better).
    """
    n_crit, n_alt = weighted_mat.shape
    scores = np.zeros(n_alt)
    for i in range(n_alt):
        d = 0.0
        for j in range(n_crit):
            rng = abs(bounds[j, 1] - bounds[j, 0])
            if rng < 1e-12:
                continue
            d += abs(weighted_mat[j, i] - bounds[j, 1]) / rng
        scores[i] = d
    return rank_with_ties(scores, ascending=True)


MCDM_FUNCS = {
    "topsis": lambda wm, w, b=None: topsis(wm),
    "vikor": lambda wm, w, b=None: vikor(wm, w),
    "electre": lambda wm, w, b=None: electre1(wm, w),
    "multimoora": lambda wm, w, b=None: multimoora(wm),
    "waspas": lambda wm, w, b=None: waspas(wm),
    "spotis": lambda wm, w, b=None: spotis(wm, b) if b is not None else rank_with_ties(np.zeros(wm.shape[1]), ascending=True),
}
METHOD_LABELS = {
    "topsis": "TOPSIS", "vikor": "VIKOR", "electre": "ELECTRE I",
    "multimoora": "MULTIMOORA", "waspas": "WASPAS", "spotis": "SPOTIS",
}
ALL_MCDM_KEYS = ["topsis", "vikor", "electre", "multimoora", "waspas", "spotis"]
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


def run_mcdm_suite(weighted_mat, weights, methods, spotis_bounds=None):
    ranks = {}
    for m in methods:
        ranks[m] = MCDM_FUNCS[m](weighted_mat, weights, spotis_bounds)
    return ranks


def get_spotis_bounds(l3_cats, final_w):
    """Build SPOTIS bounds array from session state, scaled by category weights."""
    if "spotis" not in st.session_state.sel_mcdm_methods:
        return None
    bounds_list = []
    for ci, ckey in enumerate(l3_cats):
        mn = st.session_state.spotis_bounds.get(f"min_{ckey}", 0.0) * final_w[ci]
        mx = st.session_state.spotis_bounds.get(f"max_{ckey}", 1.0) * final_w[ci]
        bounds_list.append([mn, mx])
    return np.array(bounds_list)


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
        "step": 1,
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
        "spotis_bounds": {},
        "spotis_buffer_pct": 20.0,
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
    names = list(cat["indicators"])
    units = [st.session_state.sel_units.get(f"{ckey}_{j}", cat["default_units"][j])
             for j in range(len(cat["indicators"]))]
    benefits = list(cat["benefit"])
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
    "11. MCDM methods", "11.5. SPOTIS boundaries (if selected)", "12. Results", "13. Validation (optional)",
    "14. Auxiliary assessment (optional)",
]

with st.sidebar:
    st.title("🧭 PRISM")
    st.caption("Performance Ranking via Integrated Sustainability Metrics")
    st.divider()
    for i, label in enumerate(STEP_LABELS, start=1):
        if i < st.session_state.step:
            st.success(label, icon="✅")
        elif i == st.session_state.step:
            st.info(label, icon="➡️")
        else:
            st.text(label)
    st.divider()
    if st.button("Reset everything", use_container_width=True):
        reset_all()
        st.rerun()


# ============================================================================
# STEP 1 - PROCESSES (FIX 1: no example names pre-filled)
# ============================================================================

def step1():
    st.header("Step 1 - Define processes")
    st.caption("How many manufacturing processes are you comparing? (2-10)")

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
    st.caption("Choose one or more categories to include in the analysis")

    cols = st.columns(5)
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
    st.caption("Besides the predefined indicators, you can add your own to any selected category.")

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
    st.header("Step 4 - Select units for predefined indicators")
    st.caption('Choose a preset unit from the dropdown, or pick "Custom..." to type your own.')

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        st.markdown(f"**{cat['label']}**")
        for j, ind in enumerate(cat["indicators"]):
            key = f"{ckey}_{j}"
            current = st.session_state.sel_units.get(key, cat["default_units"][j])
            options = list(cat["unit_options"][j])
            is_preset = current in options
            display_options = options + [CUSTOM_SENTINEL]
            default_index = options.index(current) if is_preset else len(options)

            c1, c2, c3 = st.columns([2, 1.3, 1.3])
            with c1:
                st.text(ind)
            with c2:
                chosen = st.selectbox(
                    "unit", display_options, index=default_index,
                    key=f"unitsel_{key}", label_visibility="collapsed",
                )
            with c3:
                if chosen == CUSTOM_SENTINEL:
                    custom_val = st.text_input(
                        "custom unit", value=current if not is_preset else "",
                        key=f"unitcustom_{key}", label_visibility="collapsed",
                        placeholder="type unit",
                    )
                    final_unit = custom_val.strip() or "unit"
                else:
                    final_unit = chosen
                    st.caption(" ")
            st.session_state.sel_units[key] = final_unit
        st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("Next ->", type="primary"):
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
    st.caption("Fill in measured values for each process")

    names = st.session_state.proc_names

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;'>"
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
    st.caption(
        "Spearman's rank correlation between indicators within each category. "
        "Informational only - no indicators are removed. Pairs with |rho| > 0.8 are flagged."
    )

    names = st.session_state.proc_names
    n_proc = len(names)
    flagged_any = False

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        ind_names, _, _ = get_full_indicators(ckey)
        n_ind = len(ind_names)

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        if n_ind < 2:
            st.caption("Only one indicator in this category - correlation not applicable.")
            st.write("")
            continue
        if n_proc < 3:
            st.caption("Need at least 3 processes for a meaningful Spearman correlation. Skipped.")
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

        def highlight(val):
            if isinstance(val, (int, float)) and 0.8 < abs(val) < 0.999:
                return "background-color: rgba(220,38,38,0.25)"
            return ""

        st.dataframe(df_corr.style.map(highlight), use_container_width=True)

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
    st.caption("MEREC weight per indicator, computed from MEREC normalisation "
               "(benefit = min/x, cost = x/max) via the standard removal-effect formula.")

    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)
        w = st.session_state.merec_w[ckey]

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;'>"
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
    st.caption("Score = sum(N2 normalised value x MEREC weight). Higher = better.")

    names = st.session_state.proc_names
    for ckey in ordered_sel_cats():
        cat = CATS[ckey]
        scores = st.session_state.cat_scores[ckey]

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        fig = go.Figure(go.Bar(
            x=scores, y=names, orientation="h", marker_color=cat["color"],
            text=[f"{s:.4f}" for s in scores], textposition="outside",
        ))
        fig.update_layout(height=120 + 30 * len(names), margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title=None, yaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

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
    st.caption("Choose which category scores to carry into the MCDM aggregation")

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
    st.caption("Select one or more. Multiple selections trigger RCW consolidation automatically.")

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

        if len(sets) > 1:
            st.info("Multiple methods selected - RCW consolidation applied automatically.")

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
    st.caption("Select one or more. Multiple selections apply PSI compromise ranking.")

    options = {
        "topsis": "TOPSIS - Closest to ideal solution",
        "vikor": "VIKOR - Maximum group utility",
        "electre": "ELECTRE I - Outranking / dominance",
        "multimoora": "MULTIMOORA - Three subordinate rankings",
        "waspas": "WASPAS - Weighted sum-product aggregation",
        "spotis": "SPOTIS - Rank-reversal free, fixed boundary reference",
    }
    for key, label in options.items():
        checked = key in st.session_state.sel_mcdm_methods
        new_val = st.checkbox(label, value=checked, key=f"mmchk_{key}")
        if new_val:
            st.session_state.sel_mcdm_methods.add(key)
        else:
            st.session_state.sel_mcdm_methods.discard(key)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 10
            st.rerun()
    with c2:
        if st.button("Calculate results ->", type="primary"):
            if not st.session_state.sel_mcdm_methods:
                st.error("Select at least one MCDM method.")
            else:
                if "spotis" in st.session_state.sel_mcdm_methods:
                    st.session_state.step = 11.5
                else:
                    st.session_state.step = 12
                st.rerun()


def step11_5():
    st.header("Step 11.5 - SPOTIS boundary values")
    st.caption(
        "SPOTIS requires a fixed ideal reference point defined by the theoretical "
        "minimum and maximum possible value for each category score — independent "
        "of the alternatives in your matrix. Auto-derived from your data with a "
        "percentage buffer. Override any value manually if you have domain knowledge."
    )

    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    names = st.session_state.proc_names

    buffer_pct = st.slider(
        "Boundary buffer (% beyond observed range)",
        min_value=0, max_value=50,
        value=int(st.session_state.spotis_buffer_pct),
        step=5, key="spotis_buffer_slider",
    )
    st.session_state.spotis_buffer_pct = float(buffer_pct)

    st.markdown("**Review and adjust category score bounds**")
    st.caption(
        "Min bound = theoretical worst score. Max bound = theoretical best score. "
        "SPOTIS measures each alternative's distance from the max bound."
    )

    header_cols = st.columns([2, 1, 1, 1, 1])
    with header_cols[0]: st.markdown("**Category**")
    with header_cols[1]: st.markdown("**Observed min**")
    with header_cols[2]: st.markdown("**Observed max**")
    with header_cols[3]: st.markdown("**Bound min**")
    with header_cols[4]: st.markdown("**Bound max**")

    for ci, ckey in enumerate(l3_cats):
        scores = cat_scores[ckey]
        obs_min, obs_max = float(scores.min()), float(scores.max())
        rng = obs_max - obs_min
        buf = rng * buffer_pct / 100.0

        auto_min = obs_min - buf
        auto_max = obs_max + buf

        bkey_min = f"spotis_min_{ckey}"
        bkey_max = f"spotis_max_{ckey}"

        # Initialise or update when buffer changes
        current_min = st.session_state.spotis_bounds.get(f"min_{ckey}", auto_min)
        current_max = st.session_state.spotis_bounds.get(f"max_{ckey}", auto_max)

        row_cols = st.columns([2, 1, 1, 1, 1])
        with row_cols[0]:
            st.markdown(
                f"<span style='background:{CATS[ckey]['bg']};color:{CATS[ckey]['color']};"
                f"padding:2px 8px;border-radius:8px;font-size:13px;font-weight:600;'>"
                f"{CATS[ckey]['label']}</span>", unsafe_allow_html=True,
            )
        with row_cols[1]: st.caption(f"{obs_min:.4f}")
        with row_cols[2]: st.caption(f"{obs_max:.4f}")
        with row_cols[3]:
            val_min = st.number_input(
                "min", value=current_min, key=bkey_min,
                label_visibility="collapsed", format="%g", step=None,
            )
        with row_cols[4]:
            val_max = st.number_input(
                "max", value=current_max, key=bkey_max,
                label_visibility="collapsed", format="%g", step=None,
            )

        st.session_state.spotis_bounds[f"min_{ckey}"] = float(val_min)
        st.session_state.spotis_bounds[f"max_{ckey}"] = float(val_max)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 11
            st.rerun()
    with c2:
        if st.button("Calculate results ->", type="primary"):
            # Validate bounds
            errors = []
            for ckey in l3_cats:
                mn = st.session_state.spotis_bounds.get(f"min_{ckey}", 0)
                mx = st.session_state.spotis_bounds.get(f"max_{ckey}", 1)
                if mn >= mx:
                    errors.append(f"{CATS[ckey]['label']}: min must be less than max")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state.step = 12
                st.rerun()


# ============================================================================
# STEP 12 - RESULTS
# FIX 5: combination chart x-axis labels are HORIZONTAL (tickangle=0).
# ============================================================================

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

    # Build SPOTIS bounds array if SPOTIS is selected
    spotis_bounds = None
    if "spotis" in methods:
        bounds_list = []
        for ci, ckey in enumerate(l3_cats):
            mn = st.session_state.spotis_bounds.get(f"min_{ckey}", 0.0) * final_w[ci]
            mx = st.session_state.spotis_bounds.get(f"max_{ckey}", 1.0) * final_w[ci]
            bounds_list.append([mn, mx])
        spotis_bounds = np.array(bounds_list)

    method_ranks = run_mcdm_suite(weighted_mat, final_w, methods, spotis_bounds)
    st.session_state.last_method_ranks = method_ranks

    st.subheader("MCDM rankings")
    st.caption("Rank 1 = best performing process. Equal scores receive equal (tied) ranks.")

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
        fig = go.Figure()
        for i, name in enumerate(names):
            series = [calc_psi(method_ranks, methods, p)[i] for p in p_range]
            fig.add_trace(go.Scatter(x=p_range, y=series, mode="lines", name=name,
                                      line=dict(color=PROC_COLORS[i % len(PROC_COLORS)], width=2)))
        fig.add_vline(x=p_val, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="p", yaxis_title="PSI", height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Ranking across category combinations")
        n_combo = 2 ** len(l3_cats) - 1
        st.caption(
            f"For every non-empty combination of your {len(l3_cats)} selected Level 3 "
            f"categories ({n_combo} combinations): single-category columns show the "
            "FIXED rank from Step 8 (no MCDM/PSI). Multi-category columns recompute "
            "category weights, re-run the selected MCDM methods, and apply PSI at the "
            "p-value you choose below."
        )

        combo_p = st.slider("p for combination view", min_value=0.0, max_value=1.0,
                             value=0.5, step=0.01, key="combo_p_slider")

        combos = get_combinations(l3_cats)
        cat_initial = {c: CATS[c]["label"][:3] for c in l3_cats}
        combo_labels = ["+".join(cat_initial[c] for c in combo) for combo in combos]

        rank_grid = np.zeros((n_proc, len(combos)), dtype=int)
        for ci, combo in enumerate(combos):
            if len(combo) == 1:
                rank_grid[:, ci] = rank_with_ties(cat_scores[combo[0]], ascending=False)
            else:
                sub_mat = np.array([cat_scores[c] for c in combo])
                sub_w = get_category_weights(sub_mat, st.session_state.sel_weight_methods)
                sub_weighted = sub_mat * sub_w[:, None]
                sub_ranks = run_mcdm_suite(sub_weighted, sub_w, methods, get_spotis_bounds(l3_cats, sub_w))
                psi_combo = calc_psi(sub_ranks, methods, combo_p)
                rank_grid[:, ci] = rank_with_ties(psi_combo, ascending=False)

        fig2 = go.Figure()
        for pi, name in enumerate(names):
            fig2.add_trace(go.Scatter(
                x=list(range(len(combos))), y=rank_grid[pi, :],
                mode="markers", name=name,
                marker=dict(size=11, color=PROC_COLORS[pi % len(PROC_COLORS)],
                            symbol="square", line=dict(width=0)),
            ))
        fig2.update_layout(
            xaxis=dict(
                tickmode="array", tickvals=list(range(len(combos))), ticktext=combo_labels,
                tickangle=0,
                title="Category combination",
                tickfont=dict(size=10),
                automargin=True,
            ),
            yaxis=dict(
                title="Rank", autorange="reversed",
                dtick=1, tick0=1,
                range=[0.5, n_proc + 0.5],
            ),
            height=420,
            margin=dict(l=10, r=10, t=10, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

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
        if st.button("Auxiliary assessment ->", type="primary"):
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
    st.caption(
        "Temporarily exclude one or more alternatives and re-run the full PRISM pipeline "
        "(MEREC → N2 → category scores → category weighting → RCW → MCDM suite → PSI). "
        "If the relative ranking of the remaining alternatives is unchanged, the result "
        "is robust. Any position change is flagged as a rank-reversal event."
    )

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
        st.info("Select at least one alternative above to run the test.")
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
    ranks_pert = run_mcdm_suite(weighted_mat_pert, w_cat_pert, sel_mcdm_methods, get_spotis_bounds(l3_cats, w_cat_pert))

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
        st.warning(
            "⚠️ Rank-reversal detected: one or more alternatives changed position after "
            "exclusion. This indicates sensitivity to the composition of the alternative set."
        )
    else:
        st.success(
            "✅ No rank-reversal detected: the relative ranking of the remaining alternatives "
            "is identical to their baseline positions. The result is robust to the exclusion "
            f"of {', '.join(excluded)}."
        )

    st.caption(
        "Baseline ranks are the relative positions of the kept alternatives within the "
        "full original results. Reduced ranks are computed by re-running the entire "
        "PRISM pipeline (MEREC → N2 → category scores → weighting → MCDM → PSI) "
        "on the reduced alternative set only."
    )



def validation_ranking_correlation():
    st.subheader("5. Ranking correlation (Spearman vs Weighted Spearman)")
    st.caption(
        "Compares how consistently each pair of MCDM methods ranks your alternatives, "
        "using two complementary coefficients. Standard Spearman treats all rank "
        "disagreements equally. Weighted Spearman (r_w) penalises disagreements at the "
        "top of the ranking more heavily — directly relevant when the identity of the "
        "top-ranked alternative is the primary decision output."
    )

    method_ranks = st.session_state.get("last_method_ranks", {})
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    names = st.session_state.proc_names
    n = len(names)

    # Filter to methods that actually have results
    available = [m for m in sel_mcdm_methods if m in method_ranks]
    if len(available) < 2:
        st.warning("At least 2 MCDM methods are needed to compute ranking correlation.")
        return

    labels = [METHOD_LABELS[m] for m in available]

    def weighted_spearman(r1, r2):
        r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
        n_ = len(r1)
        r_avg = (r1 + r2) / 2.0
        w = n_ - r_avg + 1.0
        d2 = (r1 - r2) ** 2
        denom = n_ ** 4 + n_ ** 3 - n_ ** 2 - n_
        return 1.0 - 6.0 * np.sum(d2 * w) / denom

    k = len(available)
    spearman_mat = np.ones((k, k))
    ws_mat = np.ones((k, k))

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            r1 = np.array([method_ranks[available[i]][pi] for pi in range(n)], dtype=float)
            r2 = np.array([method_ranks[available[j]][pi] for pi in range(n)], dtype=float)
            rho, _ = spearmanr(r1, r2)
            spearman_mat[i, j] = round(float(rho), 4)
            ws_mat[i, j] = round(weighted_spearman(r1, r2), 4)

    df_sp = pd.DataFrame(spearman_mat, index=labels, columns=labels).round(4)
    df_ws = pd.DataFrame(ws_mat, index=labels, columns=labels).round(4)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Standard Spearman (ρ)**")
        st.caption("Treats all rank position disagreements equally.")
        st.dataframe(df_sp, use_container_width=True)
    with c2:
        st.markdown("**Weighted Spearman (r_w)**")
        st.caption("Top-rank disagreements penalised more heavily.")
        st.dataframe(df_ws, use_container_width=True)

    # Interpretation
    st.divider()
    st.markdown("**Interpretation**")

    diffs = []
    for i in range(k):
        for j in range(i + 1, k):
            sp = spearman_mat[i, j]
            ws = ws_mat[i, j]
            diffs.append((labels[i], labels[j], sp, ws, ws - sp))

    if diffs:
        rows_interp = []
        for la, lb, sp, ws, delta in diffs:
            if delta > 0.01:
                note = "r_w > ρ: top-rank agreement is stronger than overall agreement"
            elif delta < -0.01:
                note = "r_w < ρ: methods diverge more at the top than overall"
            else:
                note = "r_w ≈ ρ: rank disagreement is evenly distributed across positions"
            rows_interp.append({
                "Method pair": f"{la} vs {lb}",
                "ρ (Spearman)": round(sp, 4),
                "r_w (Weighted)": round(ws, 4),
                "Δ (r_w − ρ)": round(delta, 4),
                "Note": note,
            })
        st.dataframe(pd.DataFrame(rows_interp), use_container_width=True, hide_index=True)

    st.caption(
        "A positive Δ means methods agree more on who ranks first than on the full "
        "ordering — a favourable result when the top-ranked process is the primary "
        "decision output. Values close to 1.0 in both matrices indicate high method "
        "consensus and strengthen the PSI composite ranking."
    )



def validation_normalisation_sensitivity():
    st.subheader("6. Normalisation sensitivity")
    st.caption(
        "Tests whether the final ranking is sensitive to the choice of normalisation "
        "method used to compute category scores. MEREC-norm and MEREC indicator weights "
        "are held fixed throughout — only the normalisation applied to produce the "
        "category score (currently N2) is swapped. The full pipeline then reruns: "
        "category scores → category weighting → RCW → MCDM suite → PSI."
    )

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
    ranks_alt = run_mcdm_suite(weighted_mat_alt, w_cat_alt, sel_mcdm_methods, get_spotis_bounds(l3_cats, w_cat_alt))

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
        st.success(
            f"✅ No rank change: the ranking is identical under both N2 and {alt_norm_label} "
            "normalisation. This confirms robustness to normalisation method choice."
        )

    st.caption(
        "MEREC-norm (used to compute indicator weights) is held fixed in all cases — "
        "only the normalisation that produces the category score is varied. "
        "Min-max scales relative to the best/worst value in the set. "
        "Vector norm divides by the Euclidean length of each indicator column."
    )



def validation_intro():
    st.header("Step 13 - Validation (optional)")
    st.caption("Six optional checks to stress-test how robust your ranking is.")

    choice = st.radio(
        "Choose a validation method",
        [
            "None - skip validation",
            "1. Weighting-method sensitivity",
            "2. Benefit/Cost indicator sensitivity",
            "3. Monte Carlo uncertainty (Dirichlet)",
            "4. Rank-reversal test",
            "5. Ranking correlation (Spearman vs Weighted Spearman)",
            "6. Normalisation sensitivity",
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
        validation_ranking_correlation()
    elif choice.startswith("6."):
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
    st.caption(
        "MEREC weights (indicator level) are left unchanged. Only the Level 3 "
        "category-weighting method varies across all 6 combinations: Equal, Entropy, "
        "CRITIC, and the three pairwise RCW consolidations. For each, all 5 MCDM "
        "methods are run and ranks are shown - no PSI, ranks only."
    )

    names = st.session_state.proc_names
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    mat = np.array([cat_scores[c] for c in l3_cats])

    rows = [{"Process": name} for name in names]
    for combo_methods, combo_label in WEIGHT_COMBO_SETS:
        w = get_category_weights(mat, set(combo_methods))
        weighted_mat = mat * w[:, None]
        ranks = run_mcdm_suite(weighted_mat, w, ALL_MCDM_KEYS, get_spotis_bounds(l3_cats, w))
        for pi in range(len(names)):
            for mkey in ALL_MCDM_KEYS:
                col = f"{combo_label} - {METHOD_LABELS[mkey]}"
                rows[pi][col] = int(ranks[mkey][pi])

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "Read across each process's row: if ranks stay consistent across all 30 "
        "columns (6 weighting schemes x 5 MCDM methods), the result is robust to "
        "the choice of weighting method."
    )

    st.divider()
    st.markdown("**Overall ranking across category combinations (this weighting scheme set)**")
    st.caption(
        "Same 2^n - 1 combination logic as Step 12, but every combination here uses "
        "whichever weighting method(s) you toggle below, re-run through all 5 MCDM "
        "methods and PSI at the chosen p. Single-category columns remain fixed (Step 8)."
    )

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
    cat_initial = {c: CATS[c]["label"][:3] for c in l3_cats}
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
            sub_ranks = run_mcdm_suite(sub_weighted, sub_w, ALL_MCDM_KEYS, get_spotis_bounds(l3_cats, sub_w))
            psi_combo = calc_psi(sub_ranks, ALL_MCDM_KEYS, sens1_p)
            rank_grid[:, ci] = rank_with_ties(psi_combo, ascending=False)

    fig = go.Figure()
    for pi, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=list(range(len(combos))), y=rank_grid[pi, :],
            mode="markers", name=name,
            marker=dict(size=11, color=PROC_COLORS[pi % len(PROC_COLORS)],
                        symbol="square", line=dict(width=0)),
        ))
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(len(combos))), ticktext=combo_labels,
                   tickangle=0, title="Category combination", tickfont=dict(size=10),
                   automargin=True),
        yaxis=dict(title="Rank", autorange="reversed", dtick=1, tick0=1, range=[0.5, n_proc + 0.5]),
        height=420, margin=dict(l=10, r=10, t=10, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def validation_bc_sensitivity():
    st.subheader("2. Benefit/Cost indicator sensitivity")
    st.caption(
        "All benefit-type indicators are perturbed by one %, and all cost-type "
        "indicators by another % - applied simultaneously, across all processes, "
        "within your selected Level 3 categories. The full pipeline (MEREC -> N2 -> "
        "category score -> weighting -> RCW -> 5 MCDM -> PSI) re-runs at a chosen p."
    )

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
    ranks = run_mcdm_suite(weighted_mat, w, sel_mcdm_methods, get_spotis_bounds(l3_cats, w))

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
    st.caption(
        "Compare to the baseline above. This applies uniformly across every process "
        "and every selected category, simulating systematic measurement bias or "
        "optimistic/pessimistic forecasting."
    )


def validation_monte_carlo():
    st.subheader("3. Monte Carlo uncertainty (Dirichlet)")
    st.caption(
        "10,000 Dirichlet-distributed draws are generated around the RCW category "
        "weights from Step 10. The concentration parameter k is data-driven, not "
        "user-set: computed from how much Equal, Entropy, and CRITIC weighting "
        "methods agree with each other. High agreement gives high k (tight sampling). "
        "High disagreement gives low k (wide sampling)."
    )

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

    with st.expander("Show Equal / Entropy / CRITIC weights used to compute k"):
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
            ranks_draw = run_mcdm_suite(weighted_mat, w_draw, sel_mcdm_methods, get_spotis_bounds(l3_cats, w_draw))
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

        st.caption(
            "Each cell shows the percentage of the 10,000 simulated weight draws in "
            "which that process landed at that rank. A process with most of its "
            "probability mass concentrated in one rank column is a stable result; "
            "spread across multiple columns indicates sensitivity to weighting uncertainty."
        )


# ============================================================================
# STEP 14 - AUXILIARY ASSESSMENT (optional)
# ============================================================================

def auxiliary_contribution_decomposition():
    st.subheader("1. Indicator & category contribution decomposition")
    st.caption(
        "Breaks down each process's PSI score to show exactly how much each category "
        "— and within that, each indicator — contributed to the final ranking. "
        "Answers the question: why does one process outperform another?"
    )

    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    cat_scores = st.session_state.cat_scores
    final_w = st.session_state.final_cat_weights
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    merec_w = st.session_state.merec_w
    n2_data = st.session_state.n2_data

    p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                       value=0.5, step=0.01, key="aux_decomp_p")

    psi_vals = calc_psi(method_ranks, sel_mcdm_methods, p_val)

    # ---- CATEGORY-LEVEL CONTRIBUTION ----
    st.divider()
    st.markdown("#### Category-level contribution")
    st.caption(
        "Each category's contribution to a process's weighted score = "
        "category score × category weight. Shown as absolute value and % of total."
    )

    cat_labels = [CATS[c]["label"] for c in l3_cats]
    cat_contribs = np.array([cat_scores[c] * final_w[i]
                              for i, c in enumerate(l3_cats)])  # shape: (n_cats, n_proc)

    # Stacked bar chart — one bar per process
    fig_cat = go.Figure()
    for ci, ckey in enumerate(l3_cats):
        fig_cat.add_trace(go.Bar(
            name=CATS[ckey]["label"],
            x=names,
            y=cat_contribs[ci],
            marker_color=CATS[ckey]["color"],
            text=[f"{v:.4f}" for v in cat_contribs[ci]],
            textposition="inside",
        ))
    fig_cat.update_layout(
        barmode="stack", height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Weighted category score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Percentage breakdown table
    cat_totals = cat_contribs.sum(axis=0)
    cat_pct = np.where(cat_totals > 0, cat_contribs / cat_totals * 100, 0.0)
    rows_cat = []
    for pi, name in enumerate(names):
        row = {"Process": name}
        for ci, ckey in enumerate(l3_cats):
            row[f"{CATS[ckey]['label']} (%)"] = round(cat_pct[ci, pi], 1)
        rows_cat.append(row)
    st.dataframe(pd.DataFrame(rows_cat), use_container_width=True, hide_index=True)

    # ---- INDICATOR-LEVEL CONTRIBUTION ----
    st.divider()
    st.markdown("#### Indicator-level contribution")
    st.caption(
        "Within each category: indicator contribution = N2-normalised value × MEREC weight × "
        "category weight. Shows which specific indicators drive each process's advantage."
    )

    process_sel = st.selectbox(
        "Select process to inspect", names, key="aux_decomp_proc"
    )
    pi = names.index(process_sel)

    for ci, ckey in enumerate(l3_cats):
        cat = CATS[ckey]
        ind_names, ind_units, _ = get_full_indicators(ckey)
        n2 = n2_data[ckey]          # shape: (n_ind, n_proc)
        w_ind = merec_w[ckey]       # shape: (n_ind,)
        w_cat = final_w[ci]

        ind_contribs = n2[:, pi] * w_ind * w_cat   # contribution of each indicator

        st.markdown(
            f"<span style='background:{cat['bg']};color:{cat['color']};"
            f"padding:2px 8px;border-radius:10px;font-size:13px;font-weight:600;'>"
            f"{cat['label']}</span>", unsafe_allow_html=True,
        )

        total_ind = ind_contribs.sum() or 1.0
        rows_ind = []
        for j, iname in enumerate(ind_names):
            rows_ind.append({
                "Indicator": f"{iname} ({ind_units[j]})",
                "MEREC weight": round(w_ind[j], 4),
                "N2 value": round(n2[j, pi], 4),
                "Contribution": round(ind_contribs[j], 6),
                "% of category": round(ind_contribs[j] / total_ind * 100, 1),
            })

        df_ind = pd.DataFrame(rows_ind)
        st.dataframe(df_ind, use_container_width=True, hide_index=True)

        # Mini bar chart
        fig_ind = go.Figure(go.Bar(
            x=[r["Indicator"] for r in rows_ind],
            y=[r["Contribution"] for r in rows_ind],
            marker_color=cat["color"],
            text=[f"{r['% of category']:.1f}%" for r in rows_ind],
            textposition="outside",
        ))
        fig_ind.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=60),
            xaxis_tickangle=-30, yaxis_title="Contribution",
            showlegend=False,
        )
        st.plotly_chart(fig_ind, use_container_width=True)
        st.write("")

    # ---- RANKING GAP DECOMPOSITION ----
    st.divider()
    st.markdown("#### Ranking gap between processes")
    st.caption(
        "For each pair of processes, shows how much each category contributes to "
        "the weighted score gap — positive means the first process leads in that category."
    )

    if n_proc < 2:
        st.info("Need at least 2 processes for gap decomposition.")
        return

    proc_a = st.selectbox("Process A", names, index=0, key="gap_proc_a")
    proc_b = st.selectbox("Process B", names,
                           index=min(1, n_proc - 1), key="gap_proc_b")

    if proc_a == proc_b:
        st.info("Select two different processes.")
        return

    pa, pb = names.index(proc_a), names.index(proc_b)
    gap_rows = []
    for ci, ckey in enumerate(l3_cats):
        gap = cat_contribs[ci, pa] - cat_contribs[ci, pb]
        gap_rows.append({
            "Category": CATS[ckey]["label"],
            f"{proc_a} contribution": round(cat_contribs[ci, pa], 4),
            f"{proc_b} contribution": round(cat_contribs[ci, pb], 4),
            f"Gap (A − B)": round(gap, 4),
            "Favours": proc_a if gap > 0 else (proc_b if gap < 0 else "Tied"),
        })

    df_gap = pd.DataFrame(gap_rows)
    st.dataframe(df_gap, use_container_width=True, hide_index=True)

    gaps = [r["Gap (A − B)"] for r in gap_rows]
    colours = [CATS[ckey]["color"] if g >= 0 else "#94a3b8"
               for g, ckey in zip(gaps, l3_cats)]
    fig_gap = go.Figure(go.Bar(
        x=[r["Category"] for r in gap_rows],
        y=gaps,
        marker_color=colours,
        text=[f"{g:+.4f}" for g in gaps],
        textposition="outside",
    ))
    fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_gap.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=40),
        yaxis_title=f"Score gap ({proc_a} − {proc_b})",
        xaxis_title="Category", showlegend=False,
    )
    st.plotly_chart(fig_gap, use_container_width=True)


def auxiliary_improvement_simulator():
    st.subheader("2. Process improvement simulator")
    st.caption(
        "Flip the question: instead of asking what the ranking is, ask how much a "
        "specific process needs to improve on a given indicator to reach a target rank. "
        "The full PRISM pipeline reruns at each step to find the minimum improvement required."
    )

    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})

    if n_proc < 2:
        st.warning("Need at least 2 processes for the improvement simulator.")
        return

    multi = len(sel_mcdm_methods) > 1

    c1, c2, c3 = st.columns(3)
    with c1:
        target_proc = st.selectbox("Process to improve", names, key="sim_proc")
    with c2:
        target_rank = st.selectbox("Target PSI rank", list(range(1, n_proc)),
                                    key="sim_rank")
    with c3:
        p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                           value=0.5, step=0.01, key="sim_p_slider")

    pi_target = names.index(target_proc)

    # Current PSI rank
    if multi and method_ranks:
        current_psi = calc_psi(method_ranks, sel_mcdm_methods, p_val)
        current_rank = int(rank_with_ties(current_psi, ascending=False)[pi_target])
    else:
        current_rank = "N/A (need >1 MCDM method)"

    st.info(f"**{target_proc}** current PSI rank: **{current_rank}**  →  target rank: **{target_rank}**")

    if isinstance(current_rank, int) and current_rank <= target_rank:
        st.success(f"{target_proc} already achieves rank {target_rank} or better.")
        return

    # Build list of all indicators across all l3 categories
    all_indicators = []
    for ckey in l3_cats:
        ind_names, ind_units, benefits = get_full_indicators(ckey)
        for j, (iname, iunit) in enumerate(zip(ind_names, ind_units)):
            all_indicators.append({
                "ckey": ckey, "j": j,
                "label": f"{CATS[ckey]['label']} — {iname} ({iunit})",
                "benefit": benefits[j],
            })

    sel_ind_label = st.selectbox(
        "Indicator to improve", [ind["label"] for ind in all_indicators],
        key="sim_indicator"
    )
    sel_ind = next(ind for ind in all_indicators if ind["label"] == sel_ind_label)

    def run_pipeline_with_override(ckey_override, j_override, new_val):
        """Rerun full pipeline with one indicator value overridden for target process."""
        cat_scores_sim = {}
        for ckey in l3_cats:
            ind_names, ind_units, benefits = get_full_indicators(ckey)
            n_ind = len(ind_names)
            raw, _ = get_raw_matrix(ckey)
            raw_mod = raw.copy()
            if ckey == ckey_override:
                raw_mod[j_override, pi_target] = new_val
            nm = np.zeros_like(raw_mod)
            n2 = np.zeros_like(raw_mod)
            for j in range(n_ind):
                nm[j] = merec_norm(raw_mod[j], benefits[j])
                n2[j] = n2_norm(raw_mod[j], benefits[j])
            w_ind = merec_weights(nm)
            cat_scores_sim[ckey] = (n2 * w_ind[:, None]).sum(axis=0)

        mat_sim = np.array([cat_scores_sim[c] for c in l3_cats])
        w_cat_sim = get_category_weights(mat_sim, sel_weight_methods)
        wm_sim = mat_sim * w_cat_sim[:, None]
        ranks_sim = run_mcdm_suite(wm_sim, w_cat_sim, sel_mcdm_methods, get_spotis_bounds(l3_cats, w_cat_sim))
        psi_sim = calc_psi(ranks_sim, sel_mcdm_methods, p_val)
        return int(rank_with_ties(psi_sim, ascending=False)[pi_target])

    # Get current raw value for selected indicator
    ckey_s, j_s = sel_ind["ckey"], sel_ind["j"]
    current_val = st.session_state.indicator_values.get((ckey_s, j_s, pi_target), 0.0)
    is_benefit = sel_ind["benefit"]

    st.markdown(f"**Current value:** {current_val:.4g}  |  "
                f"**Direction:** {'Benefit (higher = better)' if is_benefit else 'Cost (lower = better)'}")

    if st.button("Run improvement search", type="primary", key="sim_run"):
        # Search: improve in steps of 1% up to 200% change (100 steps)
        steps = 200
        results = []
        progress = st.progress(0, text="Searching for minimum improvement...")

        for step_i in range(1, steps + 1):
            pct = step_i * 1.0
            if is_benefit:
                new_val = current_val * (1 + pct / 100.0)
            else:
                new_val = current_val * (1 - pct / 100.0)
                if new_val <= 0:
                    new_val = current_val * 0.001

            new_rank = run_pipeline_with_override(ckey_s, j_s, new_val)
            results.append({"pct": pct, "new_val": new_val, "rank": new_rank})

            if new_rank <= target_rank:
                progress.progress(1.0, text="Target rank achieved.")
                break
            progress.progress(step_i / steps,
                               text=f"Step {step_i}/{steps} — current rank: {new_rank}")

        st.session_state.sim_results = results
        st.session_state.sim_target_rank = target_rank
        st.session_state.sim_current_val = current_val
        st.session_state.sim_proc_name = target_proc
        st.session_state.sim_ind_label = sel_ind_label
        st.session_state.sim_is_benefit = is_benefit

    if "sim_results" in st.session_state:
        results = st.session_state.sim_results
        target_r = st.session_state.sim_target_rank

        # Find first step where target is achieved
        achieved = next((r for r in results if r["rank"] <= target_r), None)

        if achieved:
            delta_pct = achieved["pct"]
            delta_abs = achieved["new_val"] - st.session_state.sim_current_val
            st.success(
                f"✅ **{st.session_state.sim_proc_name}** reaches rank **{target_r}** "
                f"with a **{delta_pct:.1f}% {'increase' if st.session_state.sim_is_benefit else 'reduction'}** "
                f"in **{st.session_state.sim_ind_label.split('—')[-1].strip()}**  "
                f"(new value: {achieved['new_val']:.4g}, "
                f"change: {delta_abs:+.4g})"
            )
        else:
            st.warning(
                f"⚠️ Even a 200% {'increase' if st.session_state.sim_is_benefit else 'reduction'} "
                f"in {st.session_state.sim_ind_label.split('—')[-1].strip()} "
                f"is not sufficient to reach rank {target_r}. "
                "Try a different indicator."
            )

        # Rank trajectory chart
        pcts = [r["pct"] for r in results]
        ranks_traj = [r["rank"] for r in results]

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=pcts, y=ranks_traj, mode="lines+markers",
            line=dict(color=PROC_COLORS[names.index(st.session_state.sim_proc_name)
                                        % len(PROC_COLORS)], width=2),
            marker=dict(size=5),
            name="PSI rank",
        ))
        fig_sim.add_hline(y=target_r, line_dash="dash", line_color="green",
                           annotation_text=f"Target rank {target_r}")
        fig_sim.update_layout(
            xaxis_title="Improvement (%)",
            yaxis_title="PSI rank",
            yaxis=dict(autorange="reversed", dtick=1, tick0=1,
                       range=[0.5, n_proc + 0.5]),
            height=320, margin=dict(l=10, r=10, t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        st.caption(
            "X-axis shows the percentage improvement applied to the selected indicator "
            "for the target process only. Y-axis shows the resulting PSI rank. "
            "The green dashed line marks the target rank."
        )


def auxiliary_multi_indicator_optimiser():
    st.subheader("3. Multi-indicator improvement optimiser")
    st.caption(
        "Given a total improvement budget spread across multiple indicators, finds the "
        "optimal allocation that maximises PSI score or achieves a target rank. "
        "Answers: given limited engineering effort, where should it be focused?"
    )

    names = st.session_state.proc_names
    n_proc = len(names)
    l3_cats = ordered_l3_cats()
    sel_weight_methods = st.session_state.sel_weight_methods
    sel_mcdm_methods = list(st.session_state.sel_mcdm_methods) or ALL_MCDM_KEYS
    method_ranks = st.session_state.get("last_method_ranks", {})
    multi = len(sel_mcdm_methods) > 1

    if n_proc < 2:
        st.warning("Need at least 2 processes for the optimiser.")
        return
    if not multi:
        st.warning("Need at least 2 MCDM methods selected for PSI-based optimisation.")
        return

    # ── User inputs ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        target_proc = st.selectbox("Process to improve", names, key="opt_proc")
    with c2:
        target_rank = st.selectbox("Target PSI rank",
                                    list(range(1, n_proc)),
                                    key="opt_rank")
    with c3:
        p_val = st.slider("p value (PSI)", min_value=0.0, max_value=1.0,
                           value=0.5, step=0.01, key="opt_p_slider")

    pi_target = names.index(target_proc)

    budget_pct = st.slider(
        "Total improvement budget (% across all selected indicators)",
        min_value=10, max_value=200, value=60, step=5, key="opt_budget"
    )

    strategy = st.radio(
        "Optimisation strategy",
        ["Greedy (fast — allocates to highest-gain indicator first)",
         "Exhaustive (thorough — tries all combinations at 5% steps)"],
        index=0, key="opt_strategy", horizontal=True,
    )

    # ── Build indicator list ──────────────────────────────────────────────────
    all_indicators = []
    for ckey in l3_cats:
        ind_names, ind_units, benefits = get_full_indicators(ckey)
        for j, (iname, iunit) in enumerate(zip(ind_names, ind_units)):
            all_indicators.append({
                "ckey": ckey, "j": j,
                "label": f"{CATS[ckey]['label']} — {iname} ({iunit})",
                "benefit": benefits[j],
            })

    sel_labels = st.multiselect(
        "Select improvable indicators (only indicators you can realistically change)",
        options=[ind["label"] for ind in all_indicators],
        default=[ind["label"] for ind in all_indicators],
        key="opt_indicators",
    )

    if not sel_labels:
        st.warning("Select at least one improvable indicator.")
        return

    sel_inds = [ind for ind in all_indicators if ind["label"] in sel_labels]

    # ── Current PSI rank ─────────────────────────────────────────────────────
    current_psi = calc_psi(method_ranks, sel_mcdm_methods, p_val)
    current_rank = int(rank_with_ties(current_psi, ascending=False)[pi_target])
    current_psi_val = float(current_psi[pi_target])

    st.info(
        f"**{target_proc}** current PSI rank: **{current_rank}**  "
        f"(PSI = {current_psi_val:.4f})  →  target rank: **{target_rank}**  |  "
        f"budget: **{budget_pct}%**"
    )

    if current_rank <= target_rank:
        st.success(f"{target_proc} already achieves rank {target_rank} or better.")
        return

    # ── Pipeline runner with multiple overrides ───────────────────────────────
    def run_pipeline_multi(overrides):
        """overrides: dict {(ckey, j): new_val} for target process."""
        cat_scores_opt = {}
        for ckey in l3_cats:
            ind_names_, ind_units_, benefits_ = get_full_indicators(ckey)
            n_ind_ = len(ind_names_)
            raw, _ = get_raw_matrix(ckey)
            raw_mod = raw.copy()
            for (ck, jj), new_val in overrides.items():
                if ck == ckey:
                    raw_mod[jj, pi_target] = new_val
            nm = np.zeros_like(raw_mod)
            n2 = np.zeros_like(raw_mod)
            for j in range(n_ind_):
                nm[j] = merec_norm(raw_mod[j], benefits_[j])
                n2[j] = n2_norm(raw_mod[j], benefits_[j])
            w_ind = merec_weights(nm)
            cat_scores_opt[ckey] = (n2 * w_ind[:, None]).sum(axis=0)

        mat_ = np.array([cat_scores_opt[c] for c in l3_cats])
        w_cat_ = get_category_weights(mat_, sel_weight_methods)
        wm_ = mat_ * w_cat_[:, None]
        ranks_ = run_mcdm_suite(wm_, w_cat_, sel_mcdm_methods,
                                 get_spotis_bounds(l3_cats, w_cat_))
        psi_ = calc_psi(ranks_, sel_mcdm_methods, p_val)
        return float(psi_[pi_target]), int(rank_with_ties(psi_, ascending=False)[pi_target])

    def apply_pct(ind, pct):
        """Return new indicator value after applying pct% improvement."""
        cur = st.session_state.indicator_values.get(
            (ind["ckey"], ind["j"], pi_target), 0.0)
        if ind["benefit"]:
            return cur * (1 + pct / 100.0)
        else:
            new = cur * (1 - pct / 100.0)
            return max(new, cur * 0.001)

    if st.button("Run optimisation", type="primary", key="opt_run"):
        allocation = {ind["label"]: 0.0 for ind in sel_inds}
        budget_remaining = float(budget_pct)
        trajectory = []  # (budget_used, psi_val, rank, allocation_snapshot)
        achieved = False

        progress = st.progress(0, text="Running optimisation...")

        if strategy.startswith("Greedy"):
            step_size = 5.0
            max_steps = int(budget_pct / step_size) + 1

            for step_i in range(max_steps):
                if budget_remaining < step_size:
                    break

                best_gain = -1.0
                best_ind = None

                # Try adding step_size% to each indicator
                for ind in sel_inds:
                    trial_alloc = dict(allocation)
                    trial_alloc[ind["label"]] += step_size
                    overrides = {
                        (ind2["ckey"], ind2["j"]): apply_pct(ind2, trial_alloc[ind2["label"]])
                        for ind2 in sel_inds if trial_alloc[ind2["label"]] > 0
                    }
                    psi_trial, _ = run_pipeline_multi(overrides)
                    gain = psi_trial - current_psi_val
                    if gain > best_gain:
                        best_gain = gain
                        best_ind = ind

                if best_ind is None:
                    break

                allocation[best_ind["label"]] += step_size
                budget_remaining -= step_size

                overrides_cur = {
                    (ind2["ckey"], ind2["j"]): apply_pct(ind2, allocation[ind2["label"]])
                    for ind2 in sel_inds if allocation[ind2["label"]] > 0
                }
                psi_cur, rank_cur = run_pipeline_multi(overrides_cur)
                budget_used = budget_pct - budget_remaining
                trajectory.append({
                    "budget_used": budget_used,
                    "psi": psi_cur,
                    "rank": rank_cur,
                    "allocation": dict(allocation),
                })

                progress.progress(
                    min(1.0, (step_i + 1) / max_steps),
                    text=f"Step {step_i+1} — PSI: {psi_cur:.4f}, rank: {rank_cur}"
                )

                if rank_cur <= target_rank:
                    achieved = True
                    break

        else:  # Exhaustive
            step_size = 5.0
            n_ind = len(sel_inds)
            max_per_ind = int(budget_pct / step_size)
            best_psi = current_psi_val
            best_alloc = {ind["label"]: 0.0 for ind in sel_inds}
            best_rank = current_rank
            total_combos = (max_per_ind + 1) ** n_ind
            combo_count = 0

            import itertools
            steps_per_ind = list(range(0, int(budget_pct) + 1, int(step_size)))

            for combo in itertools.product(steps_per_ind, repeat=n_ind):
                if sum(combo) > budget_pct:
                    combo_count += 1
                    continue
                alloc = {sel_inds[i]["label"]: combo[i] for i in range(n_ind)}
                overrides = {
                    (sel_inds[i]["ckey"], sel_inds[i]["j"]): apply_pct(sel_inds[i], combo[i])
                    for i in range(n_ind) if combo[i] > 0
                }
                psi_t, rank_t = run_pipeline_multi(overrides)
                if psi_t > best_psi:
                    best_psi = psi_t
                    best_alloc = dict(alloc)
                    best_rank = rank_t
                combo_count += 1
                if combo_count % max(1, total_combos // 20) == 0:
                    progress.progress(
                        min(1.0, combo_count / total_combos),
                        text=f"Evaluating combinations... {combo_count}/{total_combos}"
                    )
                if rank_t <= target_rank:
                    achieved = True
                    break

            allocation = best_alloc
            overrides_final = {
                (sel_inds[i]["ckey"], sel_inds[i]["j"]): apply_pct(sel_inds[i], allocation[sel_inds[i]["label"]])
                for i in range(n_ind) if allocation[sel_inds[i]["label"]] > 0
            }
            psi_final, rank_final = run_pipeline_multi(overrides_final)
            trajectory = [{
                "budget_used": sum(allocation.values()),
                "psi": psi_final,
                "rank": rank_final,
                "allocation": dict(allocation),
            }]
            if rank_final <= target_rank:
                achieved = True

        progress.progress(1.0, text="Done.")

        st.session_state.opt_results = {
            "trajectory": trajectory,
            "allocation": allocation,
            "achieved": achieved,
            "target_rank": target_rank,
            "target_proc": target_proc,
            "budget_pct": budget_pct,
            "p_val": p_val,
            "current_psi": current_psi_val,
            "current_rank": current_rank,
            "sel_inds": sel_inds,
        }

    # ── Display results ───────────────────────────────────────────────────────
    if "opt_results" in st.session_state:
        res = st.session_state.opt_results
        traj = res["trajectory"]
        allocation = res["allocation"]
        achieved = res["achieved"]

        if not traj:
            st.warning("No improvement found. Try increasing the budget or selecting more indicators.")
            return

        final = traj[-1]
        budget_used = final["budget_used"]
        final_psi = final["psi"]
        final_rank = final["rank"]

        st.divider()

        if achieved:
            st.success(
                f"✅ **{res['target_proc']}** reaches rank **{res['target_rank']}** "
                f"using **{budget_used:.0f}%** of the {res['budget_pct']}% budget  "
                f"(PSI: {res['current_psi']:.4f} → {final_psi:.4f})"
            )
        else:
            st.warning(
                f"⚠️ Target rank {res['target_rank']} not achieved within {res['budget_pct']}% budget. "
                f"Best result: rank **{final_rank}** (PSI: {final_psi:.4f}) "
                f"using full {budget_used:.0f}% budget. Try increasing the budget."
            )

        # Allocation table
        st.markdown("**Optimal budget allocation**")
        alloc_rows = []
        for ind in res["sel_inds"]:
            lbl = ind["label"]
            pct_alloc = allocation.get(lbl, 0.0)
            cur_val = st.session_state.indicator_values.get(
                (ind["ckey"], ind["j"], pi_target), 0.0)
            new_val = apply_pct(ind, pct_alloc) if pct_alloc > 0 else cur_val
            direction = "↑ increase" if ind["benefit"] else "↓ reduce"
            alloc_rows.append({
                "Indicator": lbl.split(" — ")[-1],
                "Category": lbl.split(" — ")[0],
                "Allocated (%)": round(pct_alloc, 1),
                "Direction": direction if pct_alloc > 0 else "— unchanged",
                "Current value": round(cur_val, 4),
                "New value": round(new_val, 4),
            })

        alloc_rows.sort(key=lambda x: x["Allocated (%)"], reverse=True)
        df_alloc = pd.DataFrame(alloc_rows)
        st.dataframe(df_alloc, use_container_width=True, hide_index=True)

        # Budget allocation bar chart
        fig_alloc = go.Figure(go.Bar(
            x=[r["Indicator"] for r in alloc_rows],
            y=[r["Allocated (%)"] for r in alloc_rows],
            marker_color=[
                CATS[ind["ckey"]]["color"]
                for ind in sorted(res["sel_inds"],
                                   key=lambda x: allocation.get(x["label"], 0), reverse=True)
            ],
            text=[f"{r['Allocated (%)']:.0f}%" for r in alloc_rows],
            textposition="outside",
        ))
        fig_alloc.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=60),
            xaxis_tickangle=-30, yaxis_title="Budget allocated (%)",
            showlegend=False,
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

        # Rank trajectory (greedy only — exhaustive gives single point)
        if len(traj) > 1:
            st.markdown("**Rank trajectory as budget is allocated**")
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=[t["budget_used"] for t in traj],
                y=[t["rank"] for t in traj],
                mode="lines+markers",
                line=dict(color=PROC_COLORS[pi_target % len(PROC_COLORS)], width=2),
                marker=dict(size=6),
                name="PSI rank",
            ))
            fig_traj.add_hline(
                y=res["target_rank"], line_dash="dash", line_color="green",
                annotation_text=f"Target rank {res['target_rank']}",
            )
            fig_traj.update_layout(
                xaxis_title="Budget used (%)",
                yaxis_title="PSI rank",
                yaxis=dict(autorange="reversed", dtick=1, tick0=1,
                           range=[0.5, n_proc + 0.5]),
                height=300, margin=dict(l=10, r=10, t=30, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        st.caption(
            "Budget allocation is in 5% increments. Greedy strategy allocates each "
            "increment to the indicator that gives the highest PSI gain at that step. "
            "Exhaustive strategy evaluates all valid combinations and returns the "
            "globally optimal allocation within the budget. "
            "All other processes and their indicator values remain unchanged."
        )



def auxiliary_intro():
    st.header("Step 14 - Auxiliary Assessment (optional)")
    st.caption(
        "Three analytical tools that go beyond ranking to explain why processes perform "
        "as they do and what it would take to change the outcome."
    )

    choice = st.radio(
        "Choose a tool",
        [
            "None - skip",
            "1. Indicator & category contribution decomposition",
            "2. Process improvement simulator",
            "3. Multi-indicator improvement optimiser",
        ],
        index=0, key="auxiliary_radio",
    )

    if choice.startswith("1."):
        auxiliary_contribution_decomposition()
    elif choice.startswith("2."):
        auxiliary_improvement_simulator()
    elif choice.startswith("3."):
        auxiliary_multi_indicator_optimiser()

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
    1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6,
    7: step7, 8: step8, 9: step9, 10: step10, 11: step11, 12: step12,
    11.5: step11_5, 13: validation_intro, 14: auxiliary_intro,
}

STEPS[st.session_state.step]()
