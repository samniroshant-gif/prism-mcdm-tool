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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

st.set_page_config(page_title="PRISM - Sustainability MCDM Tool", page_icon="🧭", layout="wide")

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


MCDM_FUNCS = {
    "topsis": lambda wm, w: topsis(wm),
    "vikor": lambda wm, w: vikor(wm, w),
    "electre": lambda wm, w: electre1(wm, w),
    "multimoora": lambda wm, w: multimoora(wm),
    "waspas": lambda wm, w: waspas(wm),
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
    "11. MCDM methods", "12. Results", "13. Validation (optional)",
]

with st.sidebar:
    st.title("🧭 PRISM")
    st.caption("Sustainability MCDM Assessment Tool")
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

    icons = {"env": "🌿", "eco": "💰", "soc": "👥", "qua": "🏅", "pro": "⚡"}
    cols = st.columns(5)
    for i, key in enumerate(CATEGORY_ORDER):
        cat = CATS[key]
        with cols[i]:
            checked = key in st.session_state.sel_cats
            new_val = st.checkbox(f"{icons[key]} {cat['label']}", value=checked, key=f"catchk_{key}")
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

        st.dataframe(df_corr.style.applymap(highlight), use_container_width=True)

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
    method_ranks = run_mcdm_suite(weighted_mat, final_w, methods)
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
                sub_ranks = run_mcdm_suite(sub_weighted, sub_w, methods)
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
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("<- Back"):
            st.session_state.step = 11
            st.rerun()
    with c2:
        if st.button("Validation (optional) ->", type="primary"):
            st.session_state.step = 13
            st.rerun()
    with c3:
        if st.button("Reset all"):
            reset_all()
            st.rerun()


# ============================================================================
# STEP 13 - VALIDATION (optional)
# ============================================================================

def validation_intro():
    st.header("Step 13 - Validation (optional)")
    st.caption("Three optional checks to stress-test how robust your ranking is.")

    choice = st.radio(
        "Choose a validation method",
        [
            "None - skip validation",
            "1. Weighting-method sensitivity",
            "2. Benefit/Cost indicator sensitivity",
            "3. Monte Carlo uncertainty (Dirichlet)",
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
        ranks = run_mcdm_suite(weighted_mat, w, ALL_MCDM_KEYS)
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
            sub_ranks = run_mcdm_suite(sub_weighted, sub_w, ALL_MCDM_KEYS)
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

        st.caption(
            "Each cell shows the percentage of the 10,000 simulated weight draws in "
            "which that process landed at that rank. A process with most of its "
            "probability mass concentrated in one rank column is a stable result; "
            "spread across multiple columns indicates sensitivity to weighting uncertainty."
        )


STEPS = {
    1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6,
    7: step7, 8: step8, 9: step9, 10: step10, 11: step11, 12: step12,
    13: validation_intro,
}

STEPS[st.session_state.step]()
