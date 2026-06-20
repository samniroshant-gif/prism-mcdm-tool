# PRISM — Sustainability MCDM Assessment Tool

A Streamlit web app implementing your PRISM framework: MEREC weighting,
N2 normalisation, Equal/Entropy/CRITIC category weighting with RCW
consolidation, five MCDM methods (TOPSIS, VIKOR, ELECTRE I, MULTIMOORA,
WASPAS), and PSI compromise ranking with an interactive p-curve.

## Files
- `app.py` — the full application (single file)
- `requirements.txt` — Python packages needed

## How to run it on your own laptop

### Step 1 — Install Python (skip if you already have it)
Download Python 3.10+ from https://www.python.org/downloads/
During install on Windows, tick **"Add Python to PATH"**.

### Step 2 — Open a terminal
- **Windows**: search "Command Prompt" or "PowerShell" in the Start menu
- **Mac**: open "Terminal" from Applications → Utilities

### Step 3 — Navigate to the folder containing app.py
```bash
cd path/to/prism_app
```
(Replace with wherever you saved the folder, e.g. `cd Desktop/prism_app`)

### Step 4 — Install the required packages
```bash
pip install -r requirements.txt
```
If `pip` isn't recognised, try `pip3 install -r requirements.txt` or
`python -m pip install -r requirements.txt`.

### Step 5 — Run the app
```bash
streamlit run app.py
```
A browser tab will open automatically at `http://localhost:8501`.
If it doesn't open automatically, copy that URL into your browser.

### Step 6 — Stop the app
Go back to the terminal and press `Ctrl + C`.

## How to share it with others (no install needed for them)

The easiest free option is **Streamlit Community Cloud**:

1. Create a free GitHub account at https://github.com if you don't have one
2. Create a new repository and upload `app.py` and `requirements.txt`
3. Go to https://share.streamlit.io and sign in with GitHub
4. Click "New app", select your repository, and set the main file to `app.py`
5. Click "Deploy" — you'll get a public URL (e.g. `yourapp.streamlit.app`)
   that anyone can open in their browser, no installation required

## How the tool works (matches your framework exactly)

**Level 1 — System definition**
- Select number of processes (2–10) and name them
- Select assessment categories (Environmental, Economic, Social, Quality, Productivity)
- Optionally add custom indicators to any category (name, unit, benefit/cost direction)
- Select units per predefined indicator (preset dropdown or custom text)
- Enter indicator values per process

**Correlation check (new)**
- Spearman's rank correlation is computed between every pair of indicators within
  each category and displayed as a matrix
- Pairs with |rho| > 0.8 are highlighted and flagged with a warning — this is purely
  informational; no indicators are removed automatically
- Click "Accept and proceed" to continue to normalisation

**Level 2 — Indicator processing**
- MEREC normalisation (benefit = min/x, cost = x/max) → MEREC weights
- N2 linear sum-based normalisation (benefit = x/Σx, cost = (1/x)/Σ(1/x))
- Category score = Σ(N2 × MEREC weight), shown per process

**Level 3 — Decision aggregation**
- Select categories to carry forward
- Choose Equal / Entropy / CRITIC weighting (any combination); if more than
  one is chosen, RCW (Reciprocal Composite Weight) consolidation is applied
  automatically
- Choose MCDM methods (TOPSIS, VIKOR, ELECTRE I, MULTIMOORA, WASPAS); if
  more than one is chosen, PSI compromise ranking is computed with an
  adjustable p slider and a full PSI-vs-p curve
- **Category combination rank-stability chart**: for every non-empty subset of
  your selected Level 3 categories (2ⁿ−1 combinations), category weights are
  recomputed with your chosen weighting method(s) and PSI rank is computed at
  a p-value you control with a slider (0 to 1). Each process is shown as a
  coloured square dot at its rank for every combination — tight, unchanging
  positions as you move the slider mean a stable, trustworthy ranking;
  jumping dots mean the ranking is sensitive to which categories are included
  and to the performance/stability trade-off (p)

All formulas were verified line-by-line against your presentation slides
and worked example (CM / WAAM / SAM) before being translated into Python.

## Customising the indicators

If you ever need to change the indicator list, units, or benefit/cost
direction, edit the `CATS` dictionary near the top of `app.py`. Each
category entry has:
- `indicators`: list of indicator names
- `default_units` / `unit_options`: dropdown choices per indicator
- `benefit`: `True`/`False` per indicator (True = higher is better)
