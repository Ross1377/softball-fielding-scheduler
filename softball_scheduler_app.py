# softball_scheduler_app.py — Streamlit + PuLP (CBC) scheduler
# - 3 or 4 outfielders
# - up to 17 players
# - per-player priority positions (P1..P5)
# - benchable innings per player
# - hard cap: max 2 consecutive benches
# - soft penalty for back-to-back benches (dropdown weight)
# - NEW: "Avoid consecutive innings for:" (encourage rotation for selected positions only)
# - gentle reward to keep same position for all other positions
# - post-solve ARROW CONTROLS to reorder innings (◀ ▶) and lineup (▲ ▼)
#
# requirements.txt:
#   streamlit==1.37.1
#   pandas==2.2.3
#   numpy==2.1.3
#   pulp==2.8.0

from __future__ import annotations
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import pulp


# ---------------- Page & CSS (mobile-robust + dark) ----------------

st.set_page_config(
    page_title="Softball Fielding Schedule Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

st.markdown("""
<style>
/* Hide Streamlit's default header bar */
header[data-testid="stHeader"]{
  height: 0px !important;
  visibility: hidden;
}

/* Add a little top padding back to the content so nothing is cramped */
div.block-container{
  padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Keep the content from stretching too wide on big monitors,
# so the "Penalty" select isn't way off to the right.
st.markdown(
    """
    <style>
      .block-container { max-width: 1100px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navy/dark fallback + page-level horizontal scroll + hide header menu
st.markdown("""
<style>
:root{
  --bg:#0b1f3a;      /* navy */
  --bg2:#122a4e;     /* darker navy */
  --fg:#ffffff;      /* white text */
  --btn:#ff5a5a;     /* red button */
}

/* Always dark */
body, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--fg) !important; }
section.main .block-container{ background:var(--bg) !important; }

/* Hide Streamlit chrome (3-dot menu, footer, toolbar) */
#MainMenu, footer, [data-testid="stToolbar"] { display:none !important; }

/* PAGE is the ONLY horizontal scroller (iOS-friendly) */
section.main{
  overflow-x:auto !important;
  -webkit-overflow-scrolling:touch;
  overscroll-behavior-x:contain;
}

/* Wrapper grows to natural width; left-aligned; never smaller than viewport */
.page-canvas{
  display:inline-block;         /* shrink-to-fit */
  width:max-content;            /* natural width of content */
  min-width:100%;
  margin:0 !important;
  padding-left:max(12px, env(safe-area-inset-left));
  padding-right:max(12px, env(safe-area-inset-right));
}

/* Neutralize Streamlit's centered max-width that can hide edges */
section.main .block-container{ max-width:none !important; }

/* Make the editor/table adopt natural width (no nested scrollers) */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] div[role="grid"]{
  width:max-content !important;
  min-width:100%;
  overflow:visible !important;
}

/* High-contrast text for labels/inputs on dark bg */
label, .stMarkdown, .stText, .stCaption, .stRadio, .stSelectbox, .stNumberInput, .stDataFrame, .stTable {
  color:#fff !important; opacity:1 !important;
}
input, textarea, select, [data-baseweb="select"] *, .stSelectbox div, .stNumberInput input {
  color:#fff !important;
}

/* Primary button: red background w/ black text for contrast */
.stButton>button{
  background:var(--btn) !important;
  color:#000 !important;
  border:0 !important;
  font-weight:700 !important;
}

/* Compact arrow rows */
.arrow-row{ margin:.2rem 0; }
</style>
""", unsafe_allow_html=True)

# Wrap everything so CSS can measure natural width
st.markdown('<div class="page-canvas">', unsafe_allow_html=True)


# ---------------- Helpers ----------------

def positions_for(outfielders: int) -> List[str]:
    infield = ["P", "C", "1B", "2B", "3B", "SS"]
    of3 = ["LF", "CF", "RF"]
    of4 = ["LF", "LCF", "RCF", "RF"]
    return infield + (of3 if outfielders == 3 else of4)

# Lower is better; steeper costs favor higher priority
PRIO_COST = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10}


# ---------------- Solver (PuLP + CBC) ----------------

def build_model(
    players_data: List[Dict],
    pos_list: List[str],
    innings: int,
    bench_streak_weight: int,
    avoid_seq_positions: List[str],
) -> Tuple[pulp.LpProblem, Dict, Dict, Dict]:
    """
    players_data entry: {name, allowed:set[str], bench_max:int, prio_costs:{pos:int}}
    pos_list: list of field positions for this game (length = players on field each inning)
    avoid_seq_positions: positions for which we discourage consecutive SAME assignment (encourage rotation)
    """
    prob = pulp.LpProblem("softball_schedule", pulp.LpMinimize)
    P = range(len(players_data))
    I = range(innings)
    AVOID = set(avoid_seq_positions)

    # Decision vars
    x, y, b = {}, {}, {}  # x[p,i,pos], y[p,i]=play, b[p,i]=bench
    for p in P:
        for i in I:
            y[p, i] = pulp.LpVariable(f"y_{p}_{i}", 0, 1, cat="Binary")
            b[p, i] = pulp.LpVariable(f"b_{p}_{i}", 0, 1, cat="Binary")
            prob += y[p, i] + b[p, i] == 1
            for pos in pos_list:
                if pos in players_data[p]["allowed"]:
                    x[p, i, pos] = pulp.LpVariable(f"x_{p}_{i}_{pos}", 0, 1, cat="Binary")

    # Each position exactly once per inning
    for i in I:
        for pos in pos_list:
            elig = [x[p, i, pos] for p in P if (p, i, pos) in x]
            prob += pulp.lpSum(elig) == 1

    # One position per player per inning; link to y
    for p in P:
        for i in I:
            elig = [x[p, i, pos] for pos in pos_list if (p, i, pos) in x]
            prob += pulp.lpSum(elig) == y[p, i]

    # Fixed # on field
    need_on_field = len(pos_list)
    for i in I:
        prob += pulp.lpSum(y[p, i] for p in P) == need_on_field

    # Bench limits
    for p in P:
        prob += pulp.lpSum(b[p, i] for i in I) <= players_data[p]["bench_max"]

    # Hard cap: max 2 benches in a row
    for p in P:
        for i in range(innings - 2):
            prob += b[p, i] + b[p, i + 1] + b[p, i + 2] <= 2

    # Objective terms
    PRIO_WEIGHT = 100
    BENCH_WEIGHT = 10
    BENCH_STREAK_W = int(bench_streak_weight)
    AVOID_SAME_WEIGHT = 5   # stronger push to rotate for avoided positions
    STAY_SAME_REWARD = 1    # gentle reward for keeping same position otherwise

    terms = []

    # (A) priority adherence + (B) fairness bench cost
    max_bench = max((d["bench_max"] for d in players_data), default=0)
    for p in P:
        pr_costs = players_data[p]["prio_costs"]
        # fairness weight: those with lower bench_max get higher cost per bench
        fair_w = (max_bench - players_data[p]["bench_max"] + 1)
        for i in I:
            for pos in pos_list:
                if (p, i, pos) in x:
                    c = pr_costs.get(pos, 1000)
                    terms.append(PRIO_WEIGHT * c * x[p, i, pos])
            terms.append(BENCH_WEIGHT * fair_w * b[p, i])

    # (C) soft penalty for back-to-back benches
    if BENCH_STREAK_W > 0:
        for p in P:
            for i in range(1, innings):
                bb = pulp.LpVariable(f"bb_{p}_{i}", 0, 1, cat="Binary")
                prob += bb <= b[p, i]
                prob += bb <= b[p, i - 1]
                prob += bb >= b[p, i] + b[p, i - 1] - 1
                terms.append(BENCH_STREAK_W * bb)

    # (D) per-position consecutive SAME shaping:
    #     - For pos in AVOID: penalize consecutive SAME (encourage rotation)
    #     - Else: reward consecutive SAME (discourage churn)
    for p in P:
        for i in range(1, innings):
            for pos in pos_list:
                if (p, i, pos) in x and (p, i - 1, pos) in x:
                    s = pulp.LpVariable(f"same_{p}_{i}_{pos}", 0, 1, cat="Binary")
                    prob += s <= x[p, i, pos]
                    prob += s <= x[p, i - 1, pos]
                    prob += s >= x[p, i, pos] + x[p, i - 1, pos] - 1
                    if pos in AVOID:
                        terms.append(AVOID_SAME_WEIGHT * s)      # penalize staying same
                    else:
                        terms.append(-STAY_SAME_REWARD * s)      # reward staying same

    prob += pulp.lpSum(terms)
    return prob, x, y, b


def solve_schedule(prob: pulp.LpProblem) -> Tuple[str, float]:
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=15)
    status = prob.solve(solver)
    return pulp.LpStatus[status], pulp.value(prob.objective)


# ---------------- Arrow-based reordering (innings & lineup) ----------------

def _init_inning_order(innings: int, key: str = "inning_order"):
    if key not in st.session_state or st.session_state.get("_last_innings") != innings:
        st.session_state[key] = list(range(1, innings + 1))
        st.session_state["_last_innings"] = innings
    return st.session_state[key]

def _init_lineup_order(names: List[str], key: str = "lineup_order"):
    snap = key + "_snap"
    if key not in st.session_state or st.session_state.get(snap) != tuple(names):
        st.session_state[key] = list(range(len(names)))
        st.session_state[snap] = tuple(names)
    return st.session_state[key]

def apply_inning_order(df: pd.DataFrame, order: List[int], name_col: str) -> pd.DataFrame:
    cols = [str(i) for i in order]
    return df[[name_col] + cols] if name_col in df.columns else df[cols]

def apply_lineup_order(df: pd.DataFrame, order: List[int], name_col: str) -> pd.DataFrame:
    return df.iloc[order].reset_index(drop=True) if name_col in df.columns else df.loc[[df.index[i] for i in order]]

def reorder_with_arrows(schedule_df: pd.DataFrame, innings: int, name_col: str = "Name") -> pd.DataFrame:
    names = schedule_df[name_col].astype(str).tolist() if name_col in schedule_df.columns else schedule_df.index.astype(str).tolist()
    in_ord = _init_inning_order(innings)
    ln_ord = _init_lineup_order(names)

    st.markdown("### Reorder innings")
    per_row = 4
    for rstart in range(0, len(in_ord), per_row):
        row = st.columns(per_row)
        for i, c in enumerate(row):
            idx = rstart + i
            if idx >= len(in_ord): continue
            val = in_ord[idx]
            with c:
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    left = st.button("◀", key=f"in_l_{val}", disabled=(idx == 0))
                with c2:
                    st.caption(f"**Inning {val}**")
                with c3:
                    right = st.button("▶", key=f"in_r_{val}", disabled=(idx == len(in_ord) - 1))
                if left:
                    in_ord[idx-1], in_ord[idx] = in_ord[idx], in_ord[idx-1]
                    st.session_state["inning_order"] = in_ord
                    st.rerun()
                if right:
                    in_ord[idx+1], in_ord[idx] = in_ord[idx], in_ord[idx+1]
                    st.session_state["inning_order"] = in_ord
                    st.rerun()

    out = apply_inning_order(schedule_df, in_ord, name_col)

    st.markdown("### Reorder lineup (batting order)")
    for i, idx in enumerate(ln_ord):
        nm = names[idx]
        a, b, d = st.columns([1, 6, 1])
        with a:
            up = st.button("▲", key=f"ln_u_{i}", disabled=(i == 0))
        with b:
            st.caption(nm)
        with d:
            dn = st.button("▼", key=f"ln_d_{i}", disabled=(i == len(ln_ord) - 1))
        if up:
            ln_ord[i-1], ln_ord[i] = ln_ord[i], ln_ord[i-1]
            st.session_state["lineup_order"] = ln_ord
            st.rerun()
        if dn:
            ln_ord[i+1], ln_ord[i] = ln_ord[i], ln_ord[i+1]
            st.session_state["lineup_order"] = ln_ord
            st.rerun()

    return apply_lineup_order(out, ln_ord, name_col)


# ---------------- UI ----------------

st.title("🥎 Softball Fielding Schedule Generator")

# Game settings (top)
c1, c2, c3 = st.columns([1.2, 1.0, 1.1])
with c1:
    innings = st.number_input("Number of innings", 1, 12, 7, step=1)
with c2:
    of_choice = st.radio("Outfielders", [3, 4], index=0, horizontal=True)
with c3:
    bench_streak_weight = st.selectbox(
        "Penalty for back-to-back benches",
        options=[0,1,2,3,4,5,6,7,8,9,10],
        index=4,
        help="Higher discourages benching the same player in consecutive innings. 0 disables."
    )

pos_list = positions_for(of_choice)

# NEW: positions to avoid consecutive SAME innings
avoid_seq_positions = st.multiselect(
    "Avoid sequential innings for:",
    options=pos_list,
    default=[],
    help="Selected positions will be encouraged to rotate (penalize back-to-back same). Others get a small reward to stay the same."
)
# --- Move Streamlit data-editor toolbar to the far-right, inside the table ---
st.markdown("""
<style>
/* Anchor the table container so the toolbar can be positioned relative to it */
.stDataFrame, [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  position: relative !important;
}

/* Reposition the floating toolbar (covers multiple Streamlit versions/selectors) */
.stDataFrame [data-testid="stElementToolbar"],
[data-testid="stDataFrame"] [data-testid="stElementToolbar"],
[data-testid="stDataEditor"] [data-testid="stElementToolbar"],
div[aria-label="Data editor toolbar"],
div[aria-label="Table toolbar"] {
  position: absolute !important;
  left: auto !important;
  right: .35rem !important;    /* push to far-right */
  top: .35rem !important;      /* sit just inside the top of the table */
  transform: none !important;
  z-index: 2 !important;       /* above grid chrome but not your headings */
}

/* Add a tiny cushion above the table so the toolbar never touches your title/legend */
.roster-padding { height: .4rem; }
</style>
""", unsafe_allow_html=True)

st.subheader("Roster & Preferences")

# small vertical gap so the editor toolbar doesn't cover the heading
st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

# --- Column legend shown above the table ---
st.markdown("""
<style>
/* compact, responsive legend chips */
.legend {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: .5rem .75rem;
  margin: .35rem 0 0.6rem 0;
}
.legend .chip{
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.12);
  padding: .55rem .7rem;
  border-radius: .55rem;
  font-size: .95rem;
  line-height: 1.25rem;
}
.legend b {font-weight: 700}
</style>

<div class="legend">
  <div class="chip"><b>Name</b> — Player’s name.</div>
  <div class="chip"><b>P1–P5</b> — Position preferences in priority order.  
  Leave unused priorities blank; only chosen positions are eligible.</div>
  <div class="chip"><b>Bench (max)</b> — Max innings this player may sit (0 = never bench).</div>
</div>

<small>
<b>Position codes:</b> <code>P</code>, <code>C</code>, <code>1B</code>, <code>2B</code>,
<code>3B</code>, <code>SS</code>, <code>LF</code>, <code>CF</code>, <code>RF</code>
</small>
""", unsafe_allow_html=True)

st.markdown('<div class="roster-padding"></div>', unsafe_allow_html=True)

# Input table (horizontally laid out; page handles scrolling)
max_players = 17
df_default = pd.DataFrame({
    "Name": ["" for _ in range(max_players)],
    "P1": [None]*max_players,
    "P2": [None]*max_players,
    "P3": [None]*max_players,
    "P4": [None]*max_players,
    "P5": [None]*max_players,
    "Bench": [0]*max_players,   # shorter header avoids truncation
})

opt_list = ["— (unused) —"] + pos_list
col_cfg = {
    "Name":  st.column_config.TextColumn("Name", width="medium"),
    "P1":    st.column_config.SelectboxColumn("P1", options=opt_list, default="— (unused) —", width="small"),
    "P2":    st.column_config.SelectboxColumn("P2", options=opt_list, default="— (unused) —", width="small"),
    "P3":    st.column_config.SelectboxColumn("P3", options=opt_list, default="— (unused) —", width="small"),
    "P4":    st.column_config.SelectboxColumn("P4", options=opt_list, default="— (unused) —", width="small"),
    "P5":    st.column_config.SelectboxColumn("P5", options=opt_list, default="— (unused) —", width="small"),
    "Bench": st.column_config.NumberColumn("Bench (max)", min_value=0, max_value=innings, step=1, width="medium"),
}



df = st.data_editor(
    df_default,
    column_config=col_cfg,
    num_rows="fixed",
    hide_index=True,
    use_container_width=False,   # natural width; page does the sideways scroll
)

# Parse roster entries
players_data: List[Dict] = []
for _, row in df.iterrows():
    name = (row.get("Name") or "").strip()
    if not name:
        continue
    prios_raw = [row.get(c) for c in ["P1","P2","P3","P4","P5"]]
    prios = []
    seen = set()
    for r in prios_raw:
        if r and r != "— (unused) —" and r not in seen:
            prios.append(r); seen.add(r)
    if not prios:
        # They can still be scheduled only on Bench if needed, but that risks infeasibility.
        pass
    benchable = int(row.get("Bench") or 0)
    players_data.append({
        "name": name,
        "allowed": set(prios),
        "bench_max": benchable,
        "prio_costs": {pos: PRIO_COST[prios.index(pos) + 1] for pos in prios}
    })

st.divider()
gen = st.button("Generate Schedule", type="primary", use_container_width=True)

# ---------------- Generate & Show ----------------

def explain_infeasibility(players_data: List[Dict], pos_list: List[str], innings: int) -> str:
    counts = {pos: 0 for pos in pos_list}
    for d in players_data:
        for pos in d["allowed"]:
            if pos in counts: counts[pos] += 1
    need_benches = innings * max(0, len(players_data) - len(pos_list))
    allow_benches = sum(d["bench_max"] for d in players_data)
    lines = [
        "The schedule is infeasible with the current inputs.", "",
        f"- Innings per position: {innings}",
        "- Eligible players per position:",
    ] + [f"    • {pos}: {counts[pos]} eligible" for pos in pos_list] + [
        "",
        f"- Required benches across all innings: {need_benches}",
        f"- Sum of allowed benches (your inputs): {allow_benches}",
    ]
    if allow_benches < need_benches:
        lines.append("  → Not enough total bench allowance for required benches.")
    for pos in pos_list:
        if counts[pos] == 0:
            lines.append(f"  → No one is eligible for {pos}. Add at least one player who can play {pos}.")
    return "\n".join(lines)

if gen:
    if len(players_data) == 0:
        st.error("Add at least one player.")
        st.stop()
    if len(players_data) < len(pos_list):
        st.error(f"You need at least **{len(pos_list)}** players to fill all positions each inning.")
        st.stop()

    needed_benches = innings * (len(players_data) - len(pos_list))
    allowed_benches = sum(p["bench_max"] for p in players_data)
    if needed_benches > 0 and allowed_benches < needed_benches:
        st.error(
            f"Infeasible: across the game you must bench **{needed_benches}** times, "
            f"but the roster only allows **{allowed_benches}** bench-innings."
        )
        st.info(explain_infeasibility(players_data, pos_list, innings))
        st.stop()

    prob, x, y, b = build_model(players_data, pos_list, innings, bench_streak_weight, avoid_seq_positions)
    status_str, _ = solve_schedule(prob)

    if status_str in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined"):
        if status_str == "Optimal" or (status_str == "Not Solved" and pulp.value(prob.objective) is not None):
            names = [p["name"] for p in players_data]
            # Build wide table: Name | 1..N
            rows = []
            for p_idx, name in enumerate(names):
                row = {"Name": name}
                for i in range(innings):
                    assigned = None
                    for pos in pos_list:
                        if (p_idx, i, pos) in x and pulp.value(x[p_idx, i, pos]) > 0.5:
                            assigned = pos; break
                    row[str(i + 1)] = assigned if assigned else "Bench"
                rows.append(row)
            schedule_df = pd.DataFrame(rows, columns=["Name"] + [str(i + 1) for i in range(innings)])

            st.success(f"Schedule generated. Solver status: {status_str}")

            # Post-solve reordering (arrows)
            schedule_df = reorder_with_arrows(schedule_df, innings=innings, name_col="Name")

            # Render as simple HTML table so page handles horizontal scroll
            name_w = 140
            col_w = 120
            total_w = name_w + innings * col_w + 60

            def df_to_html_table(df: pd.DataFrame) -> str:
                styles = f"""
                <style>
                    .sched-wrapper {{
                        width: {total_w}px;
                    }}
                    table.sched {{
                        border-collapse: collapse;
                        width: {total_w}px;
                        table-layout: fixed;
                        font-size: 14px;
                        color: #fff;
                    }}
                    table.sched th, table.sched td {{
                        border: 1px solid rgba(255,255,255,0.2);
                        padding: 6px 8px;
                        text-align: center;
                    }}
                    table.sched th:first-child, table.sched td:first-child {{
                        width: {name_w}px; text-align: left; font-weight: 600;
                    }}
                    table.sched th:not(:first-child), table.sched td:not(:first-child) {{
                        width: {col_w}px;
                    }}
                </style>
                """
                html = ['<div class="sched-wrapper">', '<table class="sched">', "<thead><tr>"]
                for c in df.columns: html.append(f"<th>{c}</th>")
                html.append("</tr></thead><tbody>")
                for _, r in df.iterrows():
                    html.append("<tr>")
                    for c in df.columns: html.append(f"<td>{r[c]}</td>")
                    html.append("</tr>")
                html.append("</tbody></table></div>")
                return styles + "".join(html)

            st.markdown("## Schedule")
            st.markdown(df_to_html_table(schedule_df), unsafe_allow_html=True)

            # Download CSV
            st.download_button(
                "Download CSV",
                data=schedule_df.to_csv(index=False).encode("utf-8"),
                file_name="softball_fielding_schedule.csv",
                mime="text/csv",
            )

            # Bench summary
            play_counts = {n: 0 for n in names}
            for i in range(innings):
                for p_idx, n in enumerate(names):
                    if pulp.value(y[p_idx, i]) > 0.5:
                        play_counts[n] += 1
            bench_rows = []
            for n in names:
                played = play_counts[n]
                bench_rows.append({
                    "Player": n,
                    "Played": played,
                    "Benched": innings - played,
                    "Bench max (allowed)": next(p["bench_max"] for p in players_data if p["name"] == n),
                })
            st.markdown("**Bench Summary**")
            st.dataframe(pd.DataFrame(bench_rows).sort_values(["Benched", "Player"]),
                         use_container_width=True, hide_index=True)

        else:
            st.error(f"No feasible schedule found. Solver status: {status_str}")
            st.info(explain_infeasibility(players_data, pos_list, innings))
    else:
        st.error(f"Solver status: {status_str} (unexpected)")

# Close wrapper
st.markdown('</div>', unsafe_allow_html=True)



