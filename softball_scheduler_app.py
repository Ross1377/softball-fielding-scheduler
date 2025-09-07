# softball_scheduler_app.py — PuLP solver + desktop-identical layout on mobile
# - Single-row, horizontal entry (Name, P1..P5, Benchable)
# - Game Settings at top (no sidebar)
# - Page-level horizontal scroll (iOS-safe); no nested table scrollbars
# - Wider Name column; compact P columns
# - Bench logic: max 2 in a row (hard), soft penalty for back-to-back benches,
#   minimize position changes across innings, honor priorities

from __future__ import annotations
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="Softball Fielding Scheduler", layout="wide")

# =========================
# Layout CSS (robust mobile)
# =========================
# Make the PAGE the only horizontal scroller and let content define natural width.
# This avoids left-edge cutoff on iOS and keeps desktop layout identical on mobile.
st.markdown("""
<style>
/* PAGE is the ONLY horizontal scroller (reliable on iOS Safari) */
section.main,
[data-testid="stAppViewContainer"] > .main{
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch;
  overscroll-behavior-x: contain;
}

/* Content wrapper uses natural width; never narrower than viewport; left-aligned.
   No hardcoded pixel width: the table decides the width it needs. */
.page-canvas{
  display: inline-block;             /* shrink-to-fit content */
  width: max-content;                /* expand to natural content width */
  min-width: 100%;                   /* but never smaller than the viewport */
  margin: 0 !important;              /* left align so left edge is reachable */
  padding-left: max(12px, env(safe-area-inset-left));
  padding-right: max(12px, env(safe-area-inset-right));
}

/* Neutralize Streamlit's centered max-width which can cause hidden edges */
section.main .block-container{ max-width: none !important; }

/* Let the editor/grid use natural width (no nested horizontal scrollbars) */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] div[role="grid"]{
  width: max-content !important;     /* grow to fit all columns */
  min-width: 100%;                   /* but never smaller than viewport */
  overflow: visible !important;
}

/* Slightly smaller UI so more fits on screen; users can pinch-zoom if needed */
html, body, [data-testid="stAppViewContainer"]{ font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# Wrap everything so the CSS can measure the natural content width
st.markdown('<div class="page-canvas">', unsafe_allow_html=True)

# ---------- Helpers ----------
def positions_for(outfielders: int) -> List[str]:
    infield = ["P", "C", "1B", "2B", "3B", "SS"]
    of3 = ["LF", "CF", "RF"]
    of4 = ["LF", "LCF", "RCF", "RF"]
    return infield + (of3 if outfielders == 3 else of4)

# Lower is better; steeper costs favor higher-priority positions
PRIO_COST = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10}

def build_model(
    players_data: List[Dict],
    pos_list: List[str],
    innings: int,
    bench_streak_weight: int,
) -> Tuple[pulp.LpProblem, Dict, Dict, Dict]:
    prob = pulp.LpProblem("softball_schedule", pulp.LpMinimize)
    P = range(len(players_data))
    I = range(innings)

    x, y, b = {}, {}, {}  # x[p,i,pos]=1 if player p at pos in inning i; y=playing; b=benched
    for p in P:
        for i in I:
            y[p, i] = pulp.LpVariable(f"y_{p}_{i}", 0, 1, cat="Binary")
            b[p, i] = pulp.LpVariable(f"b_{p}_{i}", 0, 1, cat="Binary")
            prob += y[p, i] + b[p, i] == 1  # play or bench, not both
            for pos in pos_list:
                if pos in players_data[p]["allowed"]:
                    x[p, i, pos] = pulp.LpVariable(f"x_{p}_{i}_{pos}", 0, 1, cat="Binary")

    # Each position exactly once per inning
    for i in I:
        for pos in pos_list:
            elig = [x[p, i, pos] for p in P if (p, i, pos) in x]
            prob += pulp.lpSum(elig) == 1

    # One position max per inning per player; link to y
    for p in P:
        for i in I:
            elig = [x[p, i, pos] for pos in pos_list if (p, i, pos) in x]
            prob += pulp.lpSum(elig) == y[p, i]

    # Fixed number on field
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

    # Objective weights
    PRIO_WEIGHT = 100
    BENCH_WEIGHT = 10
    CHANGE_WEIGHT = 1
    BENCH_STREAK_WEIGHT = int(bench_streak_weight)

    terms = []

    # (A) Priority adherence
    for p in P:
        pc = players_data[p]["prio_costs"]
        for i in I:
            for pos in pos_list:
                if (p, i, pos) in x:
                    c = pc.get(pos, 1000)
                    if c < 1000:
                        terms.append(PRIO_WEIGHT * c * x[p, i, pos])

    # (B) Fair benching relative to bench_max (prefer benching those with larger allowance)
    max_bench = max((d["bench_max"] for d in players_data), default=0)
    for p in P:
        wt = (max_bench - players_data[p]["bench_max"] + 1)
        terms.append(BENCH_WEIGHT * wt * pulp.lpSum(b[p, i] for i in I))

    # (C) Minimize position changes across consecutive played innings
    for p in P:
        for i in range(1, innings):
            pb = pulp.LpVariable(f"pb_{p}_{i}", 0, 1, cat="Binary")
            prob += pb <= y[p, i]
            prob += pb <= y[p, i - 1]
            prob += pb >= y[p, i] + y[p, i - 1] - 1

            same = []
            for pos in pos_list:
                if (p, i, pos) in x and (p, i - 1, pos) in x:
                    s = pulp.LpVariable(f"s_{p}_{i}_{pos}", 0, 1, cat="Binary")
                    prob += s <= x[p, i, pos]
                    prob += s <= x[p, i - 1, pos]
                    prob += s >= x[p, i, pos] + x[p, i - 1, pos] - 1
                    same.append(s)
            same_sum = pulp.lpSum(same) if same else 0

            chg = pulp.LpVariable(f"chg_{p}_{i}", 0, 1, cat="Binary")
            prob += chg >= pb - same_sum
            prob += chg <= pb
            terms.append(CHANGE_WEIGHT * chg)

    # (D) Soft penalty for back-to-back benches
    if BENCH_STREAK_WEIGHT > 0:
        for p in P:
            for i in range(1, innings):
                bb = pulp.LpVariable(f"bb_{p}_{i}", 0, 1, cat="Binary")
                prob += bb <= b[p, i]
                prob += bb <= b[p, i - 1]
                prob += bb >= b[p, i] + b[p, i - 1] - 1
                terms.append(BENCH_STREAK_WEIGHT * bb)

    prob += pulp.lpSum(terms)
    return prob, x, y, b

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

# ---------- UI ----------
st.title("⚾ Softball Fielding Schedule Generator")

# Game Settings (top, always visible)
with st.container():
    cols = st.columns([1, 1, 2])
    with cols[0]:
        innings = st.number_input("Number of innings", 1, 12, 7, 1)
    with cols[1]:
        of_choice = st.radio("Outfielders", [3, 4], index=0, horizontal=True)
    with cols[2]:
        bench_streak_weight = st.selectbox(
            "Penalty for back-to-back benches",
            options=[0,1,2,3,4,5,6,7,8,9,10], index=4,
            help="Higher discourages benching the same player in consecutive innings. 0 disables."
        )
pos_list = positions_for(of_choice)
st.caption(f"Positions each inning: {', '.join(pos_list)}")

st.subheader("Roster & Preferences")

# Data editor table (single-row horizontal per player; page handles horizontal scroll)
max_players = 17
df_default = pd.DataFrame({
    "Name": ["" for _ in range(max_players)],
    "P1": [None]*max_players,
    "P2": [None]*max_players,
    "P3": [None]*max_players,
    "P4": [None]*max_players,
    "P5": [None]*max_players,
    "Benchable": [min(1, innings)]*max_players,
})

opt_list = ["— (unused) —"] + pos_list
col_cfg = {
    "Name": st.column_config.TextColumn("Name", width="large"),
    "P1": st.column_config.SelectboxColumn("P1", options=opt_list, default="— (unused) —", width="small"),
    "P2": st.column_config.SelectboxColumn("P2", options=opt_list, default="— (unused) —", width="small"),
    "P3": st.column_config.SelectboxColumn("P3", options=opt_list, default="— (unused) —", width="small"),
    "P4": st.column_config.SelectboxColumn("P4", options=opt_list, default="— (unused) —", width="small"),
    "P5": st.column_config.SelectboxColumn("P5", options=opt_list, default="— (unused) —", width="small"),
    "Benchable": st.column_config.NumberColumn(
        "Benchable (max)", min_value=0, max_value=innings, step=1, default=min(1, innings), width="small"
    ),
}

df = st.data_editor(
    df_default,
    column_config=col_cfg,
    num_rows="fixed",
    hide_index=True,
    use_container_width=False,   # IMPORTANT: let the table take its natural width
)

# Parse table into roster entries
roster_entries: List[Dict] = []
for _, row in df.iterrows():
    name = (row.get("Name") or "").strip()
    if not name:
        continue
    prios_raw = [row.get(c) for c in ["P1","P2","P3","P4","P5"]]
    prios, seen = [], set()
    for r in prios_raw:
        if r and r != "— (unused) —" and r not in seen:
            prios.append(r); seen.add(r)
    benchable = int(row.get("Benchable") or 0)
    roster_entries.append({"name": name, "prios": prios, "bench_max": benchable})

st.divider()
if st.button("Generate Schedule", type="primary", use_container_width=True):
    if not roster_entries:
        st.error("Add at least one player."); st.stop()
    if len(roster_entries) < len(pos_list):
        st.error(f"You need at least **{len(pos_list)}** players to fill all positions each inning."); st.stop()

    players_data = []
    for d in roster_entries:
        if not d["prios"]:
            st.error(f"Player **{d['name']}** has no positions selected. Add at least one preference."); st.stop()
        allowed = set(d["prios"])
        prio_costs = {pos: PRIO_COST[d["prios"].index(pos) + 1] for pos in allowed}
        players_data.append({"name": d["name"], "allowed": allowed, "bench_max": d["bench_max"], "prio_costs": prio_costs})

    needed_benches = innings * (len(players_data) - len(pos_list))
    allowed_benches = sum(p["bench_max"] for p in players_data)
    if needed_benches > 0 and allowed_benches < needed_benches:
        st.error(
            f"Infeasible: across the game you must bench **{needed_benches}** times, "
            f"but the roster only allows **{allowed_benches}** bench-innings."
        ); st.info(explain_infeasibility(players_data, pos_list, innings)); st.stop()

    prob, x, y, b = build_model(players_data, pos_list, innings, bench_streak_weight)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    status = pulp.LpStatus[prob.solve(solver)]

    if status == "Optimal" or pulp.value(prob.objective) is not None:
        names = [p["name"] for p in players_data]

        def label(pos: str | None) -> str:
            if pos is None: return "Bench"
            return {"C": "Catcher", "LCF": "LC", "RCF": "RC"}.get(pos, pos)

        # Players × Innings output
        rows = []
        for p_idx, name in enumerate(names):
            row = {"Lineup #": p_idx + 1, "Name": name}
            for i in range(innings):
                assigned = None
                for pos in pos_list:
                    if (p_idx, i, pos) in x and pulp.value(x[p_idx, i, pos]) > 0.5:
                        assigned = pos; break
                row[str(i + 1)] = label(assigned)
            rows.append(row)

        grid = pd.DataFrame(rows, columns=["Lineup #", "Name"] + [str(i + 1) for i in range(innings)])
        st.success(f"Schedule generated. Solver status: {status}")
        st.dataframe(grid, use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV (Players × Innings)",
            data=grid.to_csv(index=False).encode("utf-8"),
            file_name="softball_schedule_players_x_innings.csv",
            mime="text/csv",
        )

        # Bench summary
        plays = {n: 0 for n in names}
        for i in range(innings):
            for p_idx, n in enumerate(names):
                if pulp.value(y[p_idx, i]) > 0.5:
                    plays[n] += 1
        bench_rows = []
        for n in names:
            played = plays[n]
            bench_rows.append({
                "Player": n,
                "Played": played,
                "Benched": innings - played,
                "Bench max (allowed)": next(p["bench_max"] for p in players_data if p["name"] == n),
            })
        st.markdown("**Bench Summary**")
        st.dataframe(
            pd.DataFrame(bench_rows).sort_values(["Benched", "Player"]),
            use_container_width=True, hide_index=True
        )
    else:
        st.error(f"No feasible schedule found. Solver status: {status}")
        st.info(explain_infeasibility(players_data, pos_list, innings))

# Close the wrapper
st.markdown('</div>', unsafe_allow_html=True)

