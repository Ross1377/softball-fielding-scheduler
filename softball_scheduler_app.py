# softball_scheduler_app.py  (PuLP edition + Mobile mode)
# Streamlit GUI + MILP with PuLP (CBC solver)
# - 3 vs 4 outfielders toggle
# - Up to 17 players with 5 priority positions each
# - Bench max per player
# - Hard cap of 2 benches in a row
# - Soft penalty for back-to-back benches (dropdown weight)
# - Minimize position changes across consecutive innings
# - Mobile mode: stacked "one player per card" inputs for phones
# Run: streamlit run softball_scheduler_app.py

from __future__ import annotations
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="Softball Fielding Scheduler", layout="wide")

# ---------- tiny CSS tweaks (smaller controls on mobile) ----------
st.markdown("""
<style>
.block-container { padding-top: 0.5rem; padding-bottom: 2rem; }
[data-baseweb="select"] > div { min-height: 34px; }
.stButton>button { height: 40px; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def positions_for(outfielders: int) -> List[str]:
    infield = ["P", "C", "1B", "2B", "3B", "SS"]
    of3 = ["LF", "CF", "RF"]
    of4 = ["LF", "LCF", "RCF", "RF"]
    return infield + (of3 if outfielders == 3 else of4)

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

    x, y, b = {}, {}, {}
    for p in P:
        for i in I:
            y[p, i] = pulp.LpVariable(f"y_{p}_{i}", 0, 1, cat="Binary")
            b[p, i] = pulp.LpVariable(f"b_{p}_{i}", 0, 1, cat="Binary")
            prob += y[p, i] + b[p, i] == 1
            for pos in pos_list:
                if pos in players_data[p]["allowed"]:
                    x[p, i, pos] = pulp.LpVariable(f"x_{p}_{i}_{pos}", 0, 1, cat="Binary")

    # each position exactly once per inning
    for i in I:
        for pos in pos_list:
            elig = [x[p, i, pos] for p in P if (p, i, pos) in x]
            prob += pulp.lpSum(elig) == 1

    # a player at most one position per inning (link to y)
    for p in P:
        for i in I:
            elig = [x[p, i, pos] for pos in pos_list if (p, i, pos) in x]
            prob += pulp.lpSum(elig) == y[p, i]

    # fixed number on field
    need_on_field = len(pos_list)
    for i in I:
        prob += pulp.lpSum(y[p, i] for p in P) == need_on_field

    # bench limits
    for p in P:
        prob += pulp.lpSum(b[p, i] for i in I) <= players_data[p]["bench_max"]

    # hard cap: max 2 benches in a row
    for p in P:
        for i in range(innings - 2):
            prob += b[p, i] + b[p, i + 1] + b[p, i + 2] <= 2

    # ---- objective ----
    PRIO_WEIGHT = 100
    BENCH_WEIGHT = 10
    CHANGE_WEIGHT = 1
    BENCH_STREAK_WEIGHT = int(bench_streak_weight)

    cost_terms = []

    # (A) priority costs
    for p in P:
        pcost = players_data[p]["prio_costs"]
        for i in I:
            for pos in pos_list:
                if (p, i, pos) in x:
                    c = pcost.get(pos, 1000)
                    if c < 1000:
                        cost_terms.append(PRIO_WEIGHT * c * x[p, i, pos])

    # (B) distribute benches fairly wrt bench_max
    max_bench = max((d["bench_max"] for d in players_data), default=0)
    for p in P:
        weight = (max_bench - players_data[p]["bench_max"] + 1)
        cost_terms.append(BENCH_WEIGHT * weight * pulp.lpSum(b[p, i] for i in I))

    # (C) minimize position changes between consecutive played innings
    for p in P:
        for i in range(1, innings):
            played_both = pulp.LpVariable(f"pb_{p}_{i}", 0, 1, cat="Binary")
            prob += played_both <= y[p, i]
            prob += played_both <= y[p, i - 1]
            prob += played_both >= y[p, i] + y[p, i - 1] - 1

            sames = []
            for pos in pos_list:
                if (p, i, pos) in x and (p, i - 1, pos) in x:
                    s = pulp.LpVariable(f"same_{p}_{i}_{pos}", 0, 1, cat="Binary")
                    prob += s <= x[p, i, pos]
                    prob += s <= x[p, i - 1, pos]
                    prob += s >= x[p, i, pos] + x[p, i - 1, pos] - 1
                    sames.append(s)
            same_sum = pulp.lpSum(sames) if sames else 0

            change = pulp.LpVariable(f"chg_{p}_{i}", 0, 1, cat="Binary")
            prob += change >= played_both - same_sum
            prob += change <= played_both
            cost_terms.append(CHANGE_WEIGHT * change)

    # (D) soft penalty: discourage two consecutive benches
    if BENCH_STREAK_WEIGHT > 0:
        for p in P:
            for i in range(1, innings):
                bb = pulp.LpVariable(f"bb_{p}_{i}", 0, 1, cat="Binary")
                prob += bb <= b[p, i]
                prob += bb <= b[p, i - 1]
                prob += bb >= b[p, i] + b[p, i - 1] - 1
                cost_terms.append(BENCH_STREAK_WEIGHT * bb)

    prob += pulp.lpSum(cost_terms)
    return prob, x, y, b

def explain_infeasibility(players_data: List[Dict], pos_list: List[str], innings: int) -> str:
    counts = {pos: 0 for pos in pos_list}
    for d in players_data:
        for pos in d["allowed"]:
            if pos in counts:
                counts[pos] += 1
    needed_benches = innings * max(0, len(players_data) - len(pos_list))
    allowed_benches = sum(d["bench_max"] for d in players_data)

    lines = [
        "The schedule is infeasible with the current inputs.", "",
        f"- Innings per position: {innings}",
        "- Eligible players per position:",
    ]
    for pos in pos_list:
        lines.append(f"    • {pos}: {counts[pos]} eligible")
    lines += [
        "",
        f"- Required benches across all innings: {needed_benches}",
        f"- Sum of allowed benches (your inputs): {allowed_benches}",
    ]
    if allowed_benches < needed_benches:
        lines.append("  → Not enough total bench allowance for required benches.")
    for pos in pos_list:
        if counts[pos] == 0:
            lines.append(f"  → No one is eligible for {pos}. Add at least one player who can play {pos}.")
    return "\n".join(lines)

# ---------- UI ----------
st.title("⚾ Softball Fielding Schedule Generator")

with st.sidebar:
    st.header("Game Settings")
    innings = st.number_input("Number of innings", 1, 12, 7, 1)
    of_choice = st.radio("Outfielders", [3, 4], index=0, horizontal=True)
    pos_list = positions_for(of_choice)

    bench_streak_weight = st.selectbox(
        "Penalty for back-to-back benches",
        options=[0,1,2,3,4,5,6,7,8,9,10],
        index=4,
        help="Higher discourages benching the same player in consecutive innings. 0 disables."
    )

    mobile_mode = st.toggle("Mobile mode (stacked inputs)", value=True,
                            help="Turn off for desktop grid entry.")

st.subheader("Roster & Preferences")
st.caption(
    "Enter up to 17 players. For each player, select up to five positions in priority order. "
    "Unselected priorities are ignored; only selected positions are eligible. "
    "Set 'Benchable innings' to the maximum they can sit (0 allowed)."
)

max_players = 17
roster_entries: List[Dict] = []

if mobile_mode:
    # -------- mobile friendly: one player per card ----------
    for row in range(max_players):
        with st.expander(f"Player {row+1}", expanded=(row < 2)):
            name = st.text_input("Name", key=f"name_m_{row}")
            benchable = st.number_input(
                "Benchable (max innings)",
                min_value=0, max_value=innings, value=min(1, innings), step=1, key=f"bench_m_{row}"
            )
            prios, chosen = [], []
            for pidx in range(5):
                remaining = ["— (unused) —"] + [p for p in pos_list if p not in chosen]
                sel = st.selectbox(
                    f"Priority {pidx+1}",
                    remaining,
                    index=0,
                    key=f"prio_m_{row}_{pidx}",
                )
                if sel != "— (unused) —":
                    prios.append(sel); chosen.append(sel)
            if name.strip():
                roster_entries.append({"name": name.strip(), "prios": prios, "bench_max": int(benchable)})
else:
    # -------- desktop grid ----------
    cols_header = st.columns([2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.4])
    for c, label in zip(cols_header, ["Name","P1","P2","P3","P4","P5","Benchable (max)"]):
        c.markdown(f"**{label}**")

    for row in range(max_players):
        cols = st.columns([2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.4])
        name = cols[0].text_input("name", key=f"name_{row}", label_visibility="collapsed")

        chosen, prios = [], []
        for pidx in range(5):
            remaining = ["— (unused) —"] + [p for p in pos_list if p not in chosen]
            sel = cols[pidx + 1].selectbox(
                f"prio{pidx+1}",
                remaining,
                index=0,
                key=f"prio_{row}_{pidx}",
                label_visibility="collapsed",
            )
            if sel != "— (unused) —":
                prios.append(sel); chosen.append(sel)

        benchable = cols[6].number_input(
            "bench",
            min_value=0, max_value=innings, value=min(1, innings), step=1,
            key=f"bench_{row}", label_visibility="collapsed",
        )
        if name.strip():
            roster_entries.append({"name": name.strip(), "prios": prios, "bench_max": int(benchable)})

st.divider()
generate = st.button("Generate Schedule", type="primary", use_container_width=True)

if generate:
    if not roster_entries:
        st.error("Add at least one player.")
        st.stop()
    if len(roster_entries) < len(pos_list):
        st.error(f"You need at least **{len(pos_list)}** players to fill all positions each inning.")
        st.stop()

    players_data = []
    for d in roster_entries:
        if not d["prios"]:
            st.error(f"Player **{d['name']}** has no positions selected. Add at least one preference.")
            st.stop()
        allowed = set(d["prios"])
        prio_costs = {pos: PRIO_COST[d["prios"].index(pos) + 1] for pos in allowed}
        players_data.append(
            {"name": d["name"], "allowed": allowed, "bench_max": d["bench_max"], "prio_costs": prio_costs}
        )

    # benches feasibility
    needed_benches = innings * (len(players_data) - len(pos_list))
    allowed_benches = sum(p["bench_max"] for p in players_data)
    if needed_benches > 0 and allowed_benches < needed_benches:
        st.error(
            f"Infeasible: across the game you must bench **{needed_benches}** times, "
            f"but the roster only allows **{allowed_benches}** bench-innings."
        )
        st.info(explain_infeasibility(players_data, pos_list, innings))
        st.stop()

    prob, x, y, b = build_model(players_data, pos_list, innings, bench_streak_weight)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    result_status = prob.solve(solver)
    status_str = pulp.LpStatus[result_status]

    if status_str in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined"):
        if status_str == "Optimal" or (status_str == "Not Solved" and pulp.value(prob.objective) is not None):
            names = [p["name"] for p in players_data]

            def label(pos: str | None) -> str:
                if pos is None:
                    return "Bench"
                return {"C": "Catcher", "LCF": "LC", "RCF": "RC"}.get(pos, pos)

            # Players × Innings grid
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
            st.success(f"Schedule generated. Solver status: {status_str}")
            st.dataframe(grid, use_container_width=True, hide_index=True)

            st.download_button(
                "Download CSV (Players × Innings)",
                data=grid.to_csv(index=False).encode("utf-8"),
                file_name="softball_schedule_players_x_innings.csv",
                mime="text/csv",
            )

            # bench summary
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
            st.dataframe(
                pd.DataFrame(bench_rows).sort_values(["Benched", "Player"]),
                use_container_width=True, hide_index=True
            )
        else:
            st.error(f"No feasible schedule found. Solver status: {status_str}")
            st.info(explain_infeasibility(players_data, pos_list, innings))

