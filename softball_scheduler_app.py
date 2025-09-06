# softball_scheduler_app.py
# Streamlit GUI + OR-Tools scheduler with:
# - 3 vs 4 outfielders toggle
# - Up to 17 players, 5 priority positions each
# - Benchable-innings (max) per player
# - Tie-break: minimize position changes across consecutive innings
# - MAX 2 consecutive benches (hard cap)
# - Bench streak penalty (soft) NOW selectable in the sidebar
# - Output: Players × Innings grid (cells show position or "Bench")

from __future__ import annotations
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

st.set_page_config(page_title="Softball Fielding Scheduler", layout="wide")

# ---------------- Helpers ----------------
def positions_for(outfielders: int) -> List[str]:
    infield = ["P", "C", "1B", "2B", "3B", "SS"]
    if outfielders == 3:
        of = ["LF", "CF", "RF"]
    else:
        of = ["LF", "LCF", "RCF", "RF"]  # common 4-OF layout
    return infield + of

# Priority costs (lower is better)
PRIO_COST = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10}

def build_model(
    players_data: List[Dict],
    pos_list: List[str],
    innings: int,
    bench_streak_weight: int,   # <<< new, from the GUI
) -> Tuple[cp_model.CpModel, Dict, Dict]:
    """
    players_data item:
      {
        'name': str,
        'allowed': set[str],         # positions this player may play
        'bench_max': int,            # max innings this player can be benched
        'prio_costs': dict[str,int]  # position -> cost (lower better)
      }
    """
    model = cp_model.CpModel()
    P = range(len(players_data))
    I = range(innings)

    # Decision variables
    x = {}  # x[p,i,pos] = 1 if player p plays pos in inning i
    y = {}  # y[p,i]     = 1 if player p plays some position in inning i
    b = {}  # b[p,i]     = 1 if player p is benched in inning i

    for p in P:
        name = players_data[p]["name"]
        allowed = players_data[p]["allowed"]
        for i in I:
            y[p, i] = model.NewBoolVar(f"plays_{name}_inn{i+1}")
            b[p, i] = model.NewBoolVar(f"bench_{name}_inn{i+1}")
            # exactly one of (play, bench)
            model.Add(y[p, i] + b[p, i] == 1)
            for pos in pos_list:
                if pos in allowed:
                    x[p, i, pos] = model.NewBoolVar(f"x_{name}_inn{i+1}_{pos}")

    # 1) Each position filled exactly once each inning
    for i in I:
        for pos in pos_list:
            elig = [x[p, i, pos] for p in P if (p, i, pos) in x]
            if not elig:
                impossible = model.NewIntVar(1, 0, f"impossible_{pos}_inn{i+1}")
                model.Add(impossible == 0)  # forces infeasible if uncovered
            else:
                model.Add(sum(elig) == 1)

    # 2) A player plays at most one position in an inning; link to y
    for p in P:
        for i in I:
            elig = [x[p, i, pos] for pos in pos_list if (p, i, pos) in x]
            if elig:
                model.Add(sum(elig) == y[p, i])
            else:
                model.Add(y[p, i] == 0)

    # 3) Exactly |pos_list| players on field each inning
    need_on_field = len(pos_list)
    for i in I:
        model.Add(sum(y[p, i] for p in P) == need_on_field)

    # 4) Bench limits (+ link b to y)
    for p in P:
        bench_max = players_data[p]["bench_max"]
        model.Add(sum(b[p, i] for i in I) == innings - sum(y[p, i] for i in I))
        model.Add(sum(b[p, i] for i in I) <= bench_max)

    # 5) Hard cap: NO three benches in a row (max 2 consecutive benches)
    for p in P:
        for i in range(innings - 2):
            model.Add(b[p, i] + b[p, i + 1] + b[p, i + 2] <= 2)

    # -------- Objective (lexi-style weights) --------
    PRIO_WEIGHT   = 100  # primary: honor priorities
    BENCH_WEIGHT  = 10   # secondary: distribute benches toward players who can sit more
    CHANGE_WEIGHT = 1    # tie-break: minimize position changes
    BENCH_STREAK_WEIGHT = int(bench_streak_weight)  # GUI control (0..10 typical)

    cost_terms = []

    # A) priorities
    for p in P:
        pcost = players_data[p]["prio_costs"]
        for i in I:
            for pos in pos_list:
                if (p, i, pos) in x:
                    c = pcost.get(pos, 1000)
                    if c < 1000:
                        cost_terms.append(PRIO_WEIGHT * c * x[p, i, pos])

    # B) benches weighted by how little they can sit
    max_bench = max((d["bench_max"] for d in players_data), default=0)
    for p in P:
        weight = (max_bench - players_data[p]["bench_max"] + 1)  # smaller bench_max => larger penalty
        cost_terms.append(BENCH_WEIGHT * weight * sum(b[p, i] for i in I))

    # C) position-change tie-break while playing consecutive innings
    for p in P:
        for i in range(1, innings):
            played_both = model.NewBoolVar(f"played_both_p{p}_i{i}")
            model.Add(played_both <= y[p, i])
            model.Add(played_both <= y[p, i-1])
            model.Add(played_both >= y[p, i] + y[p, i-1] - 1)

            sames = []
            for pos in pos_list:
                if (p, i, pos) in x and (p, i-1, pos) in x:
                    s = model.NewBoolVar(f"samepos_p{p}_i{i}_{pos}")
                    model.Add(s <= x[p, i, pos])
                    model.Add(s <= x[p, i-1, pos])
                    model.Add(s >= x[p, i, pos] + x[p, i-1, pos] - 1)
                    sames.append(s)
            same_any = model.NewIntVar(0, 1, f"same_any_p{p}_i{i}")
            if sames:
                model.Add(same_any == sum(sames))
            else:
                model.Add(same_any == 0)

            change = model.NewBoolVar(f"change_p{p}_i{i}")
            model.Add(change >= played_both - same_any)
            model.Add(change <= played_both)
            cost_terms.append(CHANGE_WEIGHT * change)

    # D) soft penalty: discourage 2 benches in a row
    if BENCH_STREAK_WEIGHT > 0:
        for p in P:
            for i in range(1, innings):
                bench_both = model.NewBoolVar(f"bench_streak_p{p}_i{i}")
                model.Add(bench_both <= b[p, i])
                model.Add(bench_both <= b[p, i - 1])
                model.Add(bench_both >= b[p, i] + b[p, i - 1] - 1)
                cost_terms.append(BENCH_STREAK_WEIGHT * bench_both)

    model.Minimize(sum(cost_terms))
    return model, x, y


def explain_infeasibility(players_data: List[Dict], pos_list: List[str], innings: int) -> str:
    counts = {pos: 0 for pos in pos_list}
    for d in players_data:
        for pos in d["allowed"]:
            if pos in counts:
                counts[pos] += 1

    needed_benches = innings * max(0, len(players_data) - len(pos_list))
    allowed_benches = sum(d["bench_max"] for d in players_data)

    lines = [
        "The schedule is infeasible with the current inputs.",
        "",
        f"- Innings to fill per position: {innings}",
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
        lines.append("  → Not enough total bench allowance to cover required benches.")
    for pos in pos_list:
        if counts[pos] == 0:
            lines.append(f"  → No one is eligible for {pos}. Add at least one player who can play {pos}.")
    return "\n".join(lines)

# ---------------- UI ----------------
st.title("⚾ Softball Fielding Schedule Generator")

with st.sidebar:
    st.header("Game Settings")
    innings = st.number_input("Number of innings", min_value=1, max_value=12, value=7, step=1)
    of_choice = st.radio("Outfielders", options=[3, 4], index=0, horizontal=True)
    pos_list = positions_for(of_choice)

    # NEW: dropdown for bench streak weight
    bench_streak_weight = st.selectbox(
        "Penalty for back-to-back benches",
        options=[0,1,2,3,4,5,6,7,8,9,10],
        index=4,  # default 4
        help="Higher discourages benching the same player in two consecutive innings. 0 disables this penalty."
    )

    st.caption(f"Positions each inning: {', '.join(pos_list)}")

st.subheader("Roster & Preferences")
st.caption(
    "Enter up to 17 players. For each player, select up to five position preferences in order. "
    "Leaving a priority as 'unused' means it won't be considered. "
    "Only the selected positions are eligible for that player. "
    "Set 'Benchable innings' to the **maximum** they can sit (0 allowed)."
)

max_players = 17
cols_header = st.columns([2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.6])
for c, label in zip(cols_header, ["Name", "P1", "P2", "P3", "P4", "P5", "Benchable (max)"]):
    c.markdown(f"**{label}**")

roster_entries = []
for row in range(max_players):
    cols = st.columns([2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.6])
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
            prios.append(sel)
            chosen.append(sel)

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

    # Build solver inputs
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

    # Quick feasibility check on benches
    needed_benches = innings * (len(players_data) - len(pos_list))
    allowed_benches = sum(p["bench_max"] for p in players_data)
    if needed_benches > 0 and allowed_benches < needed_benches:
        st.error(
            f"Infeasible: across the game you must bench **{needed_benches}** times, "
            f"but the roster only allows **{allowed_benches}** bench-innings."
        )
        st.info(explain_infeasibility(players_data, pos_list, innings))
        st.stop()

    # Solve
    model, x, y = build_model(players_data, pos_list, innings, bench_streak_weight)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        names = [p["name"] for p in players_data]

        # Nice labels for display
        def label(pos: str | None) -> str:
            if pos is None:
                return "Bench"
            return {"C": "Catcher", "LCF": "LC", "RCF": "RC"}.get(pos, pos)

        # Build Players × Innings grid
        grid_rows = []
        for p_idx, name in enumerate(names):
            row = {"Lineup #": p_idx + 1, "Name": name}
            for i in range(innings):
                assigned_pos = None
                for pos in pos_list:
                    k = (p_idx, i, pos)
                    if k in x and solver.Value(x[k]) == 1:
                        assigned_pos = pos
                        break
                row[str(i + 1)] = label(assigned_pos)  # position or Bench
            grid_rows.append(row)

        grid = pd.DataFrame(grid_rows, columns=["Lineup #", "Name"] + [str(i + 1) for i in range(innings)])

        st.success("Schedule generated.")
        st.dataframe(grid, use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV (Players × Innings)",
            data=grid.to_csv(index=False).encode("utf-8"),
            file_name="softball_schedule_players_x_innings.csv",
            mime="text/csv",
        )

        # Bench summary
        play_counts = {n: 0 for n in names}
        for i in range(innings):
            for p_idx, n in enumerate(names):
                if solver.Value(y[p_idx, i]) == 1:
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
        st.error("No feasible schedule found with the given constraints.")
        st.info(explain_infeasibility(players_data, pos_list, innings))

