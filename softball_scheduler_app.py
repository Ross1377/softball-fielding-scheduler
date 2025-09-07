# softball_scheduler_app.py
# Streamlit GUI + OR-Tools scheduler with:
# - 3 or 4 outfielders
# - up to 17 players
# - per-player priority positions (P1..P5)
# - benchable innings per player
# - hard cap: max 2 consecutive benches
# - soft penalty for back-to-back benches (dropdown weight)
# - tie-break via consecutive-position logic:
#     * For positions selected in "Avoid sequential innings for:", we penalize consecutive SAME positions
#     * For all other positions, we reward consecutive SAME positions (gentle stability)
# - post-solve ARROW CONTROLS to reorder innings (◀ ▶) and lineup (▲ ▼)
#
# NOTE: requirements.txt must include:
#   streamlit==1.37.1
#   pandas==2.2.3
#   numpy==2.1.3
#   ortools==9.9.3963

from __future__ import annotations
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model


# -------------- Page + CSS --------------

st.set_page_config(
    page_title="Softball Fielding Schedule Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# Hide Streamlit toolbar / main menu, and allow page-level horizontal scrolling
st.markdown(
    """
    <style>
    /* Hide Streamlit's 3-dot and hamburger menu */
    div[data-testid="stToolbar"] {visibility: hidden !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Let the page, not the inner widgets, handle horizontal scroll */
    html, body { overflow-x: auto; }
    /* Give the main container a minimum width so the page can scroll on mobile */
    .block-container { min-width: 1600px; padding-top: 1rem; padding-bottom: 2rem; }

    /* Red primary button with strong text */
    .stButton>button {
        background: #ff4b4b !important;
        color: #0b1221 !important;   /* very dark text for contrast */
        border: 1px solid rgba(255,255,255,0.15) !important;
        font-weight: 700 !important;
    }

    .arrow-row { margin-top: .2rem; margin-bottom: .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------- Helpers --------------

def positions_for(outfielders: int) -> List[str]:
    """Return ordered positions for a given OF configuration."""
    infield = ["P", "C", "1B", "2B", "3B", "SS"]
    if outfielders == 3:
        of = ["LF", "CF", "RF"]
    else:
        # Common slowpitch 4-OF layout
        of = ["LF", "LCF", "RCF", "RF"]
    return infield + of


# Priority cost map (lower better)
PRIO_COST = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10}


def _priority_costs_for_row(row: Dict, all_positions: List[str]) -> Dict[str, int]:
    """Build position->cost for a player's selected priorities."""
    pos_cost: Dict[str, int] = {}
    picks = [row.get(f"P{i}") for i in range(1, 6)]
    prios = [p for p in picks if p and p != "None"]
    for rank, pos in enumerate(prios, start=1):
        pos_cost[pos] = PRIO_COST.get(rank, 12)
    return pos_cost


def build_model(
    players_df: pd.DataFrame,
    pos_list: List[str],
    innings: int,
    bench_streak_weight: int,
    avoid_seq_positions: List[str],
) -> Tuple[cp_model.CpModel, Dict, Dict]:
    """
    Build CP-SAT model.
      - players_df columns: Name, P1..P5, Benchable (int)
      - pos_list: field positions (no "Bench")
      - avoid_seq_positions: positions for which we DISCOURAGE consecutive SAME assignments
      - bench hard cap: max 2 in a row (window of 3)
      - objective: priority costs + small bench cost + consecutive-position shaping
    Returns: (model, X, meta)
    """
    model = cp_model.CpModel()
    n_pos = len(pos_list)
    bench_name = "Bench"
    avoid_set = set(avoid_seq_positions)

    # Keep only players with a non-empty name
    rows = []
    for _, r in players_df.iterrows():
        name = str(r["Name"]).strip()
        if name:
            rows.append(r)
    if not rows:
        raise ValueError("No players provided.")
    players = list(range(len(rows)))
    names = [str(r["Name"]).strip() for r in rows]

    # Per player, build allowed position set & costs
    allowed: List[set] = []
    pos_costs: List[Dict[str, int]] = []
    bench_max: List[int] = []
    for r in rows:
        costs = _priority_costs_for_row(r, pos_list)
        pos_costs.append(costs)
        allowed_positions = set(costs.keys())
        allowed.append(allowed_positions)
        bm = int(r.get("Benchable", 0) or 0)
        bench_max.append(max(0, bm))

    # Decision variables: x[p,i,pos] includes bench as a pseudo-position
    all_pos = pos_list + [bench_name]
    X = {(p, t, k): model.NewBoolVar(f"x_{p}_{t}_{all_pos[k]}")
         for p in players for t in range(innings) for k in range(n_pos + 1)}

    # Exactly one assignment per player & inning
    for p in players:
        for t in range(innings):
            model.Add(sum(X[p, t, k] for k in range(n_pos + 1)) == 1)

    # Position coverage per inning: each field position exactly one player
    for t in range(innings):
        for k in range(n_pos):
            model.Add(sum(X[p, t, k] for p in players) == 1)

    # Eligibility: player can only take positions they listed (or bench)
    for p in players:
        allowed_pos = allowed[p]
        for t in range(innings):
            for k in range(n_pos):
                pos_name = pos_list[k]
                if pos_name not in allowed_pos:
                    model.Add(X[p, t, k] == 0)

    # Bench totals per player
    bench_idx = n_pos
    for p in players:
        model.Add(sum(X[p, t, bench_idx] for t in range(innings)) <= bench_max[p])

    # Hard cap: no 3 consecutive benches (=> max 2 in a row)
    for p in players:
        if innings >= 3:
            for t in range(innings - 2):
                model.Add(
                    X[p, t, bench_idx] + X[p, t + 1, bench_idx] + X[p, t + 2, bench_idx] <= 2
                )

    # Objective
    objective_terms = []

    # (A) priority cost for field assignments + small bench per inning
    for p in players:
        costs = pos_costs[p]
        for t in range(innings):
            for k in range(n_pos):
                pos_name = pos_list[k]
                c = costs.get(pos_name, 1000)  # disallowed positions are already forced to 0
                objective_terms.append(c * X[p, t, k])
            # per-bench small cost (prefer fewer benches when choice)
            objective_terms.append(1 * X[p, t, bench_idx])

    # (B) soft penalty for back-to-back benches (t and t+1)
    if bench_streak_weight > 0 and innings >= 2:
        for p in players:
            for t in range(innings - 1):
                y = model.NewBoolVar(f"bb_{p}_{t}")
                # y == 1 if both benches
                model.Add(y <= X[p, t, bench_idx])
                model.Add(y <= X[p, t + 1, bench_idx])
                model.Add(y >= X[p, t, bench_idx] + X[p, t + 1, bench_idx] - 1)
                objective_terms.append(bench_streak_weight * y)

    # (C) position-sequence shaping between consecutive innings:
    #     - For positions in avoid_set: penalize staying in the SAME pos on t & t+1  (encourage change)
    #     - For other positions       : reward staying in the SAME pos on t & t+1   (discourage change)
    AVOID_SAME_WEIGHT = 5      # stronger push to rotate for avoided positions
    STAY_SAME_REWARD  = 1      # gentle reward for stability otherwise
    if innings >= 2:
        for p in players:
            for t in range(innings - 1):
                for k in range(n_pos):  # field positions only (exclude bench)
                    a = model.NewBoolVar(f"same_{p}_{t}_{k}")
                    # a == X[p,t,k] AND X[p,t+1,k]
                    model.Add(a <= X[p, t, k])
                    model.Add(a <= X[p, t + 1, k])
                    model.Add(a >= X[p, t, k] + X[p, t + 1, k] - 1)
                    pos_name = pos_list[k]
                    if pos_name in avoid_set:
                        objective_terms.append(AVOID_SAME_WEIGHT * a)       # penalize same-in-a-row
                    else:
                        objective_terms.append(-STAY_SAME_REWARD * a)       # reward same-in-a-row

    model.Minimize(sum(objective_terms))

    meta = {
        "players": players,
        "names": names,
        "pos_list": pos_list,
        "bench_idx": bench_idx,
        "n_pos": n_pos,
        "innings": innings,
    }
    return model, X, meta


def solve_and_table(model: cp_model.CpModel, X: Dict, meta: Dict) -> Tuple[pd.DataFrame, str]:
    """Solve model; return schedule dataframe and status text."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(), "Infeasible (no schedule satisfies all constraints)."

    players = meta["players"]
    names = meta["names"]
    pos_list = meta["pos_list"]
    n_pos = meta["n_pos"]
    bench_idx = meta["bench_idx"]
    innings = meta["innings"]

    # Build wide schedule: Name | 1 | 2 | ... | N
    data = {"Name": names}
    for t in range(innings):
        col = []
        for p in players:
            chosen = None
            for k in range(n_pos + 1):
                if solver.Value(X[p, t, k]) == 1:
                    chosen = pos_list[k] if k < n_pos else "Bench"
                    break
            col.append(chosen if chosen else "")
        data[str(t + 1)] = col

    df = pd.DataFrame(data)
    stat = "Optimal" if status == cp_model.OPTIMAL else "Feasible"
    return df, stat


# -------------- Arrow-based reordering (innings & lineup) --------------

def _init_inning_order(innings: int, state_key: str = "inning_order"):
    if state_key not in st.session_state or st.session_state.get("_last_innings") != innings:
        st.session_state[state_key] = list(range(1, innings + 1))
        st.session_state["_last_innings"] = innings
    return st.session_state[state_key]


def _init_lineup_order(current_names: List[str], state_key: str = "lineup_order"):
    snapshot_key = state_key + "_snapshot"
    if state_key not in st.session_state or st.session_state.get(snapshot_key) != tuple(current_names):
        st.session_state[state_key] = list(range(len(current_names)))
        st.session_state[snapshot_key] = tuple(current_names)
    return st.session_state[state_key]


def _apply_inning_order(df: pd.DataFrame, order: List[int], name_col: str):
    inning_cols = [str(i) for i in order]
    if name_col in df.columns:
        keep = [name_col] + inning_cols
        return df[keep]
    else:
        return df[inning_cols]


def _apply_lineup_order(df: pd.DataFrame, order: List[int], name_col: str):
    if name_col in df.columns:
        return df.iloc[order].reset_index(drop=True)
    else:
        names = df.index.tolist()
        new_index = [names[i] for i in order]
        return df.loc[new_index]


def reorder_with_arrows(schedule_df: pd.DataFrame, innings: int, name_col: str = "Name") -> pd.DataFrame:
    """Render arrow controls and return reordered dataframe."""
    # names snapshot
    if name_col in schedule_df.columns:
        names = schedule_df[name_col].astype(str).tolist()
    else:
        names = [str(x) for x in schedule_df.index.tolist()]

    inning_order = _init_inning_order(innings)
    lineup_order = _init_lineup_order(names)

    st.markdown("### Reorder innings")
    per_row = 4
    for rstart in range(0, len(inning_order), per_row):
        sub = st.columns(per_row)
        for i, c in enumerate(sub):
            idx = rstart + i
            if idx >= len(inning_order):
                continue
            val = inning_order[idx]
            with c:
                cc1, cc2, cc3 = st.columns([1, 2, 1])
                with cc1:
                    left = st.button("◀", key=f"move_in_left_{val}", disabled=(idx == 0))
                with cc2:
                    st.caption(f"**Inning {val}**")
                with cc3:
                    right = st.button("▶", key=f"move_in_right_{val}", disabled=(idx == len(inning_order)-1))
                if left:
                    inning_order[idx-1], inning_order[idx] = inning_order[idx], inning_order[idx-1]
                    st.session_state["inning_order"] = inning_order
                    st.rerun()
                if right:
                    inning_order[idx+1], inning_order[idx] = inning_order[idx], inning_order[idx+1]
                    st.session_state["inning_order"] = inning_order
                    st.rerun()

    # Apply inning order
    out_df = _apply_inning_order(schedule_df, inning_order, name_col)

    st.markdown("### Reorder lineup (batting order)")
    for i, idx in enumerate(lineup_order):
        shown_name = names[idx]
        a, b, d = st.columns([1, 6, 1])
        with a:
            up = st.button("▲", key=f"ln_up_{i}", disabled=(i == 0))
        with b:
            st.caption(shown_name)
        with d:
            down = st.button("▼", key=f"ln_dn_{i}", disabled=(i == len(lineup_order)-1))
        if up:
            lineup_order[i-1], lineup_order[i] = lineup_order[i], lineup_order[i-1]
            st.session_state["lineup_order"] = lineup_order
            st.rerun()
        if down:
            lineup_order[i+1], lineup_order[i] = lineup_order[i], lineup_order[i+1]
            st.session_state["lineup_order"] = lineup_order
            st.rerun()

    # Apply lineup order
    out_df = _apply_lineup_order(out_df, lineup_order, name_col)
    return out_df


# -------------- UI: Settings --------------

st.markdown("## 🥎 Softball Fielding Schedule Generator")

with st.container():
    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        innings = st.number_input("Number of innings", 3, 9, 7, step=1)
    with cols[1]:
        outfielders = st.radio("Outfielders", [3, 4], horizontal=True, index=1)
    with cols[2]:
        bench_streak_weight = st.selectbox(
            "Penalty for back-to-back benches",
            [0, 1, 2, 3, 4, 6, 8],
            index=4,
            help="Additional cost if a player is benched in two consecutive innings. 0 = no penalty.",
        )

# New option: avoid back-to-back SAME position (encourage rotation) for selected positions
all_positions = positions_for(outfielders)
avoid_seq_positions = st.multiselect(
    "Avoid consecutive innings for:",
    options=all_positions,
    default=[],
    help="Players will be encouraged to switch positions across consecutive innings for the selected positions."
)

st.markdown("## Roster & Preferences")

# Build roster editor (up to 17 players)
priority_opts = ["None"] + all_positions

if "roster_df" not in st.session_state:
    st.session_state["roster_df"] = pd.DataFrame(
        {
            "Name": ["" for _ in range(17)],
            "P1": ["None"] * 17,
            "P2": ["None"] * 17,
            "P3": ["None"] * 17,
            "P4": ["None"] * 17,
            "P5": ["None"] * 17,
            "Benchable": [0] * 17,
        }
    )

# Ensure options reflect current OF setting
editor = st.data_editor(
    st.session_state["roster_df"],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Name": st.column_config.TextColumn("Name", width="medium", required=False),
        "P1": st.column_config.SelectboxColumn("P1", options=priority_opts, width="small"),
        "P2": st.column_config.SelectboxColumn("P2", options=priority_opts, width="small"),
        "P3": st.column_config.SelectboxColumn("P3", options=priority_opts, width="small"),
        "P4": st.column_config.SelectboxColumn("P4", options=priority_opts, width="small"),
        "P5": st.column_config.SelectboxColumn("P5", options=priority_opts, width="small"),
        "Benchable": st.column_config.NumberColumn("Benchable", min_value=0, max_value=innings, step=1, width="small"),
    },
    key="roster_editor",
)

st.session_state["roster_df"] = editor

st.write("")  # breathing room
gen = st.button("Generate Schedule", type="primary")

# -------------- Generate + Show Schedule --------------

if gen:
    # Basic validation
    df_in = st.session_state["roster_df"].copy()
    df_in["Name"] = df_in["Name"].astype(str).str.strip()
    df_in = df_in[df_in["Name"] != ""].reset_index(drop=True)

    if df_in.empty:
        st.error("Please enter at least one player name.")
        st.stop()

    # Warn players with no field positions selected
    bad = []
    for i, r in df_in.iterrows():
        if all((r.get(f"P{k}") in (None, "", "None") for k in range(1, 6))):
            bad.append(r["Name"])
    if bad:
        st.warning(
            "These players have no field positions selected; they would only be eligible for Bench and may cause infeasibility: "
            + ", ".join(bad)
        )

    try:
        model, X, meta = build_model(df_in, all_positions, innings, bench_streak_weight, avoid_seq_positions)
        schedule_df, status_txt = solve_and_table(model, X, meta)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if schedule_df.empty:
        st.error(status_txt)
        st.info("Check that: each inning covers all field positions, each player has at least one allowed position, and bench limits allow enough sits.")
        st.stop()

    st.success(f"Solver status: {status_txt}")

    # Allow interactive reordering via arrows
    schedule_df = reorder_with_arrows(schedule_df, innings=innings, name_col="Name")

    # --------- Display as HTML table so the PAGE scrolls horizontally (not just the widget) ---------
    name_w = 220
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
        for c in df.columns:
            html.append(f"<th>{c}</th>")
        html.append("</tr></thead><tbody>")
        for _, r in df.iterrows():
            html.append("<tr>")
            for c in df.columns:
                html.append(f"<td>{r[c]}</td>")
            html.append("</tr>")
        html.append("</tbody></table></div>")
        return styles + "".join(html)

    st.markdown("## Schedule")
    st.markdown(df_to_html_table(schedule_df), unsafe_allow_html=True)

    # Download CSV
    csv_bytes = schedule_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="softball_fielding_schedule.csv",
        mime="text/csv",
    )

