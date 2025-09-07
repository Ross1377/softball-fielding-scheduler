# softball_scheduler_app.py — page-level horizontal scroll (iOS-friendly),
# no nested scrollbars, bench column not truncated, navy fallback theme.

from __future__ import annotations
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="Softball Fielding Scheduler", layout="wide")

# ------------- NAVY THEME (fallback if .streamlit/config.toml isn't present) -------------
st.markdown("""
<style>
:root{
  --bg:#0b1f3a;             /* navy */
  --bg2:#122a4e;            /* slightly lighter navy */
  --fg:#ffffff;
  --accent:#4cc3ff;
}

/* Fallback dark theme if config.toml not applied */
[data-testid="stAppViewContainer"], body{
  background:var(--bg) !important; color:var(--fg) !important;
}
section.main .block-container{ background:var(--bg) !important; }

/* Links / primary buttons */
a, .stButton>button{ color:var(--accent) !important; }
/* Cards & widgets */
.stSelectbox, .stNumberInput, .stRadio, .stDataFrame, .stDownloadButton, .stDownloadButton>button,
[data-testid="stDataFrame"], [data-testid="stTable"]{
  color:var(--fg) !important;
}
</style>
""", unsafe_allow_html=True)

# --------- LAYOUT CSS: PAGE is the ONLY horizontal scroller (no nested) ---------
st.markdown("""
<style>
/* 1) The page (main section) scrolls horizontally */
section.main{
  overflow-x:auto !important;
  -webkit-overflow-scrolling:touch;
  overscroll-behavior-x:contain;
}

/* 2) Wrap everything in a natural-width canvas so both edges are reachable */
.page-canvas{
  display:inline-block;              /* shrink-to-fit */
  width:max-content;                 /* natural width (content defines it) */
  min-width:100%;                    /* never narrower than viewport */
  margin:0 !important;               /* no centering on small screens */
  padding-left:max(12px, env(safe-area-inset-left));
  padding-right:max(12px, env(safe-area-inset-right));
}

/* 3) Neutralize Streamlit's centered max-width */
section.main .block-container{ max-width:none !important; }

/* 4) Nuke ALL nested scrollbars inside the data editor */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] *{
  overflow:visible !important;       /* disable inner scroll areas */
}

/* Slightly smaller base size so more fits; users can pinch-zoom */
html, body, [data-testid="stAppViewContainer"]{ font-size:15px; }
</style>
""", unsafe_allow_html=True)

# Wrap everything so the CSS can measure natural width
st.markdown('<div class="page-canvas">', unsafe_allow_html=True)

# ---------------- Solver helpers ----------------
def positions_for(outfielders:int)->List[str]:
    infield = ["P","C","1B","2B","3B","SS"]
    of3 = ["LF","CF","RF"]
    of4 = ["LF","LCF","RCF","RF"]
    return infield + (of3 if outfielders==3 else of4)

PRIO_COST = {1:0, 2:1, 3:3, 4:6, 5:10}

def build_model(players_data:List[Dict], pos_list:List[str], innings:int, bench_streak_weight:int
) -> Tuple[pulp.LpProblem, Dict, Dict, Dict]:
    prob = pulp.LpProblem("softball_schedule", pulp.LpMinimize)
    P = range(len(players_data)); I = range(innings)

    x,y,b = {},{},{}
    for p in P:
        for i in I:
            y[p,i] = pulp.LpVariable(f"y_{p}_{i}",0,1,cat="Binary")
            b[p,i] = pulp.LpVariable(f"b_{p}_{i}",0,1,cat="Binary")
            prob += y[p,i] + b[p,i] == 1
            for pos in pos_list:
                if pos in players_data[p]["allowed"]:
                    x[p,i,pos] = pulp.LpVariable(f"x_{p}_{i}_{pos}",0,1,cat="Binary")

    # each position once per inning
    for i in I:
        for pos in pos_list:
            prob += pulp.lpSum(x[p,i,pos] for p in P if (p,i,pos) in x) == 1

    # one pos per player per inning => link to y
    for p in P:
        for i in I:
            prob += pulp.lpSum(x[p,i,pos] for pos in pos_list if (p,i,pos) in x) == y[p,i]

    # fixed number on field
    need_on_field = len(pos_list)
    for i in I:
        prob += pulp.lpSum(y[p,i] for p in P) == need_on_field

    # bench limits
    for p in P:
        prob += pulp.lpSum(b[p,i] for i in I) <= players_data[p]["bench_max"]

    # hard cap: max 2 benches in a row
    for p in P:
        for i in range(innings-2):
            prob += b[p,i] + b[p,i+1] + b[p,i+2] <= 2

    # objective
    PRIO_WEIGHT = 100; BENCH_WEIGHT = 10; CHANGE_WEIGHT = 1
    BENCH_STREAK_WEIGHT = int(bench_streak_weight)
    terms=[]

    # (A) priority adherence
    for p in P:
        pc = players_data[p]["prio_costs"]
        for i in I:
            for pos in pos_list:
                if (p,i,pos) in x:
                    c = pc.get(pos,1000)
                    if c<1000: terms.append(PRIO_WEIGHT * c * x[p,i,pos])

    # (B) fair benching (prefer benching those with larger allowance)
    max_bench = max((d["bench_max"] for d in players_data), default=0)
    for p in P:
        wt = (max_bench - players_data[p]["bench_max"] + 1)
        terms.append(BENCH_WEIGHT * wt * pulp.lpSum(b[p,i] for i in I))

    # (C) minimize position changes across consecutive played innings
    for p in P:
        for i in range(1,innings):
            pb = pulp.LpVariable(f"pb_{p}_{i}",0,1,cat="Binary")
            prob += pb <= y[p,i]; prob += pb <= y[p,i-1]
            prob += pb >= y[p,i] + y[p,i-1] - 1

            same=[]
            for pos in pos_list:
                if (p,i,pos) in x and (p,i-1,pos) in x:
                    s = pulp.LpVariable(f"s_{p}_{i}_{pos}",0,1,cat="Binary")
                    prob += s <= x[p,i,pos]; prob += s <= x[p,i-1,pos]
                    prob += s >= x[p,i,pos] + x[p,i-1,pos] - 1
                    same.append(s)
            same_sum = pulp.lpSum(same) if same else 0
            chg = pulp.LpVariable(f"chg_{p}_{i}",0,1,cat="Binary")
            prob += chg >= pb - same_sum
            prob += chg <= pb
            terms.append(CHANGE_WEIGHT * chg)

    # (D) soft penalty: back-to-back benches
    if BENCH_STREAK_WEIGHT>0:
        for p in P:
            for i in range(1,innings):
                bb = pulp.LpVariable(f"bb_{p}_{i}",0,1,cat="Binary")
                prob += bb <= b[p,i]; prob += bb <= b[p,i-1]
                prob += bb >= b[p,i] + b[p,i-1] - 1
                terms.append(BENCH_STREAK_WEIGHT * bb)

    prob += pulp.lpSum(terms)
    return prob,x,y,b

def explain_infeasibility(players_data:List[Dict], pos_list:List[str], innings:int)->str:
    counts={pos:0 for pos in pos_list}
    for d in players_data:
        for pos in d["allowed"]:
            if pos in counts: counts[pos]+=1
    need_benches = innings*max(0,len(players_data)-len(pos_list))
    allow_benches = sum(d["bench_max"] for d in players_data)

    lines=["The schedule is infeasible with the current inputs.","",
           f"- Innings per position: {innings}",
           "- Eligible players per position:"] \
           + [f"    • {pos}: {counts[pos]} eligible" for pos in pos_list] + ["",
           f"- Required benches across all innings: {need_benches}",
           f"- Sum of allowed benches (your inputs): {allow_benches}"]
    if allow_benches<need_benches:
        lines.append("  → Not enough total bench allowance for required benches.")
    for pos in pos_list:
        if counts[pos]==0:
            lines.append(f"  → No one is eligible for {pos}. Add at least one player who can play {pos}.")
    return "\n".join(lines)

# ---------------- UI ----------------
st.title("⚾ Softball Fielding Schedule Generator")

with st.container():
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        innings = st.number_input("Number of innings",1,12,7,1)
    with c2:
        of_choice = st.radio("Outfielders",[3,4],index=0,horizontal=True)
    with c3:
        bench_streak_weight = st.selectbox(
            "Penalty for back-to-back benches",
            options=[0,1,2,3,4,5,6,7,8,9,10], index=4,
            help="Higher discourages benching the same player in consecutive innings."
        )

pos_list = positions_for(of_choice)
st.caption(f"Positions each inning: {', '.join(pos_list)}")

st.subheader("Roster & Preferences")

max_players = 17
df_default = pd.DataFrame({
    "Name":["" for _ in range(max_players)],
    "P1":[None]*max_players,
    "P2":[None]*max_players,
    "P3":[None]*max_players,
    "P4":[None]*max_players,
    "P5":[None]*max_players,
    "Bench":[min(1,7)]*max_players,   # label shortened to avoid truncation
})

opt_list = ["— (unused) —"] + pos_list
col_cfg = {
    "Name":  st.column_config.TextColumn("Name", width="large"),
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
    use_container_width=False,   # IMPORTANT: natural width; page handles scrolling
)

# build roster entries
roster_entries:List[Dict] = []
for _,row in df.iterrows():
    name = (row.get("Name") or "").strip()
    if not name: continue
    prios_raw=[row.get(c) for c in ["P1","P2","P3","P4","P5"]]
    prios,seen=[],set()
    for r in prios_raw:
        if r and r!="— (unused) —" and r not in seen:
            prios.append(r); seen.add(r)
    benchable = int(row.get("Bench") or 0)
    roster_entries.append({"name":name,"prios":prios,"bench_max":benchable})

st.divider()
if st.button("Generate Schedule", type="primary", use_container_width=True):
    if not roster_entries:
        st.error("Add at least one player."); st.stop()
    if len(roster_entries)<len(pos_list):
        st.error(f"You need at least **{len(pos_list)}** players to fill all positions each inning."); st.stop()

    players_data=[]
    for d in roster_entries:
        if not d["prios"]:
            st.error(f"Player **{d['name']}** has no positions selected."); st.stop()
        allowed=set(d["prios"])
        prio_costs={pos:PRIO_COST[d["prios"].index(pos)+1] for pos in allowed}
        players_data.append({"name":d["name"],"allowed":allowed,"bench_max":d["bench_max"],"prio_costs":prio_costs})

    needed_benches = innings * (len(players_data)-len(pos_list))
    allowed_benches = sum(p["bench_max"] for p in players_data)
    if needed_benches>0 and allowed_benches<needed_benches:
        st.error(
            f"Infeasible: you must bench **{needed_benches}** times total, "
            f"but only **{allowed_benches}** are allowed by the roster."
        )
        st.info(explain_infeasibility(players_data,pos_list,innings))
        st.stop()

    prob,x,y,b = build_model(players_data,pos_list,innings,bench_streak_weight)
    status = pulp.LpStatus[pulp.PULP_CBC_CMD(msg=False, timeLimit=10).solve(prob)]

    if status=="Optimal" or pulp.value(prob.objective) is not None:
        names=[p["name"] for p in players_data]
        def label(pos:str|None)->str:
            if pos is None: return "Bench"
            return {"C":"Catcher","LCF":"LC","RCF":"RC"}.get(pos,pos)
        rows=[]
        for p_idx,name in enumerate(names):
            row={"Lineup #":p_idx+1,"Name":name}
            for i in range(innings):
                assigned=None
                for pos in pos_list:
                    if (p_idx,i,pos) in x and pulp.value(x[p_idx,i,pos])>0.5:
                        assigned=pos; break
                row[str(i+1)] = label(assigned)
            rows.append(row)
        grid=pd.DataFrame(rows, columns=["Lineup #","Name"]+[str(i+1) for i in range(innings)])
        st.success(f"Schedule generated. Solver status: {status}")
        st.dataframe(grid, use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV (Players × Innings)",
            data=grid.to_csv(index=False).encode("utf-8"),
            file_name="softball_schedule_players_x_innings.csv",
            mime="text/csv",
        )

        plays={n:0 for n in names}
        for i in range(innings):
            for p_idx,n in enumerate(names):
                if pulp.value(y[p_idx,i])>0.5: plays[n]+=1
        bench_rows=[]
        for n in names:
            played=plays[n]
            bench_rows.append({"Player":n,"Played":played,"Benched":innings-played,
                               "Bench max (allowed)": next(p["bench_max"] for p in players_data if p["name"]==n)})
        st.markdown("**Bench Summary**")
        st.dataframe(pd.DataFrame(bench_rows).sort_values(["Benched","Player"]),
                     use_container_width=True, hide_index=True)
    else:
        st.error(f"No feasible schedule found. Solver status: {status}")
        st.info(explain_infeasibility(players_data,pos_list,innings))

st.markdown('</div>', unsafe_allow_html=True)

