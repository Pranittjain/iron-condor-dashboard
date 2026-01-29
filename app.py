import base64
from io import BytesIO
from datetime import date, datetime

import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------
# Config + style
# --------------------------
st.set_page_config(page_title="Options Strategy Journal", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.kpi { border: 1px solid rgba(255,255,255,0.14); border-radius: 16px; padding: 14px 16px;
       background: rgba(255,255,255,0.04); }
.kpi-title { font-size: 0.85rem; color: rgba(255,255,255,0.65); margin-bottom: 4px; }
.kpi-value { font-size: 1.4rem; font-weight: 800; }
.small { font-size: 0.85rem; color: rgba(255,255,255,0.65); }
hr { border: none; height: 1px; background: rgba(255,255,255,0.10); margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("Options Strategy Journal")
st.caption("Templates → Runs → Daily Logs (EOD). Built for repeatable strategies and analytics.")

# --------------------------
# Helpers
# --------------------------
def now_id(prefix: str) -> str:
    return f"{prefix}_{int(datetime.now().timestamp())}"

def img_to_b64(uploaded) -> str:
    if uploaded is None:
        return ""
    return base64.b64encode(uploaded.getvalue()).decode("utf-8")

def b64_to_img(b64_str: str):
    if not b64_str:
        return None
    try:
        raw = base64.b64decode(b64_str)
        return Image.open(BytesIO(raw))
    except Exception:
        return None

def safe_dt(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

# --------------------------
# Data model
# --------------------------
TEMPLATE_COLS = [
    "template_id",
    "name",
    "strategy_type",
    "underlying",
    "default_lot_size",
    "legs_desc",          # free text (or strikes template)
    "entry_reason",
    "risk_rules",
    "exit_plan",
]

RUN_COLS = [
    "run_id",
    "template_id",
    "run_name",           # optional
    "start_date",
    "expiry",
    "entry_spot",
    "margin_used",
    "qty_lots",
    "lot_size",
    "screenshot_b64",     # payoff/graph for THIS run
    "status",             # ACTIVE / CLOSED
]

LOG_COLS = [
    "log_id",
    "run_id",
    "date",
    "mtm_inr",            # MTM since entry (total)
    "nifty_close",
    "notes"
]

# --------------------------
# Session state
# --------------------------
if "templates" not in st.session_state:
    st.session_state.templates = pd.DataFrame(columns=TEMPLATE_COLS)

if "runs" not in st.session_state:
    st.session_state.runs = pd.DataFrame(columns=RUN_COLS)

if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=LOG_COLS)

# --------------------------
# Sidebar nav + backup
# --------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Run Desk", "Strategy Analytics", "Create Template", "Create Run", "Backup / Restore"],
        index=0
    )

    st.divider()
    st.caption("Tip: This app stores data in memory. Use CSV backup to persist across sessions.")

# --------------------------
# Backup / Restore page
# --------------------------
def backup_restore_page():
    st.subheader("Backup / Restore")

    t_csv = st.session_state.templates.to_csv(index=False).encode("utf-8")
    r_csv = st.session_state.runs.to_csv(index=False).encode("utf-8")
    l_csv = st.session_state.logs.to_csv(index=False).encode("utf-8")

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download templates.csv", t_csv, file_name="templates.csv", mime="text/csv")
    c2.download_button("Download runs.csv", r_csv, file_name="runs.csv", mime="text/csv")
    c3.download_button("Download logs.csv", l_csv, file_name="logs.csv", mime="text/csv")

    st.markdown("hr", unsafe_allow_html=True)

    up1, up2, up3 = st.columns(3)
    up_t = up1.file_uploader("Upload templates.csv", type=["csv"])
    up_r = up2.file_uploader("Upload runs.csv", type=["csv"])
    up_l = up3.file_uploader("Upload logs.csv", type=["csv"])

    if st.button("Import CSVs"):
        if up_t is not None:
            df = pd.read_csv(up_t)
            st.session_state.templates = ensure_cols(df, TEMPLATE_COLS)

        if up_r is not None:
            df = pd.read_csv(up_r)
            st.session_state.runs = ensure_cols(df, RUN_COLS)

        if up_l is not None:
            df = pd.read_csv(up_l)
            st.session_state.logs = ensure_cols(df, LOG_COLS)

        st.success("Import complete.")

    st.markdown("hr", unsafe_allow_html=True)
    if st.button("Reset everything (in memory)"):
        st.session_state.templates = pd.DataFrame(columns=TEMPLATE_COLS)
        st.session_state.runs = pd.DataFrame(columns=RUN_COLS)
        st.session_state.logs = pd.DataFrame(columns=LOG_COLS)
        st.warning("Reset done.")

# --------------------------
# Create Template
# --------------------------
def create_template_page():
    st.subheader("Create Strategy Template")
    st.caption("A template is your strategy model. You can run it multiple times across expiries.")

    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Template name", placeholder="IC 300/50 Low Risk")
    strategy_type = c2.text_input("Strategy type", value="Iron Condor")
    underlying = c3.text_input("Underlying", value="NIFTY")

    default_lot_size = st.number_input("Default lot size", min_value=1, value=50, step=1)

    legs_desc = st.text_area("Legs / setup description (free text)", height=80,
                             placeholder="Example: Sell 10-delta CE/PE, wings 50 points, adjust on breach, etc.")

    entry_reason = st.text_area("Entry logic", height=80, placeholder="IVP/VIX regime, expected range, event filter...")
    risk_rules = st.text_area("Risk rules", height=80, placeholder="Max loss, adjustment rules, hedge triggers...")
    exit_plan = st.text_area("Exit plan", height=80, placeholder="Profit target, time exit, IV crush exit...")

    if st.button("Add template"):
        if not name.strip():
            st.error("Template name is required.")
            return
        tid = now_id("TEMPLATE")
        row = {
            "template_id": tid,
            "name": name.strip(),
            "strategy_type": strategy_type.strip(),
            "underlying": underlying.strip(),
            "default_lot_size": int(default_lot_size),
            "legs_desc": legs_desc.strip(),
            "entry_reason": entry_reason.strip(),
            "risk_rules": risk_rules.strip(),
            "exit_plan": exit_plan.strip()
        }
        st.session_state.templates = pd.concat([st.session_state.templates, pd.DataFrame([row])], ignore_index=True)
        st.success("Template created.")

    st.markdown("hr", unsafe_allow_html=True)

    # delete template (only if no runs exist)
    if not st.session_state.templates.empty:
        st.subheader("Delete template")
        pick = st.selectbox(
            "Select template to delete",
            st.session_state.templates["template_id"].tolist(),
            format_func=lambda x: st.session_state.templates.loc[st.session_state.templates.template_id == x, "name"].iloc[0]
        )
        runs_exist = not st.session_state.runs[st.session_state.runs["template_id"] == pick].empty
        if runs_exist:
            st.warning("You cannot delete this template because runs exist under it. Delete runs first.")
        else:
            if st.button("Delete selected template"):
                st.session_state.templates = st.session_state.templates[st.session_state.templates["template_id"] != pick].reset_index(drop=True)
                st.success("Template deleted.")

# --------------------------
# Create Run
# --------------------------
def create_run_page():
    st.subheader("Create a Run (Instance)")
    st.caption("A run is one deployment of a template (new expiry / new entry / new margin).")

    if st.session_state.templates.empty:
        st.info("Create a template first.")
        return

    tid = st.selectbox(
        "Pick template",
        st.session_state.templates["template_id"].tolist(),
        format_func=lambda x: st.session_state.templates.loc[st.session_state.templates.template_id == x, "name"].iloc[0]
    )
    t = st.session_state.templates[st.session_state.templates["template_id"] == tid].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    run_name = c1.text_input("Run name (optional)", placeholder="Week 1 | conservative entry")
    start_date = c2.date_input("Start date", value=date.today())
    expiry = c3.text_input("Expiry", placeholder="06-Feb-2026")
    status = c4.selectbox("Status", ["ACTIVE", "CLOSED"], index=0)

    c5, c6, c7, c8 = st.columns(4)
    entry_spot = c5.number_input("NIFTY at entry", step=1.0)
    margin_used = c6.number_input("Margin used (INR)", min_value=0.0, step=1000.0)
    qty_lots = c7.number_input("Quantity (lots)", min_value=1, value=1, step=1)
    lot_size = c8.number_input("Lot size", min_value=1, value=int(t["default_lot_size"]), step=1)

    st.markdown("### Upload payoff / graph screenshot (for this run)")
    shot = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if st.button("Create run"):
        if margin_used <= 0:
            st.error("Margin used is required (must be > 0).")
            return
        rid = now_id("RUN")
        row = {
            "run_id": rid,
            "template_id": tid,
            "run_name": run_name.strip(),
            "start_date": str(start_date),
            "expiry": expiry.strip(),
            "entry_spot": float(entry_spot),
            "margin_used": float(margin_used),
            "qty_lots": int(qty_lots),
            "lot_size": int(lot_size),
            "screenshot_b64": img_to_b64(shot),
            "status": status
        }
        st.session_state.runs = pd.concat([st.session_state.runs, pd.DataFrame([row])], ignore_index=True)
        st.success("Run created. Open it in Run Desk.")

# --------------------------
# Analytics calculations
# --------------------------
def run_logs_df(run_id: str) -> pd.DataFrame:
    df = st.session_state.logs[st.session_state.logs["run_id"] == run_id].copy()
    if df.empty:
        return df
    df["date"] = df["date"].apply(safe_dt)
    df = df.dropna(subset=["date"]).sort_values("date")
    df["mtm_inr"] = pd.to_numeric(df["mtm_inr"], errors="coerce").fillna(0.0)
    df["nifty_close"] = pd.to_numeric(df["nifty_close"], errors="coerce")
    df["daily_change"] = df["mtm_inr"].diff()
    df["cum_max"] = df["mtm_inr"].cummax()
    df["drawdown"] = df["mtm_inr"] - df["cum_max"]
    return df

def run_kpis(run_row: pd.Series, df: pd.DataFrame) -> dict:
    margin = float(run_row["margin_used"]) if pd.notna(run_row["margin_used"]) else 0.0
    if df is None or df.empty:
        return {
            "days_logged": 0,
            "latest_mtm": 0.0,
            "profit_pct": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "max_drawdown": 0.0
        }
    latest = float(df["mtm_inr"].iloc[-1])
    profit_pct = (latest / margin * 100.0) if margin else 0.0
    best_day = float(df["daily_change"].fillna(0.0).max())
    worst_day = float(df["daily_change"].fillna(0.0).min())
    max_dd = float(df["drawdown"].min()) if "drawdown" in df.columns else 0.0
    return {
        "days_logged": int(df.shape[0]),
        "latest_mtm": latest,
        "profit_pct": profit_pct,
        "best_day": best_day,
        "worst_day": worst_day,
        "max_drawdown": max_dd
    }

# --------------------------
# Dashboard (main page = analytics)
# --------------------------
def dashboard_page():
    st.subheader("Dashboard")
    templates = st.session_state.templates.copy()
    runs = st.session_state.runs.copy()
    logs = st.session_state.logs.copy()

    # Basic KPIs
    total_templates = templates.shape[0]
    total_runs = runs.shape[0]
    active_runs = runs[runs["status"] == "ACTIVE"].shape[0] if not runs.empty else 0
    total_logs = logs.shape[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Templates</div><div class='kpi-value'>{total_templates}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Runs</div><div class='kpi-value'>{total_runs}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Active runs</div><div class='kpi-value'>{active_runs}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='kpi-title'>EOD logs</div><div class='kpi-value'>{total_logs}</div></div>", unsafe_allow_html=True)

    st.markdown("hr", unsafe_allow_html=True)

    if runs.empty:
        st.info("Create a template, then create a run. Your dashboard will populate automatically.")
        return

    # Compute latest MTM + profit % per run
    rows = []
    for _, r in runs.iterrows():
        df = run_logs_df(r["run_id"])
        k = run_kpis(r, df)
        rows.append({
            "run_id": r["run_id"],
            "template_id": r["template_id"],
            "status": r["status"],
            "expiry": r["expiry"],
            "margin_used": r["margin_used"],
            "latest_mtm": k["latest_mtm"],
            "profit_pct": k["profit_pct"],
            "days_logged": k["days_logged"],
            "max_drawdown": k["max_drawdown"],
        })
    run_perf = pd.DataFrame(rows)

    run_perf = run_perf.merge(
        templates[["template_id","name","strategy_type","underlying"]],
        on="template_id",
        how="left"
    )
    run_perf["run_label"] = run_perf["name"] + " | Exp: " + run_perf["expiry"].fillna("-") + " | " + run_perf["status"]

    # Today's missing logs (for active runs)
    today_str = str(date.today())
    logged_today = set(logs[logs["date"] == today_str]["run_id"].tolist()) if not logs.empty else set()
    active_run_ids = set(runs[runs["status"] == "ACTIVE"]["run_id"].tolist())
    missing_today = sorted(list(active_run_ids - logged_today))

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.markdown("### Top runs (by Profit %)")
        top = run_perf.sort_values("profit_pct", ascending=False)[
            ["run_label","latest_mtm","margin_used","profit_pct","days_logged","max_drawdown","run_id"]
        ].head(12)
        st.dataframe(top, use_container_width=True, height=420)

    with c2:
        st.markdown("### Active runs missing today's log")
        if not missing_today:
            st.success("All active runs have a log for today.")
        else:
            miss_df = runs[runs["run_id"].isin(missing_today)].merge(
                templates[["template_id","name"]],
                on="template_id",
                how="left"
            )
            miss_df["label"] = miss_df["name"] + " | Exp: " + miss_df["expiry"].fillna("-")
            st.dataframe(miss_df[["label","run_id"]], use_container_width=True, height=240)

        st.markdown("hr", unsafe_allow_html=True)
        st.markdown("### Portfolio curve (sum of MTM across runs)")
        if logs.empty:
            st.info("No logs yet.")
        else:
            tmp = logs.copy()
            tmp["date"] = tmp["date"].apply(safe_dt)
            tmp = tmp.dropna(subset=["date"]).sort_values("date")
            tmp["mtm_inr"] = pd.to_numeric(tmp["mtm_inr"], errors="coerce").fillna(0.0)
            # sum of MTM snapshots by date across runs (not perfect equity curve, but useful)
            curve = tmp.groupby(tmp["date"].dt.date)["mtm_inr"].sum().reset_index()
            curve["date"] = pd.to_datetime(curve["date"])
            st.line_chart(curve.set_index("date")["mtm_inr"], height=220)

# --------------------------
# Run Desk (day-wise record + deeper analysis)
# --------------------------
def run_desk_page():
    st.subheader("Run Desk")
    runs = st.session_state.runs.copy()
    templates = st.session_state.templates.copy()

    if runs.empty:
        st.info("No runs yet. Create a template, then create a run.")
        return

    # Pick run
    runs = runs.merge(templates[["template_id","name"]], on="template_id", how="left")
    runs["label"] = runs["name"] + " | Exp: " + runs["expiry"].fillna("-") + " | " + runs["status"]

    pick = st.selectbox("Select run", runs["run_id"].tolist(), format_func=lambda rid: runs.loc[runs.run_id == rid, "label"].iloc[0])
    r = st.session_state.runs[st.session_state.runs["run_id"] == pick].iloc[0]
    t = st.session_state.templates[st.session_state.templates["template_id"] == r["template_id"]].iloc[0]

    df = run_logs_df(pick)
    k = run_kpis(r, df)

    # KPIs
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.markdown(f"<div class='kpi'><div class='kpi-title'>Days logged</div><div class='kpi-value'>{k['days_logged']}</div></div>", unsafe_allow_html=True)
    a2.markdown(f"<div class='kpi'><div class='kpi-title'>Latest MTM</div><div class='kpi-value'>{k['latest_mtm']:,.0f}</div></div>", unsafe_allow_html=True)
    a3.markdown(f"<div class='kpi'><div class='kpi-title'>Profit %</div><div class='kpi-value'>{k['profit_pct']:.2f}%</div></div>", unsafe_allow_html=True)
    a4.markdown(f"<div class='kpi'><div class='kpi-title'>Best day change</div><div class='kpi-value'>{k['best_day']:,.0f}</div></div>", unsafe_allow_html=True)
    a5.markdown(f"<div class='kpi'><div class='kpi-title'>Worst day change</div><div class='kpi-value'>{k['worst_day']:,.0f}</div></div>", unsafe_allow_html=True)
    a6.markdown(f"<div class='kpi'><div class='kpi-title'>Max drawdown</div><div class='kpi-value'>{k['max_drawdown']:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("hr", unsafe_allow_html=True)

    left, mid, right = st.columns([1.0, 1.1, 0.9])

    with left:
        st.markdown("### Screenshot")
        img = b64_to_img(str(r.get("screenshot_b64","")))
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.caption("No screenshot for this run.")
        new_shot = st.file_uploader("Upload/replace screenshot", type=["png","jpg","jpeg"], key="run_shot")
        if new_shot is not None and st.button("Save screenshot"):
            st.session_state.runs.loc[st.session_state.runs["run_id"] == pick, "screenshot_b64"] = img_to_b64(new_shot)
            st.success("Screenshot updated (re-open run to refresh).")

    with mid:
        st.markdown("### Run details")
        st.write({
            "Template": t["name"],
            "Type": t["strategy_type"],
            "Underlying": t["underlying"],
            "Run name": r["run_name"],
            "Start date": r["start_date"],
            "Expiry": r["expiry"],
            "Entry spot": r["entry_spot"],
            "Margin used": r["margin_used"],
            "Lots": r["qty_lots"],
            "Lot size": r["lot_size"],
            "Status": r["status"]
        })

        st.markdown("#### Template plan")
        st.write("Legs / setup:")
        st.write(t["legs_desc"] if str(t["legs_desc"]).strip() else "-")
        st.write("Entry logic:")
        st.write(t["entry_reason"] if str(t["entry_reason"]).strip() else "-")
        st.write("Risk rules:")
        st.write(t["risk_rules"] if str(t["risk_rules"]).strip() else "-")
        st.write("Exit plan:")
        st.write(t["exit_plan"] if str(t["exit_plan"]).strip() else "-")

    with right:
        st.markdown("### Add EOD log (day-wise record)")
        log_date = st.date_input("Log date", value=date.today())
        mtm = st.number_input("MTM since entry (INR)", step=100.0)
        nifty_close = st.number_input("NIFTY close (INR)", step=1.0)
        notes = st.text_area("Notes", height=120, placeholder="Adjustments, breaches, IV changes, lessons...")

        # Prevent duplicate date logs for same run
        existing = st.session_state.logs[
            (st.session_state.logs["run_id"] == pick) &
            (st.session_state.logs["date"] == str(log_date))
        ]
        if st.button("Save log"):
            if not existing.empty:
                st.error("You already logged this date for this run. Delete that log first if you want to replace.")
            else:
                row = {
                    "log_id": now_id("LOG"),
                    "run_id": pick,
                    "date": str(log_date),
                    "mtm_inr": float(mtm),
                    "nifty_close": float(nifty_close),
                    "notes": notes.strip()
                }
                st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([row])], ignore_index=True)
                st.success("Saved.")

        st.markdown("hr", unsafe_allow_html=True)
        st.markdown("### Danger zone")
        if st.button("Delete this run (and its logs)"):
            st.session_state.runs = st.session_state.runs[st.session_state.runs["run_id"] != pick].reset_index(drop=True)
            st.session_state.logs = st.session_state.logs[st.session_state.logs["run_id"] != pick].reset_index(drop=True)
            st.warning("Run deleted.")
            st.stop()

    st.markdown("hr", unsafe_allow_html=True)

    # Day-wise analytics & tables
    st.markdown("### Day-wise record")
    if df.empty:
        st.info("No logs yet for this run.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### MTM timeline")
        st.line_chart(df.set_index("date")["mtm_inr"], height=240)
    with c2:
        st.markdown("#### NIFTY close timeline")
        if df["nifty_close"].notna().any():
            st.line_chart(df.set_index("date")["nifty_close"], height=240)
        else:
            st.info("No NIFTY close values logged yet.")

    st.markdown("#### Extra analysis (useful)")
    # Simple, meaningful stuff:
    # - daily change distribution
    # - drawdown chart
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("Daily change (MTM day-to-day)")
        st.line_chart(df.set_index("date")["daily_change"], height=220)
    with a2:
        st.markdown("Drawdown (MTM below peak)")
        st.line_chart(df.set_index("date")["drawdown"], height=220)

    st.markdown("### Logs table (editable via delete)")
    show = df.copy()
    show["date"] = show["date"].dt.date
    show = show[["date","mtm_inr","daily_change","nifty_close","drawdown","notes"]].rename(columns={
        "mtm_inr":"MTM",
        "daily_change":"Day change",
        "nifty_close":"NIFTY close",
        "drawdown":"Drawdown",
        "notes":"Notes"
    })
    st.dataframe(show, use_container_width=True, height=320)

    st.markdown("#### Delete a log entry")
    # show log_id list for this run
    id_df = st.session_state.logs[st.session_state.logs["run_id"] == pick][["date","log_id"]].copy()
    st.dataframe(id_df, use_container_width=True, height=160)
    del_id = st.text_input("Paste log_id to delete")
    if st.button("Delete log"):
        before = st.session_state.logs.shape[0]
        st.session_state.logs = st.session_state.logs[st.session_state.logs["log_id"] != del_id.strip()].reset_index(drop=True)
        after = st.session_state.logs.shape[0]
        if after < before:
            st.success("Deleted.")
        else:
            st.warning("No matching log_id found.")

# --------------------------
# Strategy Analytics (aggregate across runs)
# --------------------------
def strategy_analytics_page():
    st.subheader("Strategy Analytics")
    templates = st.session_state.templates.copy()
    runs = st.session_state.runs.copy()

    if templates.empty:
        st.info("No templates yet.")
        return

    tid = st.selectbox(
        "Select template",
        templates["template_id"].tolist(),
        format_func=lambda x: templates.loc[templates.template_id == x, "name"].iloc[0]
    )
    t = templates[templates["template_id"] == tid].iloc[0]
    runs_t = runs[runs["template_id"] == tid].copy()

    if runs_t.empty:
        st.info("No runs under this template yet.")
        return

    # Aggregate per run
    rows = []
    for _, r in runs_t.iterrows():
        df = run_logs_df(r["run_id"])
        k = run_kpis(r, df)
        rows.append({
            "run_id": r["run_id"],
            "run_name": r["run_name"],
            "expiry": r["expiry"],
            "status": r["status"],
            "margin_used": r["margin_used"],
            "latest_mtm": k["latest_mtm"],
            "profit_pct": k["profit_pct"],
            "days_logged": k["days_logged"],
            "max_drawdown": k["max_drawdown"]
        })
    perf = pd.DataFrame(rows).sort_values("profit_pct", ascending=False)

    # KPIs at template level
    total_runs = perf.shape[0]
    avg_profit = float(perf["profit_pct"].mean()) if total_runs else 0.0
    best_run = float(perf["profit_pct"].max()) if total_runs else 0.0
    worst_run = float(perf["profit_pct"].min()) if total_runs else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Runs</div><div class='kpi-value'>{total_runs}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Avg Profit %</div><div class='kpi-value'>{avg_profit:.2f}%</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Best Run %</div><div class='kpi-value'>{best_run:.2f}%</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='kpi-title'>Worst Run %</div><div class='kpi-value'>{worst_run:.2f}%</div></div>", unsafe_allow_html=True)

    st.markdown("hr", unsafe_allow_html=True)

    st.markdown("### Template details")
    st.write({
        "Template": t["name"],
        "Type": t["strategy_type"],
        "Underlying": t["underlying"]
    })
    st.write("Legs / setup:")
    st.write(t["legs_desc"] if str(t["legs_desc"]).strip() else "-")

    st.markdown("### Runs leaderboard for this template")
    st.dataframe(
        perf[["run_name","expiry","status","latest_mtm","margin_used","profit_pct","days_logged","max_drawdown","run_id"]],
        use_container_width=True,
        height=420
    )

# --------------------------
# Page routing
# --------------------------
if page == "Dashboard":
    dashboard_page()
elif page == "Run Desk":
    run_desk_page()
elif page == "Strategy Analytics":
    strategy_analytics_page()
elif page == "Create Template":
    create_template_page()
elif page == "Create Run":
    create_run_page()
else:
    backup_restore_page()
