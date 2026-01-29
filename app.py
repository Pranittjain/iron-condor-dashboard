import base64
from io import BytesIO
from datetime import date, datetime

import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------
# Config + style (NO <hr>)
# --------------------------
st.set_page_config(page_title="Options Strategy Journal", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.kpi {
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.04);
}
.kpi-title { font-size: 0.85rem; color: rgba(255,255,255,0.65); margin-bottom: 4px; }
.kpi-value { font-size: 1.4rem; font-weight: 800; }
.small { font-size: 0.85rem; color: rgba(255,255,255,0.65); }
</style>
""", unsafe_allow_html=True)

st.title("Options Strategy Journal (EOD)")
st.caption("Template → Runs → Daily Logs. Built for repeatable strategies, day-wise records, screenshots, analytics.")

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
    if df is None:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

# --------------------------
# Data Model (NEW)
# --------------------------
TEMPLATE_COLS = [
    "template_id",
    "template_name",
    "strategy_type",
    "underlying",
    "rules_notes",
]

RUN_COLS = [
    "run_id",
    "template_id",
    "run_name",
    "status",                 # ACTIVE / CLOSED

    "entry_date",
    "expiry",
    "entry_spot",             # NIFTY at entry

    "qty_lots",
    "lot_size",
    "margin_used",

    "short_put",
    "long_put",
    "short_call",
    "long_call",

    "screenshot_b64",
]

LOG_COLS = [
    "log_id",
    "run_id",
    "date",
    "mtm_inr",                # MTM since entry (TOTAL)
    "nifty_level",            # NIFTY close that day
    "notes",
]

# --------------------------
# State init
# --------------------------
if "templates" not in st.session_state or st.session_state.templates is None:
    st.session_state.templates = pd.DataFrame(columns=TEMPLATE_COLS)
if "runs" not in st.session_state or st.session_state.runs is None:
    st.session_state.runs = pd.DataFrame(columns=RUN_COLS)
if "logs" not in st.session_state or st.session_state.logs is None:
    st.session_state.logs = pd.DataFrame(columns=LOG_COLS)

# --------------------------
# Backward compatibility / schema normalize
# Prevent KeyError forever (handles old CSVs)
# --------------------------
def normalize_state():
    # Convert older column names if they exist
    logs = st.session_state.logs.copy()

    # old: strategy_id -> run_id
    if "strategy_id" in logs.columns and "run_id" not in logs.columns:
        logs = logs.rename(columns={"strategy_id": "run_id"})

    # old: day_pnl_inr -> mtm_inr
    if "day_pnl_inr" in logs.columns and "mtm_inr" not in logs.columns:
        logs = logs.rename(columns={"day_pnl_inr": "mtm_inr"})

    # old: spot_close -> nifty_level
    if "spot_close" in logs.columns and "nifty_level" not in logs.columns:
        logs = logs.rename(columns={"spot_close": "nifty_level"})

    st.session_state.logs = logs

    # Ensure required schemas
    st.session_state.templates = ensure_cols(st.session_state.templates, TEMPLATE_COLS)
    st.session_state.runs = ensure_cols(st.session_state.runs, RUN_COLS)
    st.session_state.logs = ensure_cols(st.session_state.logs, LOG_COLS)

normalize_state()

# --------------------------
# Analytics helpers
# --------------------------
def run_logs_df(run_id: str) -> pd.DataFrame:
    logs = st.session_state.logs

    # Hard guard
    if logs is None or logs.empty or "run_id" not in logs.columns:
        return pd.DataFrame(columns=LOG_COLS)

    df = logs[logs["run_id"] == run_id].copy()
    if df.empty:
        return df

    df["date"] = df["date"].apply(safe_dt)
    df = df.dropna(subset=["date"]).sort_values("date")

    df["mtm_inr"] = pd.to_numeric(df["mtm_inr"], errors="coerce").fillna(0.0)
    df["nifty_level"] = pd.to_numeric(df["nifty_level"], errors="coerce")

    df["day_change"] = df["mtm_inr"].diff()
    df["peak"] = df["mtm_inr"].cummax()
    df["drawdown"] = df["mtm_inr"] - df["peak"]
    return df

def run_kpis(run_row: pd.Series, df: pd.DataFrame) -> dict:
    margin = float(run_row.get("margin_used", 0) or 0)
    if df is None or df.empty:
        return dict(days=0, latest_mtm=0.0, profit_pct=0.0, best_change=0.0, worst_change=0.0, max_dd=0.0)

    latest = float(df["mtm_inr"].iloc[-1])
    profit_pct = (latest / margin * 100.0) if margin else 0.0
    best_change = float(df["day_change"].fillna(0.0).max())
    worst_change = float(df["day_change"].fillna(0.0).min())
    max_dd = float(df["drawdown"].min()) if "drawdown" in df.columns else 0.0

    return dict(days=int(df.shape[0]), latest_mtm=latest, profit_pct=profit_pct,
                best_change=best_change, worst_change=worst_change, max_dd=max_dd)

# --------------------------
# Sidebar Navigation
# --------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Run Desk", "Create Template", "Create Run", "Backup / Restore"],
        index=0
    )
    st.caption("Use Backup/Restore so your data persists across sessions.")

# --------------------------
# Pages
# --------------------------
def page_backup_restore():
    st.subheader("Backup / Restore")

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download templates.csv",
                       st.session_state.templates.to_csv(index=False).encode("utf-8"),
                       file_name="templates.csv")
    c2.download_button("Download runs.csv",
                       st.session_state.runs.to_csv(index=False).encode("utf-8"),
                       file_name="runs.csv")
    c3.download_button("Download logs.csv",
                       st.session_state.logs.to_csv(index=False).encode("utf-8"),
                       file_name="logs.csv")

    st.markdown("### Import")
    u1, u2, u3 = st.columns(3)
    up_t = u1.file_uploader("Upload templates.csv", type=["csv"])
    up_r = u2.file_uploader("Upload runs.csv", type=["csv"])
    up_l = u3.file_uploader("Upload logs.csv", type=["csv"])

    if st.button("Import CSVs"):
        if up_t is not None:
            st.session_state.templates = ensure_cols(pd.read_csv(up_t), TEMPLATE_COLS)
        if up_r is not None:
            st.session_state.runs = ensure_cols(pd.read_csv(up_r), RUN_COLS)
        if up_l is not None:
            logs_df = pd.read_csv(up_l)
            st.session_state.logs = logs_df
        normalize_state()
        st.success("Imported.")

    if st.button("Reset everything (in memory)"):
        st.session_state.templates = pd.DataFrame(columns=TEMPLATE_COLS)
        st.session_state.runs = pd.DataFrame(columns=RUN_COLS)
        st.session_state.logs = pd.DataFrame(columns=LOG_COLS)
        st.warning("Reset done.")

def page_create_template():
    st.subheader("Create Template (Strategy model)")
    st.caption("Template = your strategy concept + rules. Runs store expiry/strikes/entry details.")

    c1, c2, c3 = st.columns(3)
    template_name = c1.text_input("Template name", placeholder="IC 300/50 Conservative")
    strategy_type = c2.text_input("Strategy type", value="Iron Condor")
    underlying = c3.text_input("Underlying", value="NIFTY")

    rules_notes = st.text_area("Rules / notes (template)", height=140,
                               placeholder="Entry filter, adjustment rules, stop loss, target, regime notes...")

    if st.button("Add template"):
        if not template_name.strip():
            st.error("Template name is required.")
            return
        tid = now_id("TEMPLATE")
        row = dict(
            template_id=tid,
            template_name=template_name.strip(),
            strategy_type=strategy_type.strip(),
            underlying=underlying.strip(),
            rules_notes=rules_notes.strip()
        )
        st.session_state.templates = pd.concat([st.session_state.templates, pd.DataFrame([row])], ignore_index=True)
        st.success("Template created.")

def page_create_run():
    st.subheader("Create Run (Expiry + strikes + entry details + screenshot)")

    if st.session_state.templates.empty:
        st.info("Create a template first.")
        return

    tid = st.selectbox(
        "Pick template",
        st.session_state.templates["template_id"].tolist(),
        format_func=lambda x: st.session_state.templates.loc[
            st.session_state.templates.template_id == x, "template_name"
        ].iloc[0]
    )

    c1, c2, c3 = st.columns(3)
    run_name = c1.text_input("Run name (optional)", placeholder="Week 1 | 10D entry | no event")
    status = c2.selectbox("Status", ["ACTIVE", "CLOSED"], index=0)
    expiry = c3.text_input("Expiry", placeholder="03-Feb-2026")

    c4, c5, c6 = st.columns(3)
    entry_date = c4.date_input("Entry date", value=date.today())
    entry_spot = c5.number_input("NIFTY at entry", step=1.0)
    margin_used = c6.number_input("Margin used (₹)", min_value=0.0, step=1000.0)

    c7, c8 = st.columns(2)
    qty_lots = c7.number_input("Qty (lots)", min_value=1, value=1, step=1)
    lot_size = c8.number_input("Lot size", min_value=1, value=50, step=1)

    st.markdown("### Strikes for this run")
    s1, s2, s3, s4 = st.columns(4)
    short_put = s1.text_input("Short Put", placeholder="e.g., 25000")
    long_put = s2.text_input("Long Put", placeholder="e.g., 24700")
    short_call = s3.text_input("Short Call", placeholder="e.g., 25700")
    long_call = s4.text_input("Long Call", placeholder="e.g., 26000")

    st.markdown("### Screenshot (payoff/graph) for this run")
    shot = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])

    if st.button("Create run"):
        if margin_used <= 0:
            st.error("Margin used must be > 0.")
            return

        rid = now_id("RUN")
        row = dict(
            run_id=rid,
            template_id=tid,
            run_name=run_name.strip(),
            status=status,
            entry_date=str(entry_date),
            expiry=expiry.strip(),
            entry_spot=float(entry_spot),
            qty_lots=int(qty_lots),
            lot_size=int(lot_size),
            margin_used=float(margin_used),
            short_put=short_put.strip(),
            long_put=long_put.strip(),
            short_call=short_call.strip(),
            long_call=long_call.strip(),
            screenshot_b64=img_to_b64(shot),
        )

        st.session_state.runs = pd.concat([st.session_state.runs, pd.DataFrame([row])], ignore_index=True)
        st.success("Run created. Open it in Run Desk.")

def page_dashboard():
    st.subheader("Dashboard (Analytics first)")

    templates = st.session_state.templates.copy()
    runs = st.session_state.runs.copy()
    logs = st.session_state.logs.copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Templates</div><div class='kpi-value'>{templates.shape[0]}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Runs</div><div class='kpi-value'>{runs.shape[0]}</div></div>", unsafe_allow_html=True)
    active_runs = runs[runs["status"] == "ACTIVE"].shape[0] if not runs.empty else 0
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Active runs</div><div class='kpi-value'>{active_runs}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='kpi-title'>EOD logs</div><div class='kpi-value'>{logs.shape[0]}</div></div>", unsafe_allow_html=True)

    if runs.empty:
        st.info("Create a template and a run to start journaling.")
        return

    # Compute performance for each run
    rows = []
    for _, r in runs.iterrows():
        df = run_logs_df(r["run_id"])
        k = run_kpis(r, df)
        rows.append({
            "run_id": r["run_id"],
            "template_id": r["template_id"],
            "status": r["status"],
            "expiry": r["expiry"],
            "entry_date": r["entry_date"],
            "margin_used": r["margin_used"],
            "latest_mtm": k["latest_mtm"],
            "profit_pct": k["profit_pct"],
            "days_logged": k["days"],
            "max_drawdown": k["max_dd"],
        })
    perf = pd.DataFrame(rows)

    perf = perf.merge(
        templates[["template_id", "template_name", "strategy_type", "underlying"]],
        on="template_id",
        how="left"
    )
    perf["label"] = perf["template_name"] + " | Exp: " + perf["expiry"].fillna("-") + " | " + perf["status"]

    c1, c2 = st.columns([0.62, 0.38])
    with c1:
        st.markdown("### Top runs (by Profit %)")
        top = perf.sort_values("profit_pct", ascending=False)[
            ["label", "latest_mtm", "margin_used", "profit_pct", "days_logged", "max_drawdown", "run_id"]
        ].head(12)
        st.dataframe(top, use_container_width=True, height=420)

    with c2:
        st.markdown("### Active runs missing today's log")
        today_str = str(date.today())
        logged_today = set(logs[logs["date"] == today_str]["run_id"].tolist()) if not logs.empty else set()
        active_ids = set(runs[runs["status"] == "ACTIVE"]["run_id"].tolist())
        missing = sorted(list(active_ids - logged_today))

        if not missing:
            st.success("All active runs have a log for today.")
        else:
            miss = runs[runs["run_id"].isin(missing)].merge(
                templates[["template_id", "template_name"]],
                on="template_id",
                how="left"
            )
            miss["label"] = miss["template_name"] + " | Exp: " + miss["expiry"].fillna("-")
            st.dataframe(miss[["label", "run_id"]], use_container_width=True, height=280)

def page_run_desk():
    st.subheader("Run Desk (Open a run → see everything + day-wise record)")

    if st.session_state.runs.empty:
        st.info("No runs yet. Create a run first.")
        return

    runs = st.session_state.runs.copy()
    templates = st.session_state.templates.copy()

    # join template names for labels
    runs = runs.merge(
        templates[["template_id", "template_name", "strategy_type", "underlying", "rules_notes"]],
        on="template_id",
        how="left"
    )
    runs["label"] = runs["template_name"] + " | Exp: " + runs["expiry"].fillna("-") + " | " + runs["status"]

    pick = st.selectbox("Select run", runs["run_id"].tolist(),
                        format_func=lambda rid: runs.loc[runs.run_id == rid, "label"].iloc[0])

    r = runs[runs["run_id"] == pick].iloc[0]
    df = run_logs_df(pick)
    k = run_kpis(r, df)

    # KPIs
    a1, a2, a3, a4 = st.columns(4)
    a1.markdown(f"<div class='kpi'><div class='kpi-title'>Days logged</div><div class='kpi-value'>{k['days']}</div></div>", unsafe_allow_html=True)
    a2.markdown(f"<div class='kpi'><div class='kpi-title'>Latest MTM</div><div class='kpi-value'>₹{k['latest_mtm']:,.0f}</div></div>", unsafe_allow_html=True)
    a3.markdown(f"<div class='kpi'><div class='kpi-title'>Profit % (MTM/Margin)</div><div class='kpi-value'>{k['profit_pct']:.2f}%</div></div>", unsafe_allow_html=True)
    a4.markdown(f"<div class='kpi'><div class='kpi-title'>Max Drawdown</div><div class='kpi-value'>₹{k['max_dd']:,.0f}</div></div>", unsafe_allow_html=True)

    left, mid, right = st.columns([1.0, 1.2, 0.9])

    with left:
        st.markdown("### Screenshot")
        img = b64_to_img(r.get("screenshot_b64", ""))
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.caption("No screenshot yet for this run.")

        new_shot = st.file_uploader("Upload/replace screenshot", type=["png", "jpg", "jpeg"], key="replace_run_shot")
        if new_shot is not None and st.button("Save screenshot"):
            st.session_state.runs.loc[st.session_state.runs["run_id"] == pick, "screenshot_b64"] = img_to_b64(new_shot)
            st.success("Screenshot updated. Re-select the run to refresh.")

    with mid:
        st.markdown("### Run details")
        st.write({
            "Template": r["template_name"],
            "Type": r["strategy_type"],
            "Underlying": r["underlying"],
            "Run name": r["run_name"],
            "Status": r["status"],
            "Entry date": r["entry_date"],
            "Expiry": r["expiry"],
            "Entry spot (NIFTY)": r["entry_spot"],
            "Margin used (₹)": r["margin_used"],
            "Lots": r["qty_lots"],
            "Lot size": r["lot_size"],
        })

        st.markdown("### Strikes")
        st.write({
            "Short Put": r["short_put"],
            "Long Put": r["long_put"],
            "Short Call": r["short_call"],
            "Long Call": r["long_call"],
        })

        st.markdown("### Template rules / notes")
        st.write(r["rules_notes"] if str(r["rules_notes"]).strip() else "-")

    with right:
        st.markdown("### Add Daily Log (day-wise record)")
        log_date = st.date_input("Log date", value=date.today())
        mtm = st.number_input("MTM since entry (₹)", step=100.0)
        nifty_level = st.number_input("NIFTY level for the day (close)", step=1.0)
        notes = st.text_area("Notes", height=120, placeholder="Adjustments, breaches, IV changes, lessons...")

        existing = st.session_state.logs[
            (st.session_state.logs["run_id"] == pick) &
            (st.session_state.logs["date"] == str(log_date))
        ]

        if st.button("Save log"):
            if not existing.empty:
                st.error("You already logged this date for this run. Delete that log if you want to replace.")
            else:
                row = dict(
                    log_id=now_id("LOG"),
                    run_id=pick,
                    date=str(log_date),
                    mtm_inr=float(mtm),
                    nifty_level=float(nifty_level),
                    notes=notes.strip()
                )
                st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([row])], ignore_index=True)
                st.success("Saved.")

        st.markdown("### Delete")
        if st.button("Delete this run (and its logs)"):
            st.session_state.runs = st.session_state.runs[st.session_state.runs["run_id"] != pick].reset_index(drop=True)
            st.session_state.logs = st.session_state.logs[st.session_state.logs["run_id"] != pick].reset_index(drop=True)
            st.warning("Run deleted.")
            st.stop()

    st.markdown("### Day-wise charts")
    if df.empty:
        st.info("No logs yet for this run.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("MTM timeline (₹)")
        st.line_chart(df.set_index("date")["mtm_inr"], height=240)
    with c2:
        st.markdown("NIFTY timeline")
        if df["nifty_level"].notna().any():
            st.line_chart(df.set_index("date")["nifty_level"], height=240)
        else:
            st.info("No NIFTY values logged yet.")

    st.markdown("### Extra analysis")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("Day-to-day MTM change")
        st.line_chart(df.set_index("date")["day_change"], height=220)
    with c4:
        st.markdown("Drawdown (MTM below peak)")
        st.line_chart(df.set_index("date")["drawdown"], height=220)

    st.markdown("### Logs table")
    table = df.copy()
    table["date"] = table["date"].dt.date
    show = table[["date", "mtm_inr", "day_change", "nifty_level", "drawdown", "notes"]].rename(columns={
        "mtm_inr": "MTM (₹)",
        "day_change": "Day change (₹)",
        "nifty_level": "NIFTY level",
        "drawdown": "Drawdown (₹)",
        "notes": "Notes"
    })
    st.dataframe(show, use_container_width=True, height=320)

    st.markdown("### Delete a log entry")
    id_df = st.session_state.logs[st.session_state.logs["run_id"] == pick][["date", "log_id"]].copy()
    st.dataframe(id_df, use_container_width=True, height=180)
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
# Router
# --------------------------
if page == "Dashboard":
    page_dashboard()
elif page == "Run Desk":
    page_run_desk()
elif page == "Create Template":
    page_create_template()
elif page == "Create Run":
    page_create_run()
else:
    page_backup_restore()
