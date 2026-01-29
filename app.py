import base64
from io import BytesIO
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image


# --------------------------
# Config + style
# --------------------------
st.set_page_config(page_title="Options Strategy Journal", layout="wide")

st.markdown(
    """
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
hr { border: none; height: 1px; background: rgba(255,255,255,0.1); margin: 16px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Options Strategy Journal (EOD)")
st.caption("Persistent storage: auto-load + auto-save to data/*.csv (local persistent; cloud depends on host)")

# --------------------------
# Data Model
# --------------------------
TEMPLATE_COLS = ["template_id", "template_name", "strategy_type", "underlying", "rules_notes"]

RUN_COLS = [
    "run_id", "template_id", "run_name", "status",
    "entry_date", "expiry", "entry_spot",
    "qty_lots", "lot_size", "margin_used",
    "short_put", "long_put", "short_call", "long_call",
    "screenshot_b64",
    # close / square-off
    "close_date", "close_spot", "close_mtm_inr", "close_notes"
]

LOG_COLS = ["log_id", "run_id", "date", "mtm_inr", "nifty_level", "notes"]

# --------------------------
# Storage (auto-save)
# --------------------------
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = APP_DIR / "data"
TEMPLATES_PATH = DATA_DIR / "templates.csv"
RUNS_PATH = DATA_DIR / "runs.csv"
LOGS_PATH = DATA_DIR / "logs.csv"

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, dtype=str)  # keep raw strings; we cast later
        return ensure_cols(df, cols)
    except Exception:
        return pd.DataFrame(columns=cols)

def atomic_write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def save_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(st.session_state.templates, TEMPLATES_PATH)
    atomic_write_csv(st.session_state.runs, RUNS_PATH)
    atomic_write_csv(st.session_state.logs, LOGS_PATH)

# --------------------------
# Helpers
# --------------------------
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"

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

def to_date_str(d: date) -> str:
    return d.isoformat()

def parse_date(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")

def to_float(x, default=0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def to_int(x, default=0) -> int:
    try:
        if x is None or str(x).strip() == "":
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)

# --------------------------
# Init session state (LOAD ON START)
# --------------------------
if "booted" not in st.session_state:
    st.session_state.templates = load_csv(TEMPLATES_PATH, TEMPLATE_COLS)
    st.session_state.runs = load_csv(RUNS_PATH, RUN_COLS)
    st.session_state.logs = load_csv(LOGS_PATH, LOG_COLS)
    st.session_state.booted = True

# --------------------------
# Normalize + backward compatibility
# --------------------------
def normalize_state():
    st.session_state.templates = ensure_cols(st.session_state.templates, TEMPLATE_COLS)
    st.session_state.runs = ensure_cols(st.session_state.runs, RUN_COLS)

    logs = st.session_state.logs.copy()
    # old schema renames
    if "strategy_id" in logs.columns and "run_id" not in logs.columns:
        logs = logs.rename(columns={"strategy_id": "run_id"})
    if "day_pnl_inr" in logs.columns and "mtm_inr" not in logs.columns:
        logs = logs.rename(columns={"day_pnl_inr": "mtm_inr"})
    if "spot_close" in logs.columns and "nifty_level" not in logs.columns:
        logs = logs.rename(columns={"spot_close": "nifty_level"})

    st.session_state.logs = ensure_cols(logs, LOG_COLS)

    # date normalization to ISO strings (prevents duplicate-check bugs)
    if not st.session_state.logs.empty:
        st.session_state.logs["date"] = (
            pd.to_datetime(st.session_state.logs["date"], errors="coerce")
            .dt.date
            .astype(str)
        )

normalize_state()

# --------------------------
# Analytics
# --------------------------
def run_logs_df(run_id: str) -> pd.DataFrame:
    logs = st.session_state.logs
    if logs is None or logs.empty:
        return pd.DataFrame(columns=LOG_COLS)

    df = logs[logs["run_id"] == run_id].copy()
    if df.empty:
        return df

    df["date_ts"] = parse_date(df["date"])
    df = df.dropna(subset=["date_ts"]).sort_values("date_ts")

    df["mtm_inr"] = pd.to_numeric(df["mtm_inr"], errors="coerce").fillna(0.0)
    df["nifty_level"] = pd.to_numeric(df["nifty_level"], errors="coerce")

    df["day_change"] = df["mtm_inr"].diff().fillna(df["mtm_inr"])
    df["peak"] = df["mtm_inr"].cummax()
    df["drawdown"] = df["mtm_inr"] - df["peak"]
    return df

def run_kpis(run_row: pd.Series, df: pd.DataFrame) -> dict:
    margin = to_float(run_row.get("margin_used", 0), 0.0)
    if df is None or df.empty:
        return dict(days=0, latest_mtm=0.0, profit_pct=0.0, max_dd=0.0)
    latest = float(df["mtm_inr"].iloc[-1])
    profit_pct = (latest / margin * 100.0) if margin else 0.0
    max_dd = float(df["drawdown"].min()) if "drawdown" in df.columns else 0.0
    return dict(days=int(df.shape[0]), latest_mtm=latest, profit_pct=profit_pct, max_dd=max_dd)

def line_chart(df: pd.DataFrame, x: str, y: str, height: int = 260, title: str = ""):
    if df.empty:
        st.info("No data.")
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            tooltip=[x, y],
        )
        .properties(height=height, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

def bar_chart(df: pd.DataFrame, x: str, y: str, height: int = 260, title: str = ""):
    if df.empty:
        st.info("No data.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            tooltip=[x, y],
        )
        .properties(height=height, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

# --------------------------
# Sidebar Navigation
# --------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Run Desk", "Create Template", "Create Run", "Backup / Restore"],
        index=0,
    )
    st.caption("Auto-saves to data/*.csv")

# --------------------------
# Pages
# --------------------------
def page_dashboard():
    st.subheader("Dashboard (Analytics)")

    templates = st.session_state.templates
    runs = st.session_state.runs
    logs = st.session_state.logs

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Templates</div><div class='kpi-value'>{templates.shape[0]}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Runs</div><div class='kpi-value'>{runs.shape[0]}</div></div>", unsafe_allow_html=True)
    active_runs = runs[runs["status"] == "ACTIVE"].shape[0] if not runs.empty else 0
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Active runs</div><div class='kpi-value'>{active_runs}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='kpi-title'>EOD logs</div><div class='kpi-value'>{logs.shape[0]}</div></div>", unsafe_allow_html=True)

    if runs.empty:
        st.info("Create a template and a run to start journaling.")
        return

    # Build performance table
    rows = []
    for _, r in runs.iterrows():
        df = run_logs_df(r["run_id"])
        k = run_kpis(r, df)
        rows.append({
            "run_id": r["run_id"],
            "expiry": r.get("expiry", ""),
            "status": r.get("status", ""),
            "margin_used": to_float(r.get("margin_used", 0), 0),
            "latest_mtm": k["latest_mtm"],
            "profit_pct": k["profit_pct"],
            "days_logged": k["days"],
            "max_drawdown": k["max_dd"],
            "template_id": r.get("template_id", "")
        })

    perf = pd.DataFrame(rows).merge(
        templates[["template_id", "template_name"]],
        on="template_id",
        how="left",
    )

    perf["label"] = perf["template_name"].fillna("Unknown") + " | Exp: " + perf["expiry"].fillna("-") + " | " + perf["status"].fillna("-")

    st.markdown("### Top runs by Profit %")
    st.dataframe(
        perf.sort_values("profit_pct", ascending=False)[
            ["label", "latest_mtm", "margin_used", "profit_pct", "days_logged", "max_drawdown", "run_id"]
        ].head(20),
        use_container_width=True,
        height=520,
    )

def page_create_template():
    st.subheader("Create Template")

    c1, c2, c3 = st.columns(3)
    template_name = c1.text_input("Template name", placeholder="IC 300/50 Conservative")
    strategy_type = c2.text_input("Strategy type", value="Iron Condor")
    underlying = c3.text_input("Underlying", value="NIFTY")
    rules_notes = st.text_area("Rules / notes", height=140)

    if st.button("Add template"):
        if not template_name.strip():
            st.error("Template name is required.")
            return
        tid = new_id("TEMPLATE")
        row = dict(
            template_id=tid,
            template_name=template_name.strip(),
            strategy_type=strategy_type.strip(),
            underlying=underlying.strip(),
            rules_notes=rules_notes.strip(),
        )
        st.session_state.templates = pd.concat([st.session_state.templates, pd.DataFrame([row])], ignore_index=True)
        save_all()
        st.success("Template created and saved.")

def page_create_run():
    st.subheader("Create Run")

    if st.session_state.templates.empty:
        st.info("Create a template first.")
        return

    tid = st.selectbox(
        "Pick template",
        st.session_state.templates["template_id"].tolist(),
        format_func=lambda x: st.session_state.templates.loc[
            st.session_state.templates.template_id == x, "template_name"
        ].iloc[0],
    )

    c1, c2, c3 = st.columns(3)
    run_name = c1.text_input("Run name (optional)")
    status = c2.selectbox("Status", ["ACTIVE", "CLOSED"], index=0)
    expiry = c3.text_input("Expiry", placeholder="2026-02-03 or 03-Feb-2026")

    c4, c5, c6 = st.columns(3)
    entry_date = c4.date_input("Entry date", value=date.today())
    entry_spot = c5.number_input("NIFTY at entry", step=1.0)
    margin_used = c6.number_input("Margin used (₹)", min_value=0.0, step=1000.0)

    c7, c8 = st.columns(2)
    qty_lots = c7.number_input("Qty (lots)", min_value=1, value=1, step=1)
    lot_size = c8.number_input("Lot size", min_value=1, value=50, step=1)

    st.markdown("### Strikes")
    s1, s2, s3, s4 = st.columns(4)
    short_put = s1.text_input("Short Put")
    long_put = s2.text_input("Long Put")
    short_call = s3.text_input("Short Call")
    long_call = s4.text_input("Long Call")

    shot = st.file_uploader("Upload payoff/graph screenshot", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.markdown("### (Optional) If already closed, add square-off details")
    cc1, cc2, cc3 = st.columns(3)
    close_date = cc1.date_input("Close date", value=date.today())
    close_spot = cc2.number_input("Close spot", step=1.0)
    close_mtm = cc3.number_input("Final MTM (₹) at square-off", step=100.0)
    close_notes = st.text_area("Close notes", height=90)

    if st.button("Create run"):
        if margin_used <= 0:
            st.error("Margin used must be > 0.")
            return
        rid = new_id("RUN")
        row = dict(
            run_id=rid,
            template_id=tid,
            run_name=run_name.strip(),
            status=status,
            entry_date=to_date_str(entry_date),
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
            close_date=to_date_str(close_date) if status == "CLOSED" else "",
            close_spot=float(close_spot) if status == "CLOSED" else "",
            close_mtm_inr=float(close_mtm) if status == "CLOSED" else "",
            close_notes=close_notes.strip() if status == "CLOSED" else "",
        )
        st.session_state.runs = pd.concat([st.session_state.runs, pd.DataFrame([row])], ignore_index=True)
        save_all()
        st.success("Run created and saved.")

def page_run_desk():
    st.subheader("Run Desk")

    if st.session_state.runs.empty:
        st.info("No runs yet.")
        return

    runs = st.session_state.runs.merge(
        st.session_state.templates[["template_id", "template_name", "rules_notes"]],
        on="template_id",
        how="left",
    )
    runs["label"] = runs["template_name"].fillna("Unknown") + " | Exp: " + runs["expiry"].fillna("-") + " | " + runs["status"].fillna("-")

    f1, f2 = st.columns([1, 1])
    with f1:
        status_filter = st.selectbox("Filter runs", ["ALL", "ACTIVE", "CLOSED"], index=0)
    with f2:
        search = st.text_input("Search (name/expiry/template)", value="")

    filtered = runs.copy()
    if status_filter != "ALL":
        filtered = filtered[filtered["status"] == status_filter]
    if search.strip():
        s = search.strip().lower()
        filtered = filtered[
            filtered["label"].str.lower().str.contains(s, na=False)
            | filtered["run_name"].fillna("").str.lower().str.contains(s, na=False)
        ]

    if filtered.empty:
        st.info("No runs match this filter/search.")
        return

    pick = st.selectbox(
        "Select run",
        filtered["run_id"].tolist(),
        format_func=lambda rid: filtered.loc[filtered.run_id == rid, "label"].iloc[0],
    )

    r = runs[runs["run_id"] == pick].iloc[0]
    df = run_logs_df(pick)
    k = run_kpis(r, df)

    a1, a2, a3, a4 = st.columns(4)
    a1.markdown(f"<div class='kpi'><div class='kpi-title'>Days logged</div><div class='kpi-value'>{k['days']}</div></div>", unsafe_allow_html=True)
    a2.markdown(f"<div class='kpi'><div class='kpi-title'>Latest MTM</div><div class='kpi-value'>₹{k['latest_mtm']:,.0f}</div></div>", unsafe_allow_html=True)
    a3.markdown(f"<div class='kpi'><div class='kpi-title'>Profit % (vs margin)</div><div class='kpi-value'>{k['profit_pct']:.2f}%</div></div>", unsafe_allow_html=True)
    a4.markdown(f"<div class='kpi'><div class='kpi-title'>Max Drawdown</div><div class='kpi-value'>₹{k['max_dd']:,.0f}</div></div>", unsafe_allow_html=True)

    left, mid, right = st.columns([1.0, 1.25, 1.05])

    with left:
        st.markdown("### Screenshot")
        img = b64_to_img(r.get("screenshot_b64", ""))
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.caption("No screenshot for this run.")

        new_shot = st.file_uploader("Upload/replace screenshot", type=["png", "jpg", "jpeg"], key="replace")
        if new_shot is not None and st.button("Save screenshot"):
            st.session_state.runs.loc[st.session_state.runs["run_id"] == pick, "screenshot_b64"] = img_to_b64(new_shot)
            save_all()
            st.success("Saved.")

    with mid:
        st.markdown("### Run details")
        st.write({
            "Template": r.get("template_name", ""),
            "Run name": r.get("run_name", ""),
            "Status": r.get("status", ""),
            "Entry date": r.get("entry_date", ""),
            "Expiry": r.get("expiry", ""),
            "Entry spot": to_float(r.get("entry_spot", 0), 0),
            "Margin": to_float(r.get("margin_used", 0), 0),
            "Lots": to_int(r.get("qty_lots", 0), 0),
            "Lot size": to_int(r.get("lot_size", 0), 0),
            "Short Put": r.get("short_put", ""),
            "Long Put": r.get("long_put", ""),
            "Short Call": r.get("short_call", ""),
            "Long Call": r.get("long_call", ""),
        })

        st.markdown("### Rules / notes")
        st.write(r["rules_notes"] if str(r.get("rules_notes", "")).strip() else "-")

        st.markdown("---")
        st.markdown("### Close / Square-off")
        st.caption("If you close the strategy, store the final square-off snapshot here.")
        cd1, cd2, cd3 = st.columns(3)
        close_date = cd1.date_input("Close date", value=date.today(), key="close_date")
        close_spot = cd2.number_input("Close spot", step=1.0, key="close_spot")
        close_mtm = cd3.number_input("Final MTM (₹) at square-off", step=100.0, key="close_mtm")
        close_notes = st.text_area("Close notes", height=90, key="close_notes")

        cbtn1, cbtn2 = st.columns(2)
        with cbtn1:
            if st.button("Mark as CLOSED (save square-off)"):
                idx = st.session_state.runs["run_id"] == pick
                st.session_state.runs.loc[idx, "status"] = "CLOSED"
                st.session_state.runs.loc[idx, "close_date"] = to_date_str(close_date)
                st.session_state.runs.loc[idx, "close_spot"] = str(float(close_spot))
                st.session_state.runs.loc[idx, "close_mtm_inr"] = str(float(close_mtm))
                st.session_state.runs.loc[idx, "close_notes"] = close_notes.strip()
                save_all()
                st.success("Run marked CLOSED and square-off saved.")
        with cbtn2:
            if st.button("Re-open (set ACTIVE)"):
                idx = st.session_state.runs["run_id"] == pick
                st.session_state.runs.loc[idx, "status"] = "ACTIVE"
                save_all()
                st.success("Run set to ACTIVE.")

        if str(r.get("status", "")).upper() == "CLOSED":
            st.markdown("**Saved square-off (last close):**")
            st.write({
                "Close date": r.get("close_date", ""),
                "Close spot": r.get("close_spot", ""),
                "Final MTM (₹)": r.get("close_mtm_inr", ""),
                "Close notes": r.get("close_notes", ""),
            })

    with right:
        st.markdown("### Add Daily Log")
        log_date = st.date_input("Log date", value=date.today(), key="log_date")
        mtm = st.number_input("MTM since entry (₹)", step=100.0, key="mtm")
        nifty_level = st.number_input("NIFTY close", step=1.0, key="nifty_close")
        notes = st.text_area("Notes", height=120, key="notes")

        # robust duplicate check (compare ISO string)
        log_date_str = to_date_str(log_date)
        existing = st.session_state.logs[
            (st.session_state.logs["run_id"] == pick) &
            (st.session_state.logs["date"] == log_date_str)
        ]

        if st.button("Save log"):
            if not existing.empty:
                st.error("This date is already logged for this run. Delete that entry first to replace it.")
            else:
                row = dict(
                    log_id=new_id("LOG"),
                    run_id=pick,
                    date=log_date_str,
                    mtm_inr=str(float(mtm)),
                    nifty_level=str(float(nifty_level)),
                    notes=notes.strip()
                )
                st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([row])], ignore_index=True)
                normalize_state()
                save_all()
                st.success("Saved log.")

        st.markdown("---")
        if st.button("Delete this run (and its logs)"):
            st.session_state.runs = st.session_state.runs[st.session_state.runs["run_id"] != pick].reset_index(drop=True)
            st.session_state.logs = st.session_state.logs[st.session_state.logs["run_id"] != pick].reset_index(drop=True)
            save_all()
            st.warning("Run deleted.")
            st.stop()

    st.markdown("---")
    if df.empty:
        st.info("No logs yet for this run.")
        return

    # Charts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### MTM (Equity Curve)")
        plot_df = df[["date_ts", "mtm_inr"]].rename(columns={"date_ts": "date"})
        line_chart(plot_df, "date:T", "mtm_inr:Q", title="MTM since entry")

    with c2:
        st.markdown("### Daily P&L")
        plot_df = df[["date_ts", "day_change"]].rename(columns={"date_ts": "date"})
        bar_chart(plot_df, "date:T", "day_change:Q", title="Day-to-day change")

    with c3:
        st.markdown("### Drawdown")
        plot_df = df[["date_ts", "drawdown"]].rename(columns={"date_ts": "date"})
        line_chart(plot_df, "date:T", "drawdown:Q", title="Drawdown")

    st.markdown("### NIFTY timeline")
    plot_df = df[["date_ts", "nifty_level"]].rename(columns={"date_ts": "date"})
    line_chart(plot_df, "date:T", "nifty_level:Q", height=240, title="NIFTY close")

    st.markdown("### Logs table")
    table = df.copy()
    table["date"] = table["date_ts"].dt.date
    st.dataframe(
        table[["date", "mtm_inr", "day_change", "nifty_level", "drawdown", "notes"]],
        use_container_width=True,
        height=320,
    )

def page_backup_restore():
    st.subheader("Backup / Restore")
    st.write("Auto-save is enabled. This is just extra backup/manual import.")

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
            st.session_state.templates = ensure_cols(pd.read_csv(up_t, dtype=str), TEMPLATE_COLS)
        if up_r is not None:
            st.session_state.runs = ensure_cols(pd.read_csv(up_r, dtype=str), RUN_COLS)
        if up_l is not None:
            st.session_state.logs = ensure_cols(pd.read_csv(up_l, dtype=str), LOG_COLS)
        normalize_state()
        save_all()
        st.success("Imported + saved to data/.")

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
