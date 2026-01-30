import base64
from io import BytesIO
from datetime import date
from uuid import uuid4

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

from supabase import create_client, Client


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
st.caption("Persistent storage: Supabase Postgres (data) + Supabase Storage (screenshots)")


# --------------------------
# Data Model
# --------------------------
TEMPLATE_COLS = ["template_id", "template_name", "strategy_type", "underlying", "rules_notes"]
RUN_COLS = [
    "run_id", "template_id", "run_name", "status",
    "entry_date", "expiry", "entry_spot",
    "qty_lots", "lot_size", "margin_used",
    "short_put", "long_put", "short_call", "long_call",
    "screenshot_url",
    "close_date", "close_spot", "close_mtm_inr", "close_notes"
]
LOG_COLS = ["log_id", "run_id", "date", "mtm_inr", "nifty_level", "notes"]


# --------------------------
# Supabase init
# --------------------------
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        st.error("Missing Supabase secrets. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets.")
        st.stop()
    return create_client(url, key)

supabase = get_supabase()
BUCKET = st.secrets.get("SUPABASE_BUCKET", "journal-screens")


# --------------------------
# Helpers
# --------------------------
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"

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

def to_date_str(d: date) -> str:
    return d.isoformat()

def parse_date(x):
    return pd.to_datetime(x, errors="coerce")

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

def df_from_rows(rows, cols):
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# --------------------------
# DB I/O
# --------------------------
def db_load(table: str, cols: list[str]) -> pd.DataFrame:
    try:
        res = supabase.table(table).select("*").execute()
        rows = res.data or []
        return df_from_rows(rows, cols)
    except Exception as e:
        st.error(f"DB load failed for {table}: {e}")
        return pd.DataFrame(columns=cols)

def db_upsert_df(table: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    try:
        # Convert NaN to None for JSON
        payload = df.where(pd.notnull(df), None).to_dict(orient="records")
        supabase.table(table).upsert(payload).execute()
    except Exception as e:
        st.error(f"DB upsert failed for {table}: {e}")

def db_delete(table: str, col: str, value: str):
    try:
        supabase.table(table).delete().eq(col, value).execute()
    except Exception as e:
        st.error(f"DB delete failed for {table}: {e}")


# --------------------------
# Storage I/O (screenshots)
# --------------------------
def upload_screenshot(file, run_id: str) -> str:
    """
    Uploads screenshot to Supabase Storage and returns a public URL.
    Uses a deterministic-ish path so replacing is easy.
    """
    if file is None:
        return ""

    ext = file.name.split(".")[-1].lower()
    if ext not in ("png", "jpg", "jpeg", "webp"):
        ext = "png"

    path = f"{run_id}/payoff.{ext}"
    data = file.getvalue()

    try:
        # upload (upsert=True replaces existing)
        supabase.storage.from_(BUCKET).upload(
            path=path,
            file=data,
            file_options={"content-type": file.type, "upsert": "true"},
        )

        # public URL
        public = supabase.storage.from_(BUCKET).get_public_url(path)
        return public
    except Exception as e:
        st.error(f"Screenshot upload failed: {e}")
        return ""

def show_image_from_url(url: str):
    if not url or not str(url).strip():
        st.caption("No screenshot for this run.")
        return
    st.image(url, use_container_width=True)


# --------------------------
# Init session state (LOAD ON START)
# --------------------------
if "booted" not in st.session_state:
    st.session_state.templates = db_load("templates", TEMPLATE_COLS)
    st.session_state.runs = db_load("runs", RUN_COLS)
    st.session_state.logs = db_load("logs", LOG_COLS)
    st.session_state.booted = True


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


# --------------------------
# Sidebar Navigation
# --------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Run Desk", "Create Template", "Create Run", "Sync / Reload"],
        index=0,
    )
    st.caption("Persistence: Supabase DB + Storage")


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
        db_upsert_df("templates", st.session_state.templates)
        st.success("Template created and saved (Supabase).")

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

    shot = st.file_uploader("Upload payoff/graph screenshot", type=["png", "jpg", "jpeg", "webp"])

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
        screenshot_url = upload_screenshot(shot, rid) if shot is not None else ""

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
            screenshot_url=screenshot_url,
            close_date=to_date_str(close_date) if status == "CLOSED" else None,
            close_spot=float(close_spot) if status == "CLOSED" else None,
            close_mtm_inr=float(close_mtm) if status == "CLOSED" else None,
            close_notes=close_notes.strip() if status == "CLOSED" else None,
        )

        st.session_state.runs = pd.concat([st.session_state.runs, pd.DataFrame([row])], ignore_index=True)
        db_upsert_df("runs", st.session_state.runs)
        st.success("Run created and saved (Supabase).")

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
        show_image_from_url(str(r.get("screenshot_url", "") or ""))

        new_shot = st.file_uploader("Upload/replace screenshot", type=["png", "jpg", "jpeg", "webp"], key="replace")
        if new_shot is not None and st.button("Save screenshot"):
            url = upload_screenshot(new_shot, pick)
            st.session_state.runs.loc[st.session_state.runs["run_id"] == pick, "screenshot_url"] = url
            db_upsert_df("runs", st.session_state.runs)
            st.success("Saved (Supabase Storage).")

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
                st.session_state.runs.loc[idx, "close_spot"] = float(close_spot)
                st.session_state.runs.loc[idx, "close_mtm_inr"] = float(close_mtm)
                st.session_state.runs.loc[idx, "close_notes"] = close_notes.strip()
                db_upsert_df("runs", st.session_state.runs)
                st.success("Run marked CLOSED and square-off saved (Supabase).")
        with cbtn2:
            if st.button("Re-open (set ACTIVE)"):
                idx = st.session_state.runs["run_id"] == pick
                st.session_state.runs.loc[idx, "status"] = "ACTIVE"
                db_upsert_df("runs", st.session_state.runs)
                st.success("Run set to ACTIVE (Supabase).")

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

        log_date_str = to_date_str(log_date)

        if st.button("Save log"):
            # DB enforces unique (run_id, date). We handle error cleanly.
            row = dict(
                log_id=new_id("LOG"),
                run_id=pick,
                date=log_date_str,
                mtm_inr=float(mtm),
                nifty_level=float(nifty_level),
                notes=notes.strip()
            )
            try:
                supabase.table("logs").insert(row).execute()
                # refresh local logs
                st.session_state.logs = db_load("logs", LOG_COLS)
                st.success("Saved log (Supabase).")
            except Exception:
                st.error("This date is already logged for this run. Delete it first to replace it.")

        st.markdown("---")
        if st.button("Delete this run (and its logs)"):
            # logs cascade delete because FK on logs(run_id) has ON DELETE CASCADE
            db_delete("runs", "run_id", pick)

            # refresh local state
            st.session_state.runs = db_load("runs", RUN_COLS)
            st.session_state.logs = db_load("logs", LOG_COLS)
            st.warning("Run deleted (Supabase).")
            st.stop()

    st.markdown("---")
    if df.empty:
        st.info("No logs yet for this run.")
        return

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

def page_sync_reload():
    st.subheader("Sync / Reload")
    st.write("If you ever feel the UI is stale, reload from Supabase.")

    if st.button("Reload everything from Supabase"):
        st.session_state.templates = db_load("templates", TEMPLATE_COLS)
        st.session_state.runs = db_load("runs", RUN_COLS)
        st.session_state.logs = db_load("logs", LOG_COLS)
        st.success("Reloaded.")


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
    page_sync_reload()
