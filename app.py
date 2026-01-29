import base64
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO

# --------------------------
# Page + basic styling
# --------------------------
st.set_page_config(page_title="Options Journal", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .kpi { border: 1px solid rgba(255,255,255,0.14); border-radius: 16px; padding: 14px 16px;
           background: rgba(255,255,255,0.04); }
    .kpi-title { font-size: 0.85rem; color: rgba(255,255,255,0.65); margin-bottom: 4px; }
    .kpi-value { font-size: 1.4rem; font-weight: 800; }
    .kpi-sub { font-size: 0.85rem; color: rgba(255,255,255,0.55); margin-top: 4px; }
    hr { border: none; height: 1px; background: rgba(255,255,255,0.10); margin: 0.8rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Iron Condor / Options Strategy Journal (EOD)")
st.caption("Manual end-of-day journaling. Upload payoff screenshots. Strategy view shows everything at a glance.")

# --------------------------
# Helpers
# --------------------------
def b64_from_upload(uploaded_file) -> str:
    """Convert uploaded image file to base64 string. Returns empty string if no file."""
    if uploaded_file is None:
        return ""
    data = uploaded_file.getvalue()
    return base64.b64encode(data).decode("utf-8")

def image_from_b64(b64_str: str):
    """Convert base64 string back to PIL Image; returns None if invalid/empty."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        raw = base64.b64decode(b64_str)
        return Image.open(BytesIO(raw))
    except Exception:
        return None

def safe_to_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

# --------------------------
# Data model (persist via CSV import/export)
# --------------------------
STRAT_COLS = [
    "strategy_id",
    "name",
    "strategy_type",         # e.g., Iron Condor / Iron Fly / Strangle
    "underlying",            # NIFTY / BANKNIFTY
    "start_date",
    "expiry",
    "qty_lots",
    "lot_size",

    # Legs (optional free fields)
    "short_put",
    "long_put",
    "short_call",
    "long_call",

    # Your plan / rules
    "entry_reason",
    "risk_rules",
    "exit_plan",

    # Screenshot stored as base64
    "screenshot_b64",
]

LOG_COLS = [
    "log_id",
    "strategy_id",
    "date",
    "day_pnl_inr",
    "cum_pnl_inr",           # optional; if you prefer, app can compute
    "notes",
    "vix",
    "ivp",
    "spot_close",
]

# --------------------------
# Session state init
# --------------------------
if "strategies" not in st.session_state:
    st.session_state.strategies = pd.DataFrame(columns=STRAT_COLS)

if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=LOG_COLS)

# --------------------------
# Sidebar: Import/Export (THIS is persistence)
# --------------------------
with st.sidebar:
    st.header("Data Backup (Important)")
    st.caption("Streamlit cloud resets storage. Use CSV export/import to persist your journal (including screenshots).")

    # Export
    strat_csv = st.session_state.strategies.to_csv(index=False).encode("utf-8")
    logs_csv = st.session_state.logs.to_csv(index=False).encode("utf-8")

    st.download_button("Download strategies.csv", strat_csv, file_name="strategies.csv", mime="text/csv")
    st.download_button("Download logs.csv", logs_csv, file_name="logs.csv", mime="text/csv")

    st.divider()
    st.subheader("Import CSVs")
    up_s = st.file_uploader("Upload strategies.csv", type=["csv"], key="up_strat")
    up_l = st.file_uploader("Upload logs.csv", type=["csv"], key="up_logs")

    if st.button("Import now"):
        if up_s is not None:
            try:
                df = pd.read_csv(up_s)
                df = ensure_cols(df, STRAT_COLS)
                st.session_state.strategies = df
            except Exception as e:
                st.error(f"Failed to import strategies.csv: {e}")

        if up_l is not None:
            try:
                df = pd.read_csv(up_l)
                df = ensure_cols(df, LOG_COLS)
                st.session_state.logs = df
            except Exception as e:
                st.error(f"Failed to import logs.csv: {e}")

        st.success("Import complete (where files were provided).")

    st.divider()
    if st.button("Reset all (memory)"):
        st.session_state.strategies = pd.DataFrame(columns=STRAT_COLS)
        st.session_state.logs = pd.DataFrame(columns=LOG_COLS)
        st.warning("Reset done (in memory).")

# --------------------------
# Tabs
# --------------------------
tab_new, tab_desk, tab_analytics = st.tabs(["Create Strategy", "Strategy Desk", "Analytics"])

# =========================================================
# TAB 1: Create Strategy
# =========================================================
with tab_new:
    st.subheader("Create a strategy (one-time)")
    c1, c2, c3, c4 = st.columns([0.40, 0.20, 0.20, 0.20])

    name = c1.text_input("Strategy name", placeholder="e.g., IC 300/50 Feb Week1 | Low Risk Theta")
    strategy_type = c2.text_input("Type", value="Iron Condor")
    underlying = c3.text_input("Underlying", value="NIFTY")
    expiry = c4.text_input("Expiry", placeholder="e.g., 06-Feb-2026")

    c5, c6, c7 = st.columns(3)
    start_date = c5.date_input("Start date", value=date.today())
    qty_lots = c6.number_input("Quantity (lots)", min_value=1, value=1, step=1)
    lot_size = c7.number_input("Lot size", min_value=1, value=50, step=1)

    st.markdown("### Legs (optional but recommended)")
    l1, l2, l3, l4 = st.columns(4)
    short_put = l1.text_input("Short Put strike", placeholder="e.g., 25000")
    long_put = l2.text_input("Long Put strike", placeholder="e.g., 24700")
    short_call = l3.text_input("Short Call strike", placeholder="e.g., 25700")
    long_call = l4.text_input("Long Call strike", placeholder="e.g., 26000")

    st.markdown("### Plan / Rules")
    entry_reason = st.text_area("Entry reason (why did you take it?)", height=90, placeholder="IVP/VIX view, range expectation, event risk, etc.")
    r1, r2 = st.columns(2)
    risk_rules = r1.text_area("Risk rules", height=90, placeholder="Max loss, adjustment rules, stop-loss logic, when to hedge, etc.")
    exit_plan = r2.text_area("Exit plan", height=90, placeholder="Profit target, time-based exit, IV crush exit, etc.")

    st.markdown("### Upload payoff / strategy graph (optional)")
    shot = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if st.button("Add strategy"):
        if not name.strip():
            st.error("Give the strategy a name.")
        else:
            strategy_id = f"STRAT_{int(datetime.now().timestamp())}"
            row = {
                "strategy_id": strategy_id,
                "name": name.strip(),
                "strategy_type": strategy_type.strip(),
                "underlying": underlying.strip(),
                "start_date": str(start_date),
                "expiry": expiry.strip(),
                "qty_lots": int(qty_lots),
                "lot_size": int(lot_size),

                "short_put": short_put.strip(),
                "long_put": long_put.strip(),
                "short_call": short_call.strip(),
                "long_call": long_call.strip(),

                "entry_reason": entry_reason.strip(),
                "risk_rules": risk_rules.strip(),
                "exit_plan": exit_plan.strip(),

                "screenshot_b64": b64_from_upload(shot),
            }
            st.session_state.strategies = pd.concat(
                [st.session_state.strategies, pd.DataFrame([row])],
                ignore_index=True
            )
            st.success("Strategy created. Open it in Strategy Desk.")

# =========================================================
# TAB 2: Strategy Desk (open a strategy, see everything, log EOD)
# =========================================================
with tab_desk:
    st.subheader("Strategy Desk")

    strategies = st.session_state.strategies.copy()
    logs = st.session_state.logs.copy()

    if strategies.empty:
        st.info("No strategies yet. Create one in 'Create Strategy'.")
        st.stop()

    # Select strategy
    def label_strategy(sid):
        row = strategies[strategies["strategy_id"] == sid].iloc[0]
        return f"{row['name']}  |  {row['underlying']}  |  Exp: {row['expiry']}"

    pick = st.selectbox("Select strategy", strategies["strategy_id"].tolist(), format_func=label_strategy)
    srow = strategies[strategies["strategy_id"] == pick].iloc[0]

    # Top KPIs for selected strategy
    slog = logs[logs["strategy_id"] == pick].copy()
    if not slog.empty:
        slog["date"] = slog["date"].apply(safe_to_datetime)
        slog = slog.sort_values("date")
        latest_pnl = float(slog["day_pnl_inr"].iloc[-1]) if pd.notna(slog["day_pnl_inr"].iloc[-1]) else 0.0
        total_pnl = float(slog["day_pnl_inr"].fillna(0).sum())
        days_logged = int(slog.shape[0])
        best_day = float(slog["day_pnl_inr"].fillna(0).max())
        worst_day = float(slog["day_pnl_inr"].fillna(0).min())
    else:
        latest_pnl = total_pnl = best_day = worst_day = 0.0
        days_logged = 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Days logged</div><div class='kpi-value'>{days_logged}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Total P&L (INR)</div><div class='kpi-value'>{total_pnl:,.0f}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Latest day P&L</div><div class='kpi-value'>{latest_pnl:,.0f}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='kpi-title'>Best day</div><div class='kpi-value'>{best_day:,.0f}</div></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='kpi'><div class='kpi-title'>Worst day</div><div class='kpi-value'>{worst_day:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("hr", unsafe_allow_html=True)

    # Layout: screenshot + details + EOD log
    left, mid, right = st.columns([0.95, 1.05, 1.0])

    with left:
        st.markdown("### Screenshot")
        img = image_from_b64(str(srow.get("screenshot_b64", "")))
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.caption("No screenshot saved for this strategy. You can upload a new one below.")
            new_shot = st.file_uploader("Upload/replace screenshot", type=["png", "jpg", "jpeg"], key="replace_shot")
            if new_shot is not None and st.button("Save screenshot"):
                b64 = b64_from_upload(new_shot)
                st.session_state.strategies.loc[st.session_state.strategies["strategy_id"] == pick, "screenshot_b64"] = b64
                st.success("Screenshot updated. Refresh or re-open the strategy.")

    with mid:
        st.markdown("### Strategy details")
        st.write({
            "Name": srow["name"],
            "Type": srow["strategy_type"],
            "Underlying": srow["underlying"],
            "Start date": srow["start_date"],
            "Expiry": srow["expiry"],
            "Lots": int(srow["qty_lots"]),
            "Lot size": int(srow["lot_size"]),
        })

        st.markdown("#### Legs")
        st.write({
            "Short Put": srow["short_put"],
            "Long Put": srow["long_put"],
            "Short Call": srow["short_call"],
            "Long Call": srow["long_call"],
        })

        st.markdown("#### Plan")
        st.write("Entry reason:")
        st.write(srow["entry_reason"] if str(srow["entry_reason"]).strip() else "-")
        st.write("Risk rules:")
        st.write(srow["risk_rules"] if str(srow["risk_rules"]).strip() else "-")
        st.write("Exit plan:")
        st.write(srow["exit_plan"] if str(srow["exit_plan"]).strip() else "-")

    with right:
        st.markdown("### EOD log (add once per day)")
        log_date = st.date_input("Date", value=date.today(), key="log_date")

        day_pnl = st.number_input("Day P&L (INR)", value=0.0, step=100.0)
        vix = st.number_input("VIX (optional)", value=0.0, step=0.1)
        ivp = st.number_input("IVP (optional)", value=0.0, step=1.0)
        spot_close = st.number_input("Spot close (optional)", value=0.0, step=1.0)

        notes = st.text_area("Notes (what happened today?)", height=120, placeholder="Adjustments, breaches, IV changes, emotions, lessons, etc.")

        if st.button("Save EOD log"):
            # Prevent duplicate log for same date for same strategy
            existing = st.session_state.logs[
                (st.session_state.logs["strategy_id"] == pick) &
                (st.session_state.logs["date"] == str(log_date))
            ]
            if not existing.empty:
                st.error("You already logged this date for this strategy. Delete/edit from table below (delete supported).")
            else:
                log_id = f"LOG_{int(datetime.now().timestamp())}"
                row = {
                    "log_id": log_id,
                    "strategy_id": pick,
                    "date": str(log_date),
                    "day_pnl_inr": float(day_pnl),
                    "cum_pnl_inr": None,
                    "notes": notes.strip(),
                    "vix": float(vix) if vix != 0 else None,
                    "ivp": float(ivp) if ivp != 0 else None,
                    "spot_close": float(spot_close) if spot_close != 0 else None,
                }
                st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([row])], ignore_index=True)
                st.success("Logged.")

        st.markdown("hr", unsafe_allow_html=True)
        if st.button("Delete this strategy (and its logs)"):
            st.session_state.strategies = st.session_state.strategies[st.session_state.strategies["strategy_id"] != pick].reset_index(drop=True)
            st.session_state.logs = st.session_state.logs[st.session_state.logs["strategy_id"] != pick].reset_index(drop=True)
            st.warning("Strategy deleted.")
            st.stop()

    st.markdown("hr", unsafe_allow_html=True)

    # Timeline + table for selected strategy
    st.markdown("### Performance timeline")
    if slog.empty:
        st.info("No logs yet for this strategy.")
    else:
        plot_df = slog.copy()
        plot_df["cum"] = plot_df["day_pnl_inr"].fillna(0).cumsum()

        c1, c2 = st.columns([0.6, 0.4])
        with c1:
            st.line_chart(plot_df.set_index("date")["cum"], height=240)
        with c2:
            st.line_chart(plot_df.set_index("date")["day_pnl_inr"], height=240)

        st.markdown("### Logs table")
        show_cols = ["date", "day_pnl_inr", "vix", "ivp", "spot_close", "notes", "log_id"]
        st.dataframe(plot_df[show_cols], use_container_width=True, height=300)

        # Delete a log row
        st.markdown("#### Delete a log entry")
        del_id = st.text_input("Paste log_id to delete", placeholder="Copy from table (log_id column)")
        if st.button("Delete log"):
            if del_id.strip():
                before = st.session_state.logs.shape[0]
                st.session_state.logs = st.session_state.logs[st.session_state.logs["log_id"] != del_id.strip()].reset_index(drop=True)
                after = st.session_state.logs.shape[0]
                if after < before:
                    st.success("Deleted log.")
                else:
                    st.warning("No matching log_id found.")

# =========================================================
# TAB 3: Analytics (all strategies)
# =========================================================
with tab_analytics:
    st.subheader("Analytics: all strategies")

    strategies = st.session_state.strategies.copy()
    logs = st.session_state.logs.copy()

    if logs.empty:
        st.info("No logs yet. Start logging EOD to unlock analytics.")
        st.stop()

    logs["date"] = logs["date"].apply(safe_to_datetime)
    logs = logs.dropna(subset=["date"])
    logs = logs.sort_values("date")

    # Total per strategy
    totals = (
        logs.assign(day_pnl_inr=logs["day_pnl_inr"].fillna(0))
        .groupby("strategy_id")["day_pnl_inr"]
        .sum()
        .reset_index()
        .rename(columns={"day_pnl_inr": "total_pnl"})
        .merge(strategies[["strategy_id", "name", "underlying", "expiry", "strategy_type"]], on="strategy_id", how="left")
        .sort_values("total_pnl", ascending=False)
    )

    # Weekly top performer (last 7 days)
    last7 = logs[logs["date"] >= (pd.Timestamp(date.today()) - pd.Timedelta(days=7))].copy()
    weekly = (
        last7.assign(day_pnl_inr=last7["day_pnl_inr"].fillna(0))
        .groupby("strategy_id")["day_pnl_inr"]
        .sum()
        .reset_index()
        .rename(columns={"day_pnl_inr": "pnl_7d"})
        .merge(strategies[["strategy_id", "name"]], on="strategy_id", how="left")
        .sort_values("pnl_7d", ascending=False)
    )

    # Today summary
    today_logs = logs[logs["date"].dt.date == date.today()].copy()
    today_sum = float(today_logs["day_pnl_inr"].fillna(0).sum()) if not today_logs.empty else 0.0

    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Total strategies</div><div class='kpi-value'>{strategies.shape[0]}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Total logged days</div><div class='kpi-value'>{logs.shape[0]}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Today total P&L</div><div class='kpi-value'>{today_sum:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("hr", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### All-time top performers (Total P&L)")
        st.dataframe(totals[["name","strategy_type","underlying","expiry","total_pnl","strategy_id"]], use_container_width=True, height=380)

    with c2:
        st.markdown("### This week top performers (7D P&L)")
        if weekly.empty:
            st.info("No logs in last 7 days.")
        else:
            st.dataframe(weekly[["name","pnl_7d","strategy_id"]], use_container_width=True, height=380)

    st.markdown("hr", unsafe_allow_html=True)
    st.markdown("### Portfolio curve (sum of daily P&L across all strategies)")
    daily_port = logs.copy()
    daily_port["day_pnl_inr"] = daily_port["day_pnl_inr"].fillna(0)
    port = daily_port.groupby(daily_port["date"].dt.date)["day_pnl_inr"].sum().reset_index()
    port["date"] = pd.to_datetime(port["date"])
    port = port.sort_values("date")
    port["cum"] = port["day_pnl_inr"].cumsum()
    st.line_chart(port.set_index("date")["cum"], height=280)
