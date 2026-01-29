import base64
from datetime import date, datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------- Page setup ----------------
st.set_page_config(page_title="Options Strategy Journal", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.kpi { border: 1px solid rgba(255,255,255,0.15); border-radius: 16px;
       padding: 14px 16px; background: rgba(255,255,255,0.04); }
.kpi-title { font-size: 0.85rem; color: rgba(255,255,255,0.65); }
.kpi-value { font-size: 1.4rem; font-weight: 800; }
.small { color: rgba(255,255,255,0.6); font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("Iron Condor / Options Strategy Journal")
st.caption("Manual EOD journal • Screenshot support • Margin-based performance")

# ---------------- Helpers ----------------
def img_to_b64(file):
    if not file:
        return ""
    return base64.b64encode(file.getvalue()).decode()

def b64_to_img(b64):
    if not b64:
        return None
    try:
        return Image.open(BytesIO(base64.b64decode(b64)))
    except Exception:
        return None

# ---------------- Data schema ----------------
STRATEGY_COLS = [
    "strategy_id","name","type","underlying","start_date","expiry",
    "entry_spot","margin","qty_lots","lot_size",
    "short_put","long_put","short_call","long_call",
    "entry_reason","risk_rules","exit_plan",
    "screenshot"
]

LOG_COLS = [
    "log_id","strategy_id","date","mtm","notes"
]

# ---------------- Session state ----------------
if "strategies" not in st.session_state:
    st.session_state.strategies = pd.DataFrame(columns=STRATEGY_COLS)

if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=LOG_COLS)

# ---------------- Sidebar: Backup ----------------
with st.sidebar:
    st.header("Backup / Restore")

    st.download_button(
        "Download strategies.csv",
        st.session_state.strategies.to_csv(index=False).encode(),
        "strategies.csv"
    )

    st.download_button(
        "Download logs.csv",
        st.session_state.logs.to_csv(index=False).encode(),
        "logs.csv"
    )

    up_s = st.file_uploader("Upload strategies.csv", type="csv")
    up_l = st.file_uploader("Upload logs.csv", type="csv")

    if st.button("Import CSVs"):
        if up_s:
            st.session_state.strategies = pd.read_csv(up_s)
        if up_l:
            st.session_state.logs = pd.read_csv(up_l)
        st.success("Data imported")

# ---------------- Tabs ----------------
tab_create, tab_desk, tab_leader = st.tabs(["Create Strategy", "Strategy Desk", "Leaderboard"])

# =====================================================
# CREATE STRATEGY
# =====================================================
with tab_create:
    st.subheader("Create New Strategy")

    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Strategy Name")
    strat_type = c2.text_input("Strategy Type", value="Iron Condor")
    underlying = c3.text_input("Underlying", value="NIFTY")

    c4, c5, c6 = st.columns(3)
    start_date = c4.date_input("Start Date", value=date.today())
    expiry = c5.text_input("Expiry")
    entry_spot = c6.number_input("NIFTY at Entry", step=1.0)

    c7, c8 = st.columns(2)
    margin = c7.number_input("Margin Used (₹)", step=1000.0)
    qty_lots = c8.number_input("Quantity (lots)", min_value=1, value=1)

    lot_size = st.number_input("Lot Size", value=50)

    st.markdown("### Strikes")
    s1, s2, s3, s4 = st.columns(4)
    short_put = s1.text_input("Short Put")
    long_put = s2.text_input("Long Put")
    short_call = s3.text_input("Short Call")
    long_call = s4.text_input("Long Call")

    entry_reason = st.text_area("Entry Reason")
    risk_rules = st.text_area("Risk Rules")
    exit_plan = st.text_area("Exit Plan")

    screenshot = st.file_uploader("Upload Payoff / Graph", type=["png","jpg","jpeg"])

    if st.button("Add Strategy"):
        if not name or margin <= 0:
            st.error("Strategy name and margin are required")
        else:
            sid = f"STRAT_{int(datetime.now().timestamp())}"
            row = {
                "strategy_id": sid,
                "name": name,
                "type": strat_type,
                "underlying": underlying,
                "start_date": str(start_date),
                "expiry": expiry,
                "entry_spot": entry_spot,
                "margin": margin,
                "qty_lots": qty_lots,
                "lot_size": lot_size,
                "short_put": short_put,
                "long_put": long_put,
                "short_call": short_call,
                "long_call": long_call,
                "entry_reason": entry_reason,
                "risk_rules": risk_rules,
                "exit_plan": exit_plan,
                "screenshot": img_to_b64(screenshot)
            }
            st.session_state.strategies = pd.concat(
                [st.session_state.strategies, pd.DataFrame([row])],
                ignore_index=True
            )
            st.success("Strategy created")

# =====================================================
# STRATEGY DESK
# =====================================================
with tab_desk:
    if st.session_state.strategies.empty:
        st.info("No strategies yet")
    else:
        pick = st.selectbox(
            "Select Strategy",
            st.session_state.strategies["strategy_id"],
            format_func=lambda x: st.session_state.strategies.loc[
                st.session_state.strategies.strategy_id==x, "name"
            ].iloc[0]
        )

        s = st.session_state.strategies.query("strategy_id == @pick").iloc[0]
        logs = st.session_state.logs.query("strategy_id == @pick").copy()

        latest_mtm = logs["mtm"].iloc[-1] if not logs.empty else 0.0
        profit_pct = (latest_mtm / s["margin"]) * 100 if s["margin"] else 0.0

        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'><div class='kpi-title'>Latest MTM</div><div class='kpi-value'>₹{latest_mtm:,.0f}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><div class='kpi-title'>Margin Used</div><div class='kpi-value'>₹{s['margin']:,.0f}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><div class='kpi-title'>Profit %</div><div class='kpi-value'>{profit_pct:.2f}%</div></div>", unsafe_allow_html=True)

        left, right = st.columns([1,1])

        with left:
            img = b64_to_img(s["screenshot"])
            if img:
                st.image(img, use_container_width=True)
            st.markdown("### Strategy Details")
            st.write(s)

        with right:
            st.markdown("### Log EOD MTM")
            mtm = st.number_input("MTM since entry (₹)", step=100.0)
            note = st.text_area("Notes")

            if st.button("Save Log"):
                lid = f"LOG_{int(datetime.now().timestamp())}"
                st.session_state.logs = pd.concat([
                    st.session_state.logs,
                    pd.DataFrame([{
                        "log_id": lid,
                        "strategy_id": pick,
                        "date": str(date.today()),
                        "mtm": mtm,
                        "notes": note
                    }])
                ], ignore_index=True)
                st.success("Log saved")

            if st.button("DELETE STRATEGY"):
                st.session_state.strategies = st.session_state.strategies.query("strategy_id != @pick")
                st.session_state.logs = st.session_state.logs.query("strategy_id != @pick")
                st.warning("Strategy deleted")
                st.stop()

        if not logs.empty:
            logs["date"] = pd.to_datetime(logs["date"])
            st.line_chart(logs.set_index("date")["mtm"])

# =====================================================
# LEADERBOARD
# =====================================================
with tab_leader:
    if st.session_state.logs.empty:
        st.info("No performance data yet")
    else:
        latest = (
            st.session_state.logs
            .sort_values("date")
            .groupby("strategy_id")
            .tail(1)
            .merge(st.session_state.strategies, on="strategy_id")
        )

        latest["profit_pct"] = (latest["mtm"] / latest["margin"]) * 100

        st.subheader("Top Performing Strategies")
        st.dataframe(
            latest.sort_values("profit_pct", ascending=False)[
                ["name","mtm","margin","profit_pct","expiry"]
            ],
            use_container_width=True
        )
