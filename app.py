import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime
from PIL import Image
from io import BytesIO

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Iron Condor Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Simple styling (NO emojis)
# --------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.kpi {
    border: 1px solid rgba(200,200,200,0.25);
    border-radius: 14px;
    padding: 14px;
    background: rgba(240,240,240,0.03);
}
.kpi-title {
    font-size: 0.85rem;
    color: #aaaaaa;
}
.kpi-value {
    font-size: 1.4rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Session state (in-memory only)
# --------------------------------------------------
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=[
        "trade_id", "created_on", "expiry", "strategy",
        "short_put", "long_put", "short_call", "long_call",
        "entry_sp", "entry_lp", "entry_sc", "entry_lc",
        "notes"
    ])

if "daily" not in st.session_state:
    st.session_state.daily = pd.DataFrame(columns=[
        "date", "trade_id", "mtm"
    ])

if "screenshots" not in st.session_state:
    st.session_state.screenshots = {}  # trade_id -> image bytes

# --------------------------------------------------
# NSE Option Chain (unofficial, may fail sometimes)
# --------------------------------------------------
def fetch_nse_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/option-chain"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=10)
    r = session.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def chain_df(raw, expiry):
    rows = []
    for d in raw["records"]["data"]:
        if d.get("expiryDate") == expiry:
            rows.append({
                "strike": d["strikePrice"],
                "CE": d.get("CE", {}).get("lastPrice"),
                "PE": d.get("PE", {}).get("lastPrice")
            })
    return pd.DataFrame(rows)

def ltp(df, strike, opt):
    r = df[df["strike"] == strike]
    if r.empty:
        return None
    return float(r[opt].iloc[0])

def calc_mtm(trade, df):
    legs = [
        ("PE", trade["long_put"],  +1, trade["entry_lp"]),
        ("PE", trade["short_put"], -1, trade["entry_sp"]),
        ("CE", trade["short_call"],-1, trade["entry_sc"]),
        ("CE", trade["long_call"], +1, trade["entry_lc"]),
    ]

    entry_val = 0
    curr_val = 0

    for opt, strike, sign, entry in legs:
        price = ltp(df, strike, opt)
        if price is None:
            return None
        entry_val += sign * entry
        curr_val += sign * price

    return (entry_val - curr_val) * 50  # NIFTY lot size

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Iron Condor Paper Trading Dashboard")
st.caption("Simple | Visual | Data-centric | No local storage")

tabs = st.tabs(["New Trade", "Trade Desk", "Leaderboards"])

# --------------------------------------------------
# TAB 1 — New Trade
# --------------------------------------------------
with tabs[0]:
    st.subheader("Create a new trade")

    try:
        raw = fetch_nse_chain()
        expiries = raw["records"]["expiryDates"]
        chain_ok = True
    except Exception:
        expiries = []
        chain_ok = False
        st.warning("Option chain not available right now. You can still enter prices manually.")

    expiry = st.selectbox("Expiry", expiries if expiries else ["N/A"])
    strategy = st.selectbox("Strategy", ["A", "B", "C"])

    c1, c2, c3, c4 = st.columns(4)
    short_put = c1.number_input("Short Put", step=50)
    long_put  = c2.number_input("Long Put", step=50)
    short_call= c3.number_input("Short Call", step=50)
    long_call = c4.number_input("Long Call", step=50)

    if chain_ok and expiry != "N/A":
        df = chain_df(raw, expiry)
        auto_sp = ltp(df, short_put, "PE")
        auto_lp = ltp(df, long_put, "PE")
        auto_sc = ltp(df, short_call, "CE")
        auto_lc = ltp(df, long_call, "CE")
    else:
        auto_sp = auto_lp = auto_sc = auto_lc = None

    e1, e2, e3, e4 = st.columns(4)
    entry_sp = e1.number_input("Entry SP", value=float(auto_sp) if auto_sp else 0.0)
    entry_lp = e2.number_input("Entry LP", value=float(auto_lp) if auto_lp else 0.0)
    entry_sc = e3.number_input("Entry SC", value=float(auto_sc) if auto_sc else 0.0)
    entry_lc = e4.number_input("Entry LC", value=float(auto_lc) if auto_lc else 0.0)

    notes = st.text_area("Notes")

    img = st.file_uploader("Upload payoff screenshot (optional)", ["png", "jpg", "jpeg"])

    if st.button("Add trade"):
        trade_id = f"{expiry}_{int(datetime.now().timestamp())}"
        row = {
            "trade_id": trade_id,
            "created_on": str(date.today()),
            "expiry": expiry,
            "strategy": strategy,
            "short_put": int(short_put),
            "long_put": int(long_put),
            "short_call": int(short_call),
            "long_call": int(long_call),
            "entry_sp": float(entry_sp),
            "entry_lp": float(entry_lp),
            "entry_sc": float(entry_sc),
            "entry_lc": float(entry_lc),
            "notes": notes
        }
        st.session_state.trades = pd.concat(
            [st.session_state.trades, pd.DataFrame([row])],
            ignore_index=True
        )

        if img:
            st.session_state.screenshots[trade_id] = img.getvalue()

        st.success("Trade added")

# --------------------------------------------------
# TAB 2 — Trade Desk
# --------------------------------------------------
with tabs[1]:
    if st.session_state.trades.empty:
        st.info("No trades yet")
    else:
        pick = st.selectbox(
            "Select trade",
            st.session_state.trades["trade_id"]
        )
        t = st.session_state.trades[
            st.session_state.trades["trade_id"] == pick
        ].iloc[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            if pick in st.session_state.screenshots:
                st.image(
                    Image.open(BytesIO(st.session_state.screenshots[pick])),
                    use_container_width=True
                )
            st.write("Notes:")
            st.write(t["notes"] if t["notes"] else "-")

        with col2:
            try:
                raw = fetch_nse_chain()
                df = chain_df(raw, t["expiry"])
                mtm = calc_mtm(t, df)
            except Exception:
                mtm = None

            if mtm is not None:
                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-title">MTM (INR)</div>
                    <div class="kpi-value">{int(mtm):,}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Auto MTM unavailable")

            manual = st.number_input("Manual MTM", value=0.0)
            use_manual = st.checkbox("Use manual MTM", value=(mtm is None))

            if st.button("Log today"):
                log_val = manual if use_manual else mtm
                st.session_state.daily = pd.concat([
                    st.session_state.daily,
                    pd.DataFrame([{
                        "date": str(date.today()),
                        "trade_id": pick,
                        "mtm": log_val
                    }])
                ], ignore_index=True)
                st.success("Logged")

        hist = st.session_state.daily[
            st.session_state.daily["trade_id"] == pick
        ]
        if not hist.empty:
            hist["date"] = pd.to_datetime(hist["date"])
            st.line_chart(hist.set_index("date")["mtm"])

# --------------------------------------------------
# TAB 3 — Leaderboards
# --------------------------------------------------
with tabs[2]:
    if st.session_state.daily.empty:
        st.info("No daily logs yet")
    else:
        latest = (
            st.session_state.daily
            .sort_values("date")
            .groupby("trade_id")
            .tail(1)
        )
        st.subheader("Top performers (latest MTM)")
        st.dataframe(
            latest.sort_values("mtm", ascending=False),
            use_container_width=True
        )

