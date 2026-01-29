import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Iron Condor Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.kpi { border: 1px solid rgba(200,200,200,0.25); border-radius: 14px; padding: 14px;
       background: rgba(240,240,240,0.03); }
.kpi-title { font-size: 0.85rem; color: #aaaaaa; }
.kpi-value { font-size: 1.4rem; font-weight: 700; }
.small { color: #aaaaaa; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- Session state --------------------
TRADES_COLS = [
    "trade_id", "created_on", "strategy_name", "symbol", "expiry",
    "qty_lots", "lot_size",
    "short_put", "long_put", "short_call", "long_call",
    "entry_sp", "entry_lp", "entry_sc", "entry_lc",
    "notes"
]
DAILY_COLS = ["date", "trade_id", "mtm"]

if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=TRADES_COLS)
if "daily" not in st.session_state:
    st.session_state.daily = pd.DataFrame(columns=DAILY_COLS)
if "screenshots" not in st.session_state:
    st.session_state.screenshots = {}  # trade_id -> bytes

# -------------------- NSE Option Chain (unofficial) --------------------
def fetch_nse_chain(symbol="NIFTY"):
    base = "https://www.nseindia.com"
    url = f"{base}/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }
    s = requests.Session()
    s.get(base, headers=headers, timeout=10)
    r = s.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def chain_df(raw, expiry):
    rows = []
    for d in raw.get("records", {}).get("data", []):
        if d.get("expiryDate") == expiry:
            rows.append({
                "strike": d.get("strikePrice"),
                "CE": d.get("CE", {}).get("lastPrice"),
                "PE": d.get("PE", {}).get("lastPrice"),
            })
    return pd.DataFrame(rows)

def ltp(df, strike, opt):
    r = df[df["strike"] == strike]
    if r.empty:
        return None
    return float(r[opt].iloc[0]) if pd.notna(r[opt].iloc[0]) else None

def calc_mtm(trade_row, df):
    # long = +1, short = -1
    qty = int(trade_row["qty_lots"])
    lot_size = int(trade_row["lot_size"])

    legs = [
        ("PE", int(trade_row["long_put"]),  +1, float(trade_row["entry_lp"])),
        ("PE", int(trade_row["short_put"]), -1, float(trade_row["entry_sp"])),
        ("CE", int(trade_row["short_call"]),-1, float(trade_row["entry_sc"])),
        ("CE", int(trade_row["long_call"]), +1, float(trade_row["entry_lc"])),
    ]

    entry_val, curr_val = 0.0, 0.0
    for opt, strike, sign, entry_price in legs:
        price = ltp(df, strike, opt)
        if price is None:
            return None
        entry_val += sign * entry_price
        curr_val  += sign * price

    pnl_points = (entry_val - curr_val)
    return pnl_points * lot_size * qty

# -------------------- Header --------------------
st.title("Iron Condor Paper Trading Dashboard")
st.caption("Strategy journal + MTM logging. Auto MTM depends on data availability.")

tabs = st.tabs(["New Trade", "Trade Desk", "Leaderboards"])

# -------------------- New Trade --------------------
with tabs[0]:
    st.subheader("Create trade")

    c1, c2, c3 = st.columns([0.45, 0.25, 0.30])
    strategy_name = c1.text_input("Strategy name", placeholder="e.g., IC Wide 300/50, Low Risk Theta")
    symbol = c2.text_input("Symbol", value="NIFTY")
    expiry = c3.text_input("Expiry (type it)", placeholder="e.g., 06-Feb-2026 (must match chain format if auto MTM)")

    c4, c5 = st.columns(2)
    qty_lots = c4.number_input("Quantity (lots)", min_value=1, value=1, step=1)
    lot_size = c5.number_input("Lot size", min_value=1, value=50, step=1)

    st.markdown("### Strikes")
    s1, s2, s3, s4 = st.columns(4)
    short_put = s1.number_input("Short Put", step=50)
    long_put  = s2.number_input("Long Put", step=50)
    short_call= s3.number_input("Short Call", step=50)
    long_call = s4.number_input("Long Call", step=50)

    st.markdown("### Entry prices")
    e1, e2, e3, e4 = st.columns(4)
    entry_sp = e1.number_input("Entry SP (PE)", value=0.0, step=0.05, format="%.2f")
    entry_lp = e2.number_input("Entry LP (PE)", value=0.0, step=0.05, format="%.2f")
    entry_sc = e3.number_input("Entry SC (CE)", value=0.0, step=0.05, format="%.2f")
    entry_lc = e4.number_input("Entry LC (CE)", value=0.0, step=0.05, format="%.2f")

    notes = st.text_area("Notes", placeholder="IVP/VIX view, rules, why this trade, etc.")

    img = st.file_uploader("Upload payoff screenshot (optional)", ["png", "jpg", "jpeg"])

    if st.button("Add trade"):
        if not strategy_name.strip():
            st.error("Please enter a strategy name.")
        elif not expiry.strip():
            st.error("Please enter expiry text (you can type it).")
        else:
            trade_id = f"{symbol}_{int(datetime.now().timestamp())}"
            row = {
                "trade_id": trade_id,
                "created_on": str(date.today()),
                "strategy_name": strategy_name.strip(),
                "symbol": symbol.strip(),
                "expiry": expiry.strip(),
                "qty_lots": int(qty_lots),
                "lot_size": int(lot_size),
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
            st.session_state.trades = pd.concat([st.session_state.trades, pd.DataFrame([row])], ignore_index=True)
            if img:
                st.session_state.screenshots[trade_id] = img.getvalue()
            st.success("Trade added to journal.")

# -------------------- Trade Desk --------------------
with tabs[1]:
    trades = st.session_state.trades
    if trades.empty:
        st.info("No trades yet. Add one in 'New Trade'.")
    else:
        pick = st.selectbox(
            "Select trade",
            trades["trade_id"].tolist(),
            format_func=lambda tid: f"{trades.loc[trades.trade_id==tid, 'strategy_name'].iloc[0]}  |  {tid}"
        )
        t = trades[trades["trade_id"] == pick].iloc[0]

        left, right = st.columns([1, 1])

        with left:
            st.markdown("#### Screenshot")
            if pick in st.session_state.screenshots:
                st.image(Image.open(BytesIO(st.session_state.screenshots[pick])), use_container_width=True)
            else:
                st.caption("No screenshot stored for this trade.")

            st.markdown("#### Notes")
            st.write(t["notes"] if str(t["notes"]).strip() else "-")

        with right:
            st.markdown("#### Live MTM (auto if chain works)")
            mtm = None
            auto_error = None
            try:
                raw = fetch_nse_chain(symbol=t["symbol"])
                df = chain_df(raw, t["expiry"])
                if df.empty:
                    auto_error = "Expiry format does not match chain. Try the exact NSE expiry text (e.g., '06-Feb-2026')."
                else:
                    mtm = calc_mtm(t, df)
                    if mtm is None:
                        auto_error = "Could not find one or more strikes in the option chain for this expiry."
            except Exception as e:
                auto_error = f"Chain fetch failed (common on cloud): {e}"

            if mtm is not None:
                st.markdown(
                    f"<div class='kpi'><div class='kpi-title'>MTM (INR)</div><div class='kpi-value'>{int(mtm):,}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("Auto MTM unavailable right now.")
                if auto_error:
                    st.caption(auto_error)

            st.markdown("#### Log today")
            manual_mtm = st.number_input("Manual MTM (INR)", value=0.0, step=100.0)
            use_manual = st.checkbox("Use manual MTM for logging", value=(mtm is None))

            if st.button("Log MTM for today"):
                log_val = float(manual_mtm) if use_manual else float(mtm)
                st.session_state.daily = pd.concat([st.session_state.daily, pd.DataFrame([{
                    "date": str(date.today()),
                    "trade_id": pick,
                    "mtm": log_val
                }])], ignore_index=True)
                st.success("Logged.")

            hist = st.session_state.daily[st.session_state.daily["trade_id"] == pick].copy()
            if not hist.empty:
                hist["date"] = pd.to_datetime(hist["date"])
                hist = hist.sort_values("date")
                st.line_chart(hist.set_index("date")["mtm"])

# -------------------- Leaderboards --------------------
with tabs[2]:
    daily = st.session_state.daily.copy()
    trades = st.session_state.trades.copy()

    if daily.empty:
        st.info("No logs yet. Log daily MTM to see leaderboards.")
    else:
        daily["date"] = pd.to_datetime(daily["date"])
        latest = daily.sort_values("date").groupby("trade_id").tail(1)
        latest = latest.merge(trades[["trade_id","strategy_name","expiry"]], on="trade_id", how="left")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='kpi'><div class='kpi-title'>Trades tracked</div><div class='kpi-value'>{trades.shape[0]}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi'><div class='kpi-title'>Logs recorded</div><div class='kpi-value'>{daily.shape[0]}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi'><div class='kpi-title'>Top MTM (latest)</div><div class='kpi-value'>{int(latest['mtm'].max()):,}</div></div>", unsafe_allow_html=True)

        st.markdown("#### Top performers (latest MTM)")
        show = latest.sort_values("mtm", ascending=False)[["strategy_name","expiry","date","mtm","trade_id"]]
        st.dataframe(show, use_container_width=True, height=420)
