
Save. ✅

---

# 📌 FILE 5 (MOST IMPORTANT): `app.py`
Now open **app.py** and paste **EVERYTHING below** (don’t edit yet):

```python
import os
from datetime import date, datetime
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------- CONFIG ----------------
DATA_DIR = "data"
SS_DIR = os.path.join(DATA_DIR, "screenshots")
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
DAILY_CSV = os.path.join(DATA_DIR, "daily_log.csv")

os.makedirs(SS_DIR, exist_ok=True)

st.set_page_config(layout="wide")
st.title("Iron Condor Dashboard")

# ---------------- HELPERS ----------------
def ensure_csv(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

ensure_csv(TRADES_CSV, [
    "trade_id","created_on","expiry","strategy",
    "short_put","long_put","short_call","long_call",
    "entry_sp","entry_lp","entry_sc","entry_lc",
    "screenshot"
])
ensure_csv(DAILY_CSV, ["date","trade_id","mtm"])

def fetch_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/option-chain"
    }
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers)
    r = s.get(url, headers=headers)
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
    row = df[df["strike"] == strike]
    if row.empty:
        return None
    return float(row[opt].iloc[0])

def calc_mtm(t, df):
    legs = [
        ("PE", t["long_put"],  +1, t["entry_lp"]),
        ("PE", t["short_put"], -1, t["entry_sp"]),
        ("CE", t["short_call"],-1, t["entry_sc"]),
        ("CE", t["long_call"], +1, t["entry_lc"]),
    ]
    entry, curr = 0, 0
    for opt, strike, sign, e in legs:
        price = ltp(df, strike, opt)
        if price is None:
            return None
        entry += sign * e
        curr  += sign * price
    return (entry - curr) * 50  # NIFTY lot size

# ---------------- UI ----------------
tab1, tab2, tab3 = st.tabs(["➕ New Trade", "📂 Open Trade", "🏆 Leaderboard"])

# -------- TAB 1: NEW TRADE --------
with tab1:
    st.subheader("Create Trade")

    raw = fetch_chain()
    expiries = raw["records"]["expiryDates"]
    expiry = st.selectbox("Expiry", expiries)
    strategy = st.selectbox("Strategy", ["A","B","C"])

    c1,c2,c3,c4 = st.columns(4)
    sp = c1.number_input("Short Put", step=50)
    lp = c2.number_input("Long Put", step=50)
    sc = c3.number_input("Short Call", step=50)
    lc = c4.number_input("Long Call", step=50)

    df = chain_df(raw, expiry)

    ss = st.file_uploader("Upload payoff screenshot", ["png","jpg"])

    if st.button("Create Trade"):
        trade_id = f"{expiry}_{sp}_{sc}_{int(datetime.now().timestamp())}"
        path = ""
        if ss:
            path = os.path.join(SS_DIR, f"{trade_id}.png")
            with open(path,"wb") as f:
                f.write(ss.getbuffer())

        row = {
            "trade_id": trade_id,
            "created_on": str(date.today()),
            "expiry": expiry,
            "strategy": strategy,
            "short_put": sp,
            "long_put": lp,
            "short_call": sc,
            "long_call": lc,
            "entry_sp": ltp(df, sp, "PE"),
            "entry_lp": ltp(df, lp, "PE"),
            "entry_sc": ltp(df, sc, "CE"),
            "entry_lc": ltp(df, lc, "CE"),
            "screenshot": path
        }

        trades = pd.read_csv(TRADES_CSV)
        trades = pd.concat([trades, pd.DataFrame([row])])
        trades.to_csv(TRADES_CSV, index=False)
        st.success("Trade created")

# -------- TAB 2: OPEN TRADE --------
with tab2:
    trades = pd.read_csv(TRADES_CSV)
    if trades.empty:
        st.info("No trades yet")
    else:
        pick = st.selectbox("Select trade", trades["trade_id"])
        t = trades[trades["trade_id"] == pick].iloc[0]

        col1,col2 = st.columns(2)

        with col1:
            if t["screenshot"] and os.path.exists(t["screenshot"]):
                st.image(Image.open(t["screenshot"]), use_container_width=True)

        with col2:
            raw = fetch_chain()
            df = chain_df(raw, t["expiry"])
            mtm = calc_mtm(t, df)
            if mtm:
                st.metric("MTM (₹)", f"{mtm:,.0f}")

                if st.button("Log today"):
                    d = pd.read_csv(DAILY_CSV)
                    d = pd.concat([d, pd.DataFrame([{
                        "date": str(date.today()),
                        "trade_id": t["trade_id"],
                        "mtm": mtm
                    }])])
                    d.to_csv(DAILY_CSV, index=False)
                    st.success("Logged")

            hist = pd.read_csv(DAILY_CSV)
            hist = hist[hist["trade_id"] == t["trade_id"]]
            if not hist.empty:
                hist["date"] = pd.to_datetime(hist["date"])
                st.line_chart(hist.set_index("date")["mtm"])

# -------- TAB 3: LEADERBOARD --------
with tab3:
    daily = pd.read_csv(DAILY_CSV)
    if daily.empty:
        st.info("No data yet")
    else:
        last = daily.sort_values("date").groupby("trade_id").tail(1)
        st.subheader("Top Performers")
        st.dataframe(last.sort_values("mtm", ascending=False))
