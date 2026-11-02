import json
import os
import numpy as np
import pandas as pd
import n30_lib as L

R3 = "/home/z/my-project/research3"
OUT = f"{R3}/results/sandbox"
os.makedirs(OUT, exist_ok=True)
rf = L.rf_daily()
SPY = L.frame("SPY")
IDX = SPY.index
BTC = L.frame("BTC")
CPER = L.frame("CPER")
CATF = L.frame("CAT")
VIXC = L.close("VIX")

SLIP = {"etf": 2.0, "stock": 2.5, "crypto": 10.0}   # bp per side


def btc_close_at(d):
    s = BTC["close"]
    return float(s.loc[d]) if d in s.index else np.nan



bsd_fire, btc_trend, copper_on = {}, {}, {}
cper_adj = CPER["adjclose"].astype(float)
for t in IDX:
    # BSD (unchanged)
    b1 = btc_close_at(t)
    b0 = btc_close_at(t - pd.Timedelta(days=1))
    spy_r = SPY["close"].pct_change().loc[t] if t in SPY.index else np.nan
    vix = VIXC.loc[t] if t in VIXC.index else np.nan
    bsd_fire[t] = bool((b1 / b0 - 1.0) > 0.02) and bool(spy_r < 0.003) and bool(vix < 40) \
        if (b1 == b1 and b0 == b0 and spy_r == spy_r and vix == vix) else False
    # BTC_TREND (unchanged)
    b60 = btc_close_at(t - pd.Timedelta(days=60))
    btc_trend[t] = bool((b1 / b60 - 1.0) > 0) if (b1 == b1 and b60 == b60 and b60 > 0) else False
    # COPPER_DEMAND: CPER 5-SESSION adj return >= +1.5%
    if t in cper_adj.index:
        i = cper_adj.index.get_loc(t)
        if i >= 5:
            r5 = cper_adj.iloc[i] / cper_adj.iloc[i - 5] - 1.0
            copper_on[t] = bool(r5 >= 0.015)
        else:
            copper_on[t] = False
    else:
        copper_on[t] = False
print(f"signals: BSD fire days={sum(bsd_fire.values())}, "
      f"BTC trend on-days={sum(1 for v in btc_trend.values() if v)}, "
      f"COPPER on-days={sum(1 for v in copper_on.values() if v)}")


# ----------------sanddbox ----------------
def run(window, scenario, lev=1.0, start_cash=100_000.0):
    lo, hi = window
    days = IDX[(IDX >= lo) & (IDX <= hi)]
    comm_per_trade = 10.0 if scenario == "fee10" else 0.0
    W = {"BSD": 0.25 * lev, "BTC": 0.20 * lev, "COPPER": 0.35 * lev}

    cash = start_cash
    btc_units = 0.0
    cat_shares = 0
    spy_bh_shares = 0
    spy_bh_cash = start_cash
    trades, daily = [], []
    eq_prev = start_cash
    peak = start_cash

    def side_cost(notional, kind):
        return notional * SLIP[kind] / 1e4

    def buy(t, tk, kind, budget, frac=False, sleeve="", why=""):
        f = BTC if tk == "BTC" else (SPY if tk == "SPY" else (CATF if tk == "CAT" else None))
        if f is None or t not in f.index:
            return 0.0, 0.0
        px = float(f["open"].loc[t])
        if not (px == px) or px <= 0:
            return 0.0, 0.0
        qty = (budget / px) if frac else np.floor(budget / px)
        if frac:
            qty = round(qty, 6)
        if qty <= 0:
            return 0.0, 0.0
        notion = qty * px
        slip = side_cost(notion, kind)
        c = comm_per_trade
        cash_d = notion + slip + c
        trades.append({"date": t.date().isoformat(), "time": "open", "ticker": tk, "side": "buy",
                       "units": round(qty, 6), "price": round(px, 4), "notional_$": round(notion, 2),
                       "commission_$": c, "slippage_$": round(slip, 2), "sleeve": sleeve, "reason": why})
        return qty, cash_d

    def sell(t, tk, kind, qty, sleeve="", why="", time_="close"):
        f = BTC if tk == "BTC" else (SPY if tk == "SPY" else (CATF if tk == "CAT" else None))
        if f is None or (tk == "BTC" and t not in f.index):
            return 0.0
        px = float(f["close"].loc[t]) if (tk == "BTC" and time_ == "close") else (
            float(f["close"].loc[t]) if t in f.index else np.nan)
        if time_ == "open":
            px = float(f["open"].loc[t]) if t in f.index else np.nan
        if not (px == px) or px <= 0:
            return 0.0
        notion = qty * px
        slip = side_cost(notion, kind)
        c = comm_per_trade
        trades.append({"date": t.date().isoformat(), "time": time_, "ticker": tk, "side": "sell",
                       "units": round(qty, 6), "price": round(px, 4), "notional_$": round(notion, 2),
                       "commission_$": c, "slippage_$": round(slip, 2), "sleeve": sleeve, "reason": why})
        return notion - slip - c

    for i, t in enumerate(days):
        sig_bsd = bsd_fire.get(days[i - 1], False) if i > 0 else False
        sig_bt = btc_trend.get(days[i - 1], False) if i > 0 else None
        sig_cu = copper_on.get(days[i - 1], False) if i > 0 else False

        # ---- benchmark
        if i == 0 and t in SPY.index:
            px = float(SPY["open"].loc[t])
            spy_bh_shares = int(np.floor(spy_bh_cash / px))
            cost = spy_bh_shares * px
            spy_bh_cash -= cost + side_cost(cost, "etf") + comm_per_trade

        # ---- morning: sync BTC
        if sig_bt is not None:
            if sig_bt and btc_units <= 0:
                qty, dc = buy(t, "BTC", "crypto", W["BTC"] * eq_prev, frac=True,
                              sleeve="BTC_TREND", why="BTC 60d return>0 at prev close")
                if qty > 0:
                    btc_units = qty
                    cash -= dc
            elif (not sig_bt) and btc_units > 0:
                proceeds = sell(t, "BTC", "crypto", btc_units, sleeve="BTC_TREND",
                                why="BTC 60d return<=0 at prev close", time_="open")
                cash += proceeds
                btc_units = 0.0

        # ---- morning: sync CAT (COPPER_DEMAND, persistent position)
        if i > 0:
            if sig_cu and cat_shares == 0:
                qty, dc = buy(t, "CAT", "stock", W["COPPER"] * eq_prev,
                              sleeve="COPPER_DEMAND",
                              why="CPER 5d adj return>=+1.5% at prev close")
                if qty > 0:
                    cat_shares = int(qty)
                    cash -= dc
            elif (not sig_cu) and cat_shares > 0:
                proceeds = sell(t, "CAT", "stock", cat_shares, sleeve="COPPER_DEMAND",
                                why="CPER 5d adj return<+1.5% at prev close", time_="open")
                cash += proceeds
                cat_shares = 0

        # ---- morning 
        intraday = []
        if sig_bsd:
            qty, dc = buy(t, "SPY", "etf", W["BSD"] * eq_prev, sleeve="BSD",
                          why="BTC>+2% & SPY<+0.3% & VIX<40 at prev close")
            if qty > 0:
                cash -= dc
                intraday.append(("SPY", "etf", qty))

        # ---- close
        for tk, kind, qty in intraday:
            cash += sell(t, tk, kind, qty, sleeve="BSD", why="intraday exit at close")
        btc_px_close = btc_close_at(t)
        btc_val = btc_units * btc_px_close if (btc_units > 0 and btc_px_close == btc_px_close) else 0.0
        cat_val = cat_shares * float(CATF["close"].loc[t]) if (cat_shares > 0 and t in CATF.index) else 0.0
        spy_close = float(SPY["close"].loc[t]) if t in SPY.index else np.nan

        r_day = float(rf.loc[t]) if t in rf.index else 0.0
        if cash >= 0:
            cash += cash * r_day
            margin_int = 0.0
        else:
            margin_int = -cash * (r_day + 0.005 / 252.0)
            cash += cash * (r_day + 0.005 / 252.0)
        spy_bh_cash += spy_bh_cash * r_day

        equity = cash + btc_val + cat_val
        peak = max(peak, equity)
        daily.append({"date": t.date().isoformat(),
                       "equity_$": round(equity, 2),
                       "cash_$": round(cash, 2),
                       "btc_units": round(btc_units, 6),
                       "btc_value_$": round(btc_val, 2),
                       "cat_shares": cat_shares,
                       "cat_value_$": round(cat_val, 2),
                       "gross_exposure": round((btc_val + cat_val) / equity, 4) if equity > 0 else np.nan,
                       "margin_interest_$": round(margin_int, 2),
                       "cash_interest_$": round(cash * r_day, 2) if cash > 0 else 0.0,
                       "daily_pnl_$": round(equity - eq_prev, 2),
                       "daily_ret": (equity / eq_prev - 1.0) if eq_prev > 0 else np.nan,
                       "spy_buyhold_$": round(spy_bh_cash + spy_bh_shares * spy_close, 2),
                       "drawdown_pct": round(100 * (equity / peak - 1.0), 2)})
        eq_prev = equity

    return pd.DataFrame(daily), pd.DataFrame(trades)


def stats(daily, trades, label):
    r = daily["daily_ret"].dropna()
    eq = daily["equity_$"]
    years = len(r) / 252.0
    out = {"run": label, "start": daily["date"].iloc[0], "end": daily["date"].iloc[-1],
           "start_equity": 100000.0, "final_equity": round(float(eq.iloc[-1]), 2),
           "total_return_pct": round(100 * (eq.iloc[-1] / 1e5 - 1), 2),
           "cagr_pct": round(100 * ((eq.iloc[-1] / 1e5) ** (1 / years) - 1), 2),
           "n_trades": int(len(trades)),
           "commissions_paid_$": round(float(trades["commission_$"].sum()), 2) if len(trades) else 0.0,
           "slippage_paid_$": round(float(trades["slippage_$"].sum()), 2) if len(trades) else 0.0,
           "margin_interest_paid_$": round(float(daily["margin_interest_$"].sum()), 2),
           "cash_interest_earned_$": round(float(daily["cash_interest_$"].sum()), 2),
           "spy_buyhold_final_$": round(float(daily["spy_buyhold_$"].iloc[-1]), 2)}
    m = L.metrics(pd.Series(r.values, index=pd.to_datetime(daily["date"])),
                  rf=rf, spy=L.close("SPY"))
    out.update({k: m.get(k) for k in ["ann_ret_pct", "ann_vol_pct", "sharpe_raw", "sharpe_excess",
                                      "t_hac", "maxdd_pct", "beta", "alpha_ann_pct", "alpha_t",
                                      "treynor_excess", "hit_daily"]})
    w = L.windows15(pd.Series(r.values, index=pd.to_datetime(daily["date"])), spy=L.close("SPY"))
    out["win15"] = w
    yr = daily.copy()
    yr["year"] = yr["date"].str[:4]
    out["per_year_return_pct"] = {y: round(100 * (g["equity_$"].iloc[-1] / g["equity_$"].iloc[0] - 1), 2)
                                  for y, g in yr.groupby("year")}
    return out


WINDOWS = {"full": ("2017-01-03", "2026-10-30"), "holdout": ("2025-01-02", "2026-10-30")}
RUNS = [("full", "free", 1.0), ("full", "fee10", 1.0), ("full", "free", 1.25),
        ("holdout", "free", 1.0), ("holdout", "fee10", 1.0), ("holdout", "free", 1.25)]

perf = {}
for win, scen, lev in RUNS:
    tag = f"{win}_{scen}_{'1x' if lev == 1.0 else '125x'}"
    daily, trades = run(WINDOWS[win], scen, lev)
    if len(trades) == 0:
        print(f"!! no trades in {tag}")
        continue
    daily.to_csv(f"{OUT}/daily_{tag}.csv", index=False)
    trades.to_csv(f"{OUT}/trades_{tag}.csv", index=False)
    perf[tag] = stats(daily, trades, tag)
    p = perf[tag]
    print(f"{tag:18s} final=${p['final_equity']:>12,.2f}  tot={p['total_return_pct']:>8.2f}%  "
          f"CAGR={p['cagr_pct']:>6.2f}%  Sharpe={p['sharpe_raw']}  alpha={p['alpha_ann_pct']}%  "
          f"alpha_t={p['alpha_t']}  maxDD={p['maxdd_pct']}%  trades={p['n_trades']}  "
          f"comm=${p['commissions_paid_$']}  SPY_BH=${p['spy_buyhold_final_$']:,.0f}")

json.dump(perf, open(f"{OUT}/performance.json", "w"), indent=1)
print("\nsaved ->", OUT)
