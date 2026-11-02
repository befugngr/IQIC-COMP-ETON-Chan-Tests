import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = "/home/z/my-project/research2"
DATA, RES = os.path.join(BASE, "data"), os.path.join(BASE, "results")

def load(t):
    df = pd.read_csv(os.path.join(DATA, f"nsd_{t}.csv"), parse_dates=["date"]).set_index("date").sort_index()
    df["ret"] = df["close"].pct_change()
    df["on"] = df["open"] / df["close"].shift(1) - 1
    df["id"] = df["close"] / df["open"] - 1
    return df

btc = pd.read_csv(os.path.join(DATA, "cb_BTC_USD.csv"), parse_dates=["date"]).set_index("date").sort_index()
btc["ret"] = btc["close"].pct_change()
spy = load("SPY")
vix = pd.read_csv(os.path.join(DATA, "cboe_VIX.csv"), parse_dates=["DATE"]).set_index("DATE").sort_index()
dtb3 = pd.read_csv(os.path.join(DATA, "fred_DGS3MO.csv"), parse_dates=["observation_date"]).rename(
    columns={"observation_date": "date", "DGS3MO": "rf"}).set_index("date")
dtb3["rf"] = pd.to_numeric(dtb3["rf"], errors="coerce") / 100 / 252  # daily rf
dtb3 = dtb3.dropna()

J = pd.DataFrame({
    "btc": btc["ret"], "spy": spy["ret"], "spy_on": spy["on"], "spy_id": spy["id"],
    "spy_open": spy["open"], "spy_close": spy["close"],
    "vix": vix["CLOSE"],
}).dropna(subset=["btc", "spy", "spy_on", "spy_id"])
J["rf"] = dtb3["rf"].reindex(J.index)
J["rf"] = J["rf"].fillna(0.0)
J["vol20"] = J["spy"].rolling(20).std() * np.sqrt(252)
J = J.dropna(subset=["vol20"])

DEV_END = "2021-06-30"
S, E = "2017-07-01", "2026-10-30"
J = J.loc[S:E]

def simulate(J, thr=0.02, spy_cap=0.003, gate=None, cost_bp_side=2.0):
    """returns daily strategy net returns (executable: open->close on day t+1)"""
    sig = (J["btc"] > thr) & (J["spy"] < spy_cap)
    sig = sig.shift(1).fillna(False)          # position active on day t+1
    if gate == "vix40":
        ok = (J["vix"].shift(1) < 40)
    elif gate == "vol40":
        ok = (J["vol20"].shift(1) < 0.40)
    else:
        ok = pd.Series(True, index=J.index)
    pos = (sig & ok).astype(float)
    trade = pos.diff().abs().fillna(pos.iloc[0])
    cost = trade * (cost_bp_side / 1e4)       # cost charged on entry and exit days
    r = pos * J["spy_id"] + (1 - pos) * J["rf"] - cost
    return r, pos

candidates = {
    "none": dict(gate=None),
    "vix40": dict(gate="vix40"),
    "vol40": dict(gate="vol40"),
}
dev = J.loc[:DEV_END]
print("=== overlay selection on DEV (Sharpe of daily net returns) ===")
best, best_sh = None, -9
for name, kw in candidates.items():
    r, pos = simulate(dev, **kw)
    sh = r.mean() / r.std() * np.sqrt(252)
    n_tr = int(pos.sum())
    print(f"{name}: dev Sharpe {sh:.2f}, trades {n_tr}")
    if sh > best_sh:
        best_sh, best = sh, name
print(f"chosen overlay: {best} (dev Sharpe {best_sh:.2f})\n")

# loss test 
def simulate_breaker(J, thr=0.02, spy_cap=0.003, k=3, cost_bp_side=2.0):
    sig_raw = ((J["btc"] > thr) & (J["spy"] < spy_cap)).shift(1).fillna(False)
    pos = pd.Series(0.0, index=J.index)
    losses = 0
    active_prev = False
    rets = []
    for i, (dt, row) in enumerate(J.iterrows()):
        if sig_raw.iloc[i]:
            if losses >= k:
                pos.iloc[i] = 0.0
            else:
                pos.iloc[i] = 1.0
        else:
            pos.iloc[i] = 0.0
    # compute with loop over trades
    pos = pd.Series(0.0, index=J.index)
    losses = 0
    for i in range(len(J)):
        if sig_raw.iloc[i] and losses < k:
            pos.iloc[i] = 1.0
            r_trade = J["spy_id"].iloc[i]
            if r_trade < 0:
                losses += 1
            else:
                losses = 0
    trade = pos.diff().abs().fillna(pos.iloc[0])
    cost = trade * (cost_bp_side / 1e4)
    r = pos * J["spy_id"] + (1 - pos) * J["rf"] - cost
    return r, pos

r_br, pos_br = simulate_breaker(dev)
sh_br = r_br.mean() / r_br.std() * np.sqrt(252)
print(f"breaker k=3 dev Sharpe {sh_br:.2f} vs plain {best_sh:.2f} -> {'ADOPT' if sh_br > best_sh + 0.1 else 'REJECT (no dev improvement)'}")

# config 
r_net, pos = simulate(J, **candidates[best])
r_gross, _ = simulate(J, **candidates[best], cost_bp_side=0.0)

nav = (1 + r_net).cumprod()
n_years = len(r_net) / 252
cagr = nav.iloc[-1] ** (1 / n_years) - 1
vol = r_net.std() * np.sqrt(252)
sharpe = (r_net.mean() - J["rf"].mean()) / r_net.std() * np.sqrt(252)
mdd = (nav / nav.cummax() - 1).min()
spy_nav = (1 + J["spy"]).cumprod()
spy_cagr = spy_nav.iloc[-1] ** (1 / n_years) - 1
spy_vol = J["spy"].std() * np.sqrt(252)
spy_mdd = (spy_nav / spy_nav.cummax() - 1).min()

# beta/alpha vs SPY 
X = sm.add_constant(J["spy"])
fit = sm.OLS(r_net, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
beta = fit.params.iloc[1]
alpha_daily = fit.params.iloc[0]
treynor = (r_net.mean() * 252 - J["rf"].mean() * 252) / beta if beta != 0 else np.nan

print("=" * 84)
print(f"FINAL STRATEGY — BTC Shock Diffusion, overlay={best}, cost 2bp/side, {S}..{E}")
print("=" * 84)
print(f"trades: {int(pos.sum())} | exposure: {pos.mean()*100:.1f}% of days")
print(f"CAGR {cagr*100:.2f}% | vol {vol*100:.2f}% | Sharpe {sharpe:.2f} | maxDD {mdd*100:.2f}%")
print(f"alpha vs SPY: {alpha_daily*1e4:.2f}bp/day (HAC t={fit.tvalues.iloc[0]:.2f}) | beta {beta:.3f} | Treynor {treynor:.3f}")
print(f"SPY B&H: CAGR {spy_cagr*100:.2f}% | vol {spy_vol*100:.2f}% | Sharpe {(J['spy'].mean()-J['rf'].mean())/J['spy'].std()*np.sqrt(252):.2f} | maxDD {spy_mdd*100:.2f}%")

# 15-day 
W = 15
rows = []
idx = r_net.index
for i in range(0, len(idx) - W + 1):
    w = slice(i, i + W)
    rs = r_net.iloc[w]
    rb = J["spy"].iloc[w]
    rf_w = J["rf"].iloc[w].mean()
    ann_ret = (1 + rs).prod() ** (252 / W) - 1
    ann_vol = rs.std() * np.sqrt(252)
    shp = (rs.mean() - rf_w) / rs.std() * np.sqrt(252) if rs.std() > 0 else 0.0
    # window CAPM alpha vs SPY
    if rb.std() > 0:
        bw = sm.OLS(rs, sm.add_constant(rb)).fit()
        w_alpha = bw.params.iloc[0] * 252
        w_beta = bw.params.iloc[1]
        tre = (ann_ret - rf_w * 252) / w_beta if abs(w_beta) > 0.02 else np.nan
    else:
        w_alpha, w_beta, tre = np.nan, np.nan, np.nan
    navw = (1 + rs).cumprod()
    rows.append({
        "start": str(idx[i].date()), "n_trades": int(pos.iloc[w].sum()),
        "total_ret_pct": round(((1 + rs).prod() - 1) * 100, 3),
        "ann_sharpe": round(shp, 2), "ann_alpha_pct": round(w_alpha * 100, 2) if pd.notna(w_alpha) else None,
        "beta": round(w_beta, 3) if pd.notna(w_beta) else None,
        "treynor": round(tre, 2) if pd.notna(tre) else None,
        "maxDD_pct": round(((navw / navw.cummax() - 1).min()) * 100, 2),
    })
R = pd.DataFrame(rows)
R.to_csv(os.path.join(RES, "rolling_3wk_windows.csv"), index=False)

zero = (R.n_trades == 0).mean() * 100
print("\n" + "=" * 84)
print(f"ROLLING 3-WEEK WINDOWS (n={len(R)}), StockTrak-style")
print("=" * 84)
print(f"zero-trade windows: {zero:.1f}% | mean trades/window: {R.n_trades.mean():.2f}")
print(f"total return: median {R.total_ret_pct.median():.3f}% | p25 {R.total_ret_pct.quantile(.25):.3f}% | p75 {R.total_ret_pct.quantile(.75):.3f}% | worst {R.total_ret_pct.min():.3f}% | best {R.total_ret_pct.max():.3f}%")
print(f"winning windows: {(R.total_ret_pct > 0).mean()*100:.1f}%")
print(f"window Sharpe: median {R.ann_sharpe.median():.2f} | p10 {R.ann_sharpe.quantile(.1):.2f} | p90 {R.ann_sharpe.quantile(.9):.2f}")
print(f"window alpha: median {R.ann_alpha_pct.median():.2f}% | >0: {(R.ann_alpha_pct > 0).mean()*100:.1f}%")
print(f"window maxDD: median {R.maxDD_pct.median():.2f}% | worst {R.maxDD_pct.min():.2f}%")

# by year
by_year = pd.DataFrame({
    "BSD_ret": r_net.groupby(r_net.index.year).apply(lambda x: (1 + x).prod() - 1),
    "SPY_ret": J["spy"].groupby(J.index.year).apply(lambda x: (1 + x).prod() - 1),
    "n_trades": pos.groupby(pos.index.year).sum(),
})
by_year["excess"] = by_year["BSD_ret"] - by_year["SPY_ret"]
by_year.round(4).to_csv(os.path.join(RES, "bsd_by_year.csv"))
print("\nBY YEAR:")
print(by_year.round(4).to_string())

# cost sensitivity
print("\nCOST SENSITIVITY (full period ann. excess vs SPY B&H):")
for c in [0, 2, 5, 10]:
    rc, _ = simulate(J, **candidates[best], cost_bp_side=float(c))
    exc = (1 + rc).prod() / (1 + J["spy"]).prod() - 1
    shc = rc.mean() / rc.std() * np.sqrt(252)
    print(f"  {c}bp/side: total excess vs SPY {exc*100:+.2f}% | Sharpe {shc:.2f}")

# save NAV + trades
pd.DataFrame({"nav": nav, "spy_nav": spy_nav, "pos": pos}).to_csv(os.path.join(RES, "bsd_nav.csv"))
trades = J.loc[pos > 0, ["spy_id"]].copy()
trades.columns = ["exec_intraday_ret"]
trades.to_csv(os.path.join(RES, "bsd_trades.csv"))

json.dump({
    "overlay": best, "n_trades": int(pos.sum()), "exposure_pct": round(pos.mean() * 100, 2),
    "cagr_pct": round(cagr * 100, 2), "vol_pct": round(vol * 100, 2), "sharpe": round(sharpe, 2),
    "maxdd_pct": round(mdd * 100, 2), "alpha_bp_day": round(alpha_daily * 1e4, 2),
    "alpha_t": round(float(fit.tvalues.iloc[0]), 2), "beta": round(float(beta), 3),
    "zero_trade_windows_pct": round(zero, 1), "mean_trades_per_3wk": round(R.n_trades.mean(), 2),
    "winning_windows_pct": round((R.total_ret_pct > 0).mean() * 100, 1),
    "median_window_ret_pct": round(R.total_ret_pct.median(), 3),
    "p25_window_ret_pct": round(R.total_ret_pct.quantile(.25), 3),
    "p75_window_ret_pct": round(R.total_ret_pct.quantile(.75), 3),
    "worst_window_ret_pct": round(R.total_ret_pct.min(), 3),
}, open(os.path.join(RES, "bsd_summary.json"), "w"), indent=1)
print("\nsaved rolling_3wk_windows.csv, bsd_nav.csv, bsd_trades.csv, bsd_by_year.csv, bsd_summary.json")
