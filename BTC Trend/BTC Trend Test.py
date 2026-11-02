import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats as sps
import n30_lib as L
import n31_strategies as S

OUT = L.RES
rf, spy = L.rf_daily(), L.close("SPY")

# ------------------------------------------------------------------ builders
def b_pos(sig, tkr):
    return pd.DataFrame({tkr: sig.astype(float)}, index=sig.index)

def f_bsd(cfg):
    btc, spy_r, vix = L.ret("BTC"), L.ret("SPY"), L.close("VIX")
    base = (btc > 0.02) & (spy_r < 0.003)
    idx = S._idx("BTC")
    base = base.reindex(idx).fillna(False)
    if cfg == "spy":
        return b_pos(base & (vix.reindex(idx) < 40), "SPY"), "intraday"
    if cfg == "qqq":
        return b_pos(base & (vix.reindex(idx) < 40), "QQQ"), "intraday"
    if cfg == "spy_cc":
        return b_pos(base & (vix.reindex(idx) < 40), "SPY"), "cc"
    if cfg == "qqq_cc":
        return b_pos(base & (vix.reindex(idx) < 40), "QQQ"), "cc"
    if cfg == "spy_ma":
        ma = L.close("SPY", adj=True).rolling(200).mean()
        return b_pos(base & (vix.reindex(idx) < 40) & (L.close("SPY").reindex(idx) > ma.reindex(idx)), "SPY"), "intraday"
    if cfg == "eth":
        return b_pos(base & (vix.reindex(idx) < 40), "ETH"), "cc"

def f_eqdip(cfg):
    thr, hold, mode = cfg
    r5 = L.close("SPY").pct_change(5, fill_method=None)
    sig = (r5 <= thr)
    pos = pd.DataFrame(0.0, index=r5.index, columns=["SPY"])
    sv, hold_until = sig.values, -1
    for i in range(len(pos)):
        if sv[i]:
            hold_until = i + hold
        pos.iloc[i, 0] = 1.0 if i < hold_until else 0.0
    return pos, mode

def f_vixspike(cfg):
    thr, mode = cfg
    dv = L.close("VIX").pct_change(fill_method=None)
    return b_pos(dv >= thr, "SPY"), mode

def _rank_rot(tickers, lb, top, reb):
    idx = S._idx(*tickers)
    mom = pd.DataFrame({t: S._pct(t, lb) for t in tickers}).reindex(idx)
    pos = pd.DataFrame(0.0, index=idx, columns=tickers)
    rebal = (np.arange(len(idx)) % reb) == 0
    cur = []
    for i, d in enumerate(idx):
        if rebal[i]:
            m = mom.loc[d].dropna()
            cur = m.sort_values(ascending=False).index[:top] if len(m) >= max(2 * top, 4) else []
        for t in tickers:
            pos.loc[d, t] = (1.0 / top) if t in cur else 0.0
    return pos

def f_sectrot(cfg):
    lb, top, reb = cfg
    return _rank_rot(L.SECTORS, lb, top, reb), "cc"

def f_ctrymom(cfg):
    lb, top, reb = cfg
    return _rank_rot(sorted(set(L.COUNTRIES)), lb, top, reb), "cc"

def _pair_pos(a, b, window, zin):
    idx = S._idx(a, b)
    ratio = np.log(L.close(a) / L.close(b)).reindex(idx)
    z = ((ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std()).fillna(0.0).values
    pos = pd.DataFrame(0.0, index=idx, columns=[a, b])
    state, zout = 0, 0.25
    for i in range(len(idx)):
        if state == 0:
            if z[i] <= -zin: state = 1
            elif z[i] >= zin: state = -1
        elif state == 1 and z[i] >= -zout: state = 0
        elif state == -1 and z[i] <= zout: state = 0
        pos.iloc[i, 0] = 0.5 * state
        pos.iloc[i, 1] = -0.5 * state
    return pos

def f_pair(cfg):
    (a, b), win, zin = cfg
    return _pair_pos(a, b, win, zin), "cc"

def f_dip(tkr, cfg):
    thr, look, hold = cfg
    r = L.close(tkr).pct_change(look, fill_method=None)
    pos = pd.DataFrame(0.0, index=r.index, columns=[tkr])
    sv, hold_until = (r <= thr).values, -1
    for i in range(len(pos)):
        if sv[i]:
            hold_until = i + hold
        pos.iloc[i, 0] = 1.0 if i < hold_until else 0.0
    return pos, "cc"

def f_gldtr(cfg):
    up, dn = cfg
    cl = L.close("GLD")
    hi, lo = cl.shift(1).rolling(up).max(), cl.shift(1).rolling(dn).min()
    pos = pd.DataFrame(0.0, index=cl.index, columns=["GLD"])
    state = 0
    for i in range(len(cl)):
        if state == 0 and cl.iloc[i] > hi.iloc[i]: state = 1
        elif state == 1 and cl.iloc[i] < lo.iloc[i]: state = 0
        pos.iloc[i, 0] = float(state)
    return pos, "cc"

def f_fxmom(cfg):
    lb, reb = cfg
    idx = S._idx(*L.FX_ETFS)
    mom = pd.DataFrame({t: S._pct(t, lb) for t in L.FX_ETFS}).reindex(idx)
    pos = pd.DataFrame(0.0, index=idx, columns=L.FX_ETFS)
    rebal = (np.arange(len(idx)) % reb) == 0
    cur = None
    for i, d in enumerate(idx):
        if rebal[i]:
            m = mom.loc[d].dropna()
            cur = m.idxmax() if len(m) >= 5 else None
        for t in L.FX_ETFS:
            pos.loc[d, t] = 1.0 if t == cur else 0.0
    return pos, "cc"

def f_btctrend(cfg):
    lb, volcap, voltgt = cfg
    cl = L.close("BTC")
    r = cl.pct_change(lb, fill_method=None)
    dvol = cl.pct_change(fill_method=None).rolling(30).std() * np.sqrt(365)
    sig = (r > 0)
    if volcap:
        sig = sig & (dvol < volcap)
    w = pd.Series(1.0, index=cl.index)
    if voltgt:
        w = (voltgt / dvol).clip(upper=1.0)
    pos = pd.DataFrame({"BTC": (sig.astype(float) * w).fillna(0.0)})
    return pos.reindex(S._idx("BTC")).fillna(0.0), "cc"

def f_cryptoxs(cfg):
    lb, top, minavail = cfg
    idx = S._idx("BTC")
    mom = pd.DataFrame({t: S._pct(t, lb) for t in L.COINS}).reindex(idx)
    pos = pd.DataFrame(0.0, index=idx, columns=L.COINS)
    rebal = (np.arange(len(idx)) % 5) == 0
    cur = []
    for i, d in enumerate(idx):
        if rebal[i]:
            m = mom.loc[d].dropna()
            cur = m.sort_values(ascending=False).index[:top] if len(m) >= minavail else []
        for t in L.COINS:
            pos.loc[d, t] = (1.0 / top) if t in cur else 0.0
    return pos, "cc"

def f_stockrev(cfg):
    nlos, hold, mindrop = cfg
    idx = S._idx(*L.MEGACAPS)
    mom = pd.DataFrame({t: S._pct(t, 5) for t in L.MEGACAPS}).reindex(idx)
    pos = pd.DataFrame(0.0, index=idx, columns=L.MEGACAPS)
    for i, d in enumerate(idx):
        m = mom.loc[d].dropna()
        if len(m) >= 16:
            lo = m.sort_values()
            sel = lo.index[:nlos]
            if mindrop is not None:
                sel = [t for t in sel if lo[t] <= mindrop]
            for t in sel:
                pos.loc[d, t] = 1.0 / max(len(sel), 1)
    return pos, "cc"

def f_volcomp(cfg):
    thr, tkr = cfg
    dv5 = L.close("VIX").pct_change(5, fill_method=None)
    return b_pos(dv5 <= thr, tkr), "cc"

def f_cryptobsd(cfg):
    btc, spy_r, vix = L.ret("BTC"), L.ret("SPY"), L.close("VIX")
    base = ((btc > 0.02) & (spy_r < 0.003) & (vix < 40)).reindex(S._idx("BTC")).fillna(False)
    if cfg == "eth":
        return b_pos(base, "ETH"), "cc"
    if cfg == "sol":
        return b_pos(base, "SOL"), "cc"

# grids
FAMILIES = {
    "BSD":      (f_bsd, ["spy", "qqq", "spy_cc", "qqq_cc", "spy_ma", "eth"]),
    "EQ_DIP":   (f_eqdip, [(-0.03, 1, "cc"), (-0.04, 1, "cc"), (-0.05, 1, "cc"), (-0.03, 3, "cc"),
                           (-0.04, 2, "cc"), (-0.03, 1, "intraday")]),
    "VIX_SPIKE":(f_vixspike, [(0.10, "cc"), (0.15, "cc"), (0.20, "cc"), (0.15, "intraday"), (0.10, "intraday")]),
    "SECT_ROT": (f_sectrot, [(21, 2, 5), (63, 2, 5), (21, 3, 5), (63, 3, 5), (21, 2, 10), (10, 2, 5)]),
    "CTRY_MOM": (f_ctrymom, [(63, 2, 5), (21, 2, 5), (63, 3, 5)]),
    "PAIR_RV":  (f_pair, [(("GLD", "SLV"), 60, 1.75), (("GLD", "SLV"), 90, 2.0), (("GDX", "GLD"), 60, 1.75),
                          (("QQQ", "SPY"), 60, 1.75), (("QQQ", "SPY"), 90, 2.0), (("IEF", "TLT"), 60, 1.75),
                          (("BTC", "ETH"), 60, 1.75)]),
    "TLT_DIP":  (lambda c: f_dip("TLT", c), [(-0.025, 10, 5), (-0.035, 10, 5), (-0.05, 10, 10), (-0.035, 10, 10),
                                              (-0.05, 20, 20), (-0.07, 20, 20)]),
    "HYG_DIP":  (lambda c: f_dip("HYG", c), [(-0.015, 5, 3), (-0.02, 5, 5), (-0.03, 5, 5), (-0.015, 5, 5)]),
    "GLD_TREND":(f_gldtr, [(20, 10), (50, 20), (100, 50)]),
    "FX_MOM":   (f_fxmom, [(21, 5), (63, 5), (21, 10), (63, 10)]),
    "BTC_TREND":(f_btctrend, [(30, 0.90, None), (60, None, None), (90, None, None), (30, 0.90, 0.60),
                              (60, 0.90, None), (30, None, None)]),
    "CRYPTO_XS":(f_cryptoxs, [(7, 2, 5), (21, 2, 5), (7, 3, 5), (30, 2, 5), (7, 2, 4)]),
    "STOCK_REV":(f_stockrev, [(4, 1, None), (4, 1, -0.05), (5, 1, None), (3, 1, -0.04)]),
    "VOL_COMP": (f_volcomp, [(-0.15, "QQQ"), (-0.20, "QQQ"), (-0.15, "SPY"), (-0.10, "QQQ")]),
    "CRYPTO_BSD":(f_cryptobsd, ["eth", "sol"]),
}

if __name__ == "__main__":
    rows, picks = [], {}
    for fam, (builder, cfgs) in FAMILIES.items():
        best = None
        for cfg in cfgs:
            pos, mode = builder(cfg)
            res = L.run_system(pos, mode=mode)
            net = res["daily"]
            dev = L.seg(net, hi=L.DEV_END)
            m = L.metrics(dev)
            ntr = len(res["ledger"][res["ledger"]["entry"] <= L.DEV_END]) if len(res["ledger"]) else 0
            rows.append({"family": fam, "cfg": str(cfg), "dev_n": m["n_days"],
                         "dev_ret": m["ann_ret_pct"], "dev_sharpe": m["sharpe_raw"],
                         "dev_t": m["t_hac"], "dev_maxdd": m["maxdd_pct"], "dev_trades": ntr})
            if (m["sharpe_raw"] is not None and (m["sharpe_raw"] == m["sharpe_raw"])
                    and ntr >= 15 and (best is None or m["sharpe_raw"] > best[1])):
                best = (cfg, m["sharpe_raw"])
        if best is None:
            best = (cfgs[0], np.nan)
        pos, mode = builder(best[0])
        res = L.run_system(pos, mode=mode)
        net = res["daily"]
        mdev = L.metrics(L.seg(net, hi=L.DEV_END))
        mval = L.metrics(L.seg(net, lo="2023-01-01", hi=L.VAL_END))
        ntr_d = len(res["ledger"][res["ledger"]["entry"] <= L.DEV_END]) if len(res["ledger"]) else 0
        p = 1 - sps.norm.cdf(mdev["t_hac"]) if mdev["t_hac"] == mdev["t_hac"] else 1.0
        status = ("survive" if (mdev["t_hac"] >= 2.0 and (mval["ann_ret_pct"] or -1) > 0 and ntr_d >= 20)
                  else ("border" if (mdev["t_hac"] >= 1.5 and (mval["ann_ret_pct"] or -1) > 0) else "reject"))
        picks[fam] = {"cfg": str(best[0]), "dev_t": mdev["t_hac"], "dev_sharpe": mdev["sharpe_raw"],
                      "val_ret": mval["ann_ret_pct"], "val_sharpe": mval["sharpe_raw"], "val_t": mval["t_hac"],
                      "dev_trades": ntr_d, "status": status, "p": p, "mode": mode}
        print(f"{fam:11s} pick={str(best[0]):28s} dev_t={mdev['t_hac']!s:>5} val_ret={mval['ann_ret_pct']!s:>6} "
              f"val_t={mval['t_hac']!s:>5} trades={ntr_d} -> {status}")

    pd.DataFrame(rows).to_csv(f"{OUT}/discovery2_grid_all.csv", index=False)
    pk = pd.DataFrame(picks).T.sort_values("p")
    m = len(pk)
    pk["holm_p"] = np.maximum.accumulate((m * pk["p"]).values) if m else []
    pk.to_csv(f"{OUT}/discovery2_picks.csv")
    json.dump({k: {kk: (vv if vv == vv else None) for kk, vv in v.items()} for k, v in picks.items()},
              open(f"{OUT}/discovery2_picks.json", "w"), indent=1)
    print("\nSURVIVE:", [k for k, v in picks.items() if v["status"] == "survive"])
    print("BORDER:", [k for k, v in picks.items() if v["status"] == "border"])
