
import json
import numpy as np
import pandas as pd
import n30_lib as L

rng = np.random.default_rng(42)
OUT = L.RES
DEV_END, VAL_END = "2022-12-30", "2024-12-31"
DV_LO = "2017-01-01"
HO_LO, HO_HI = "2025-01-01", "2026-10-30"
IDX = L.close("SPY").index
rf, SPY = L.rf_daily(), L.close("SPY")


def adj(t):
    return L.close(t, adj=True).reindex(IDX)


def raw(t):
    return L.close(t).reindex(IDX)


def nxt_ret(t):
    return L.close(t).pct_change().reindex(IDX).shift(-1)


FOLLOWERS = ["SPY", "QQQ", "IWM", "XLI", "XLB", "XLE", "XLF", "EWJ", "EEM"]
nxt = {t: nxt_ret(t) for t in FOLLOWERS}

#  signal
def spec_sig(kind, param, direction):
    """Return boolean Series on IDX. direction in {'ge','le'}."""
    if kind == "USD":
        r = adj("UUP").pct_change(param, fill_method=None)
        return r >= 0.01 if direction == "ge" else r <= -0.01
    if kind == "RATES":
        d10 = L.frame("DGS10")["close"].diff().reindex(IDX)
        # FRED is decimal percent (e.g. 4.5); a 5bp move = 0.05
        return d10 >= 0.05 if direction == "ge" else d10 <= -0.05
    if kind == "CREDIT":
        r = adj("HYG").pct_change(param, fill_method=None)
        return r <= -0.01 if direction == "le" else r >= 0.01
    if kind in ("EUROPE", "ASIA", "GOLD", "SEMIS"):
        tkr = {"EUROPE": ("EWG", "EWU"), "ASIA": ("EWJ", "EWY"), "GOLD": ("GLD",),
               "SEMIS": ("SMH",)}[kind][param]
        r = adj(tkr).pct_change(1) if kind != "SEMIS" else adj(tkr).pct_change(1)
        thr = 0.025 if kind == "SEMIS" else 0.015
        if kind == "GOLD":
            thr = 0.015
        return r >= thr if direction == "ge" else r <= -thr
    if kind == "BANKS":  # 5d relative strength vs SPY
        rs = adj("XLF").pct_change(5, fill_method=None) - adj("SPY").pct_change(5, fill_method=None)
        return rs >= 0.015 if direction == "ge" else rs <= -0.015
    if kind == "DEFENSIVE":  # XLU 5d relative strength vs SPY (risk-off)
        rs = adj("XLU").pct_change(5, fill_method=None) - adj("SPY").pct_change(5, fill_method=None)
        return rs >= 0.01 if direction == "ge" else rs <= -0.01
    if kind == "ENERGY":
        r = adj("XLE").pct_change(5, fill_method=None)
        return r >= 0.02 if direction == "ge" else r <= -0.02
    if kind == "VOL":
        v = L.close("VIX").reindex(IDX).pct_change()
        return v >= 0.10 if direction == "ge" else v <= -0.10
    raise ValueError(kind)


SPECS = []
SPECS.append(("USD_5d", "USD", 5))
SPECS.append(("RATES_1d", "RATES", None))
SPECS.append(("CREDIT_3d", "CREDIT", 3))
SPECS.append(("SEMIS_1d", "SEMIS", 0))
SPECS.append(("BANKS_5d_rs", "BANKS", 5))
SPECS.append(("DEFENSIVE_5d_rs", "DEFENSIVE", 5))
SPECS.append(("ENERGY_5d", "ENERGY", 5))
SPECS.append(("VOL_1d", "VOL", 1))
for i, tkr in enumerate(["EWG", "EWU"]):
    SPECS.append((f"EUROPE_1d_{tkr}", "EUROPE", i))
for i, tkr in enumerate(["EWJ", "EWY"]):
    SPECS.append((f"ASIA_1d_{tkr}", "ASIA", i))
SPECS.append(("GOLD_1d", "GOLD", 0))

rows = []
for spec_name, kind, param in SPECS:
    for direction in ("ge", "le"):
        try:
            sig = spec_sig(kind, param, direction).fillna(False)
        except Exception as e:
            continue
        if sig.sum() < 30:
            continue
        dev_mask = pd.Series(IDX <= pd.Timestamp(DEV_END), index=IDX)
        for foll in FOLLOWERS:
            x = nxt[foll][sig & dev_mask].dropna()
            if len(x) >= 60:
                rows.append(dict(spec=spec_name, direction=direction, follower=foll,
                                 n=len(x), mean_bp=round(10000 * float(x.mean()), 1),
                                 t=round(float(L.hac_t(x.values)), 2)))

grid = pd.DataFrame(rows).sort_values("t", ascending=False)
grid.to_csv(f"{OUT}/round5_crossmarket_grid.csv", index=False)
print("Cross-market cells with dev t>=2:")
print(grid[grid.t >= 2.0].to_string(index=False))

top = grid[grid.n >= 60].head(12)

results = []
for _, r in top.iterrows():
    sig = spec_sig(dict(SPECS)[r.spec] if False else None, None, None) if False else None
    # rebuild signal
    kind_param = {s[0]: (s[1], s[2]) for s in SPECS}[r.spec]
    sig = spec_sig(kind_param[0], kind_param[1], r.direction).fillna(False)
    foll = r.follower
    pos = pd.DataFrame({foll: sig.astype(float)}, index=IDX)
    net = L.run_system(pos, mode="cc")["daily"]
    m_dv = L.metrics(L.seg(net, DV_LO, VAL_END), rf=rf, spy=SPY)
    m_val = L.metrics(L.seg(net, "2023-01-01", VAL_END), rf=rf, spy=SPY)
    m_ho = L.metrics(L.seg(net, HO_LO, HO_HI), rf=rf, spy=SPY)
    # event
    def ev_t(sig, lo, hi):
        idx = sig.index
        m = np.array((idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi)))
        x = nxt[foll][(sig & m).fillna(False)].dropna()
        return dict(n=len(x), mean_bp=round(10000 * float(x.mean()), 1) if len(x) else None,
                    t=round(float(L.hac_t(x.values)), 2) if len(x) > 20 else None)
    results.append({
        "cell": f"{r.spec}_{r.direction}->{foll}",
        "dev": {"n": int(r.n), "mean_bp": r.mean_bp, "t": r.t},
        "val_event": ev_t(sig, "2023-01-01", VAL_END),
        "holdout_event": ev_t(sig, HO_LO, HO_HI),
        "tradeable_devval": {k: m_dv.get(k) for k in ["sharpe_raw", "t_hac", "alpha_ann_pct", "alpha_t", "maxdd_pct"]},
        "tradeable_holdout": {k: m_ho.get(k) for k in ["sharpe_raw", "t_hac", "alpha_ann_pct", "alpha_t"]},
    })
    print(f"\n{results[-1]['cell']}")
    print("  dev:", results[-1]["dev"], "| val:", results[-1]["val_event"],
          "| holdout:", results[-1]["holdout_event"])
    print("  tradeable dv:", results[-1]["tradeable_devval"])
    print("  tradeable ho:", results[-1]["tradeable_holdout"])

# consistency 
cons = sum(1 for r in results
           if (r["val_event"]["t"] or 0) > 0 and (r["holdout_event"]["t"] or 0) > 0
           and r["dev"]["t"] > 0)
res = {
    "grid_cells": len(grid),
    "dev_t2_count": int((grid.t >= 2).sum()),
    "top12": results,
    "sign_consistent_dev_val_holdout": cons,
}
json.dump(res, open(f"{OUT}/round5_crossmarket_decomposition.json", "w"), indent=1, default=float)
print(f"\n{cons}/12 top cells keep the same sign dev->val->holdout")
print("saved round5_crossmarket_decomposition.json")
