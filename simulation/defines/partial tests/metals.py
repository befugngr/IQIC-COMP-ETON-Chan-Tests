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

LEADERS = ["CPER", "GLD", "SLV", "GDX", "DBC"]
FOLLOWERS = ["XLI", "XLB", "XLE", "XLY", "IWM", "CAT"]
KS = [1, 3, 5, 10, 20]
THRS = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]

lead_ret = {t: L.close(t, adj=True).reindex(IDX) for t in LEADERS}
foll_ret = {t: L.close(t).pct_change().reindex(IDX) for t in FOLLOWERS}
foll_ret3 = {t: (1 + L.close(t).pct_change().reindex(IDX)).rolling(3).sum() - 1
             for t in FOLLOWERS}

ma200 = L.close("SPY", adj=True).rolling(200).mean()
bull = (L.close("SPY", adj=True) > ma200).reindex(IDX)
vix = L.close("VIX").reindex(IDX)
lowvix = vix < 20
lead60 = {t: lead_ret[t].pct_change(60, fill_method=None) > 0 for t in LEADERS}

rows = []
for lead in LEADERS:
    kr = lead_ret[lead].pct_change()
    for k in KS:
        sig_k = lead_ret[lead].pct_change(k, fill_method=None)
        for thr in THRS:
            base_sig = (sig_k <= thr) if thr <= 0 else (sig_k >= thr)
            for foll in FOLLOWERS:
                # next-day follower return, signal known at close t
                r1 = foll_ret[foll].shift(-1)
                r3 = foll_ret3[foll].shift(-3)
                ev = base_sig.fillna(False)
                # dev-only stats, per regime
                for regime_name, mask in [("all", pd.Series(True, index=IDX)),
                                          ("bull", bull), ("bear", ~bull.fillna(True)),
                                          ("lowVIX", lowvix), ("hiVIX", ~lowvix.fillna(False)),
                                          ("lead60up", lead60[lead]),
                                          ("lead60dn", ~lead60[lead].fillna(False))]:
                    m = (ev & mask.fillna(False)).reindex(IDX).fillna(False)
                    md = m & (IDX <= pd.Timestamp(DEV_END))
                    x = r1[md].dropna()
                    if len(x) >= 60:
                        t = L.hac_t(x.values)
                        rows.append(dict(leader=lead, k=k, thr=thr, follower=foll,
                                         regime=regime_name, n=len(x),
                                         mean_bp=round(10000 * float(x.mean()), 1),
                                         t=round(float(t), 2),
                                         mean3_bp=round(10000 * float(
                                             foll_ret3[foll].shift(-3)[md].dropna().mean()), 1)))

grid = pd.DataFrame(rows)
grid.to_csv(f"{OUT}/round5_metals_cyclicals_grid.csv", index=False)

allcells = grid[(grid.regime == "all")].copy()
allcells["abs_t"] = allcells.t.abs()
top = allcells.sort_values("abs_t", ascending=False).head(25)
print("TOP dev cells (regime=all), any direction:")
print(top[["leader", "k", "thr", "follower", "n", "mean_bp", "t"]].to_string(index=False))

# -------------------------------------------------  shift analysis -
best = top.iloc[0]
print(f"\nBest cell: {best.leader} {best.k}d {'<=' if best.thr<=0 else '>='} {best.thr*100:.0f}% "
      f"-> {best.follower} next day (dev t={best.t})")
regime_view = grid[(grid.leader == best.leader) & (grid.k == best.k) &
                   (grid.thr == best.thr) & (grid.follower == best.follower)]
print(regime_view[["regime", "n", "mean_bp", "t"]].to_string(index=False))

# concentration: how manyhave dev t>=2 by direction
up_cells = allcells[(allcells.thr > 0) & (allcells.t >= 2.0)]
dn_cells = allcells[(allcells.thr < 0) & (allcells.t >= 2.0)]
print(f"\ndev t>=2 cells: metals-UP->cyc {len(up_cells)}, metals-DOWN->cyc {len(dn_cells)}")
if len(up_cells):
    print(up_cells[["leader", "k", "thr", "follower", "n", "mean_bp", "t"]].to_string(index=False))
if len(dn_cells):
    print(dn_cells[["leader", "k", "thr", "follower", "n", "mean_bp", "t"]].to_string(index=False))


# a longonly tradeable rule
cand = allcells[(allcells.t >= 2.0) & (allcells.n >= 60) & (allcells.mean_bp > 0)].copy()
cand = cand.sort_values("t", ascending=False).drop_duplicates(["leader", "follower"]).head(3)
print("\nTradeable candidates:")
print(cand[["leader", "k", "thr", "follower", "n", "mean_bp", "t"]].to_string(index=False))

trade = {}
for _, c in cand.iterrows():
    lead, k, thr, foll = c.leader, int(c.k), float(c.thr), c.follower
    sig_k = lead_ret[lead].pct_change(k, fill_method=None)
    sig = (sig_k >= thr) if thr > 0 else (sig_k <= thr)
    sig = sig.reindex(IDX).fillna(False)
    pos = pd.DataFrame({foll: sig.astype(float)}, index=IDX)
    net = L.run_system(pos, mode="cc")["daily"]
    md = L.metrics(L.seg(net, DV_LO, VAL_END), rf=rf, spy=SPY)
    mv = L.metrics(L.seg(net, "2023-01-01", VAL_END), rf=rf, spy=SPY)
    mh = L.metrics(L.seg(net, HO_LO, HO_HI), rf=rf, spy=SPY)
    # placebo 
    dv_net = L.seg(net, DV_LO, VAL_END)
    obs = float(dv_net.sum() * 1e4)
    ps = []
    for _ in range(2000):
        sh = int(rng.integers(30, len(dv_net) - 30))
        s2 = np.roll(sig.reindex(dv_net.index).fillna(False).values, sh)
        ps.append(float((dv_net.values * s2).sum()) * 1e4)
    p = float((np.array(ps) >= obs).mean())
    key = f"{lead}_{k}d_{'ge' if thr>0 else 'le'}{abs(thr*100):.0f}->{foll}"
    trade[key] = {
        "rule": f"{lead} {k}d ret {'>=' if thr>0 else '<='} {thr*100:.0f}% -> long {foll} next day",
        "devval": {k2: md.get(k2) for k2 in ["sharpe_raw", "t_hac", "alpha_ann_pct", "alpha_t", "ann_ret_pct", "maxdd_pct"]},
        "val_only": {k2: mv.get(k2) for k2 in ["sharpe_raw", "t_hac", "alpha_ann_pct", "alpha_t"]},
        "holdout": {k2: mh.get(k2) for k2 in ["sharpe_raw", "t_hac", "alpha_ann_pct", "alpha_t", "ann_ret_pct"]},
        "placebo_p": round(p, 4),
        "n_sig_days_devval": int(sig.reindex(L.seg(net, DV_LO, VAL_END).index).sum()),
    }
    print(key, "devval:", trade[key]["devval"], "holdout:", trade[key]["holdout"], "placebo p:", p)

res = {
    "Q1_top_cells_dev": top[["leader", "k", "thr", "follower", "n", "mean_bp", "t"]].to_dict("records"),
    "Q2_regime_view_best_cell": regime_view[["regime", "n", "mean_bp", "t"]].to_dict("records"),
    "Q2_sign_concentration": {"up_cells_t2": len(up_cells), "down_cells_t2": len(dn_cells)},
    "Q3_tradeable_tests": trade,
    "grid_rows": len(grid),
}
json.dump(res, open(f"{OUT}/round5_metals_cyclicals.json", "w"), indent=1, default=float)
print("\nsaved round5_metals_cyclicals.json")
