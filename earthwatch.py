"""
EarthWatch - 5-River Rain & Flood Trend Detective
NASA Space Apps Challenge: "Be an Earth System Trend Detective"

Usage:
    python earthwatch.py            # opens the website/GUI in your browser
    python earthwatch.py --cli      # no GUI: downloads REAL NASA POWER data, saves output/ (CSV, PNG, MD, HTML)
    python earthwatch.py --demo     # no GUI: SYNTHETIC data, only to test the code (never submit these results)

Data: NASA POWER monthly API (no login/key). Variables (MERRA-2 / GPM-IMERG derived):
    PRECTOTCORR  precipitation (mm/day)      GWETTOP   surface soil wetness (0-1)
    GWETROOT     root-zone soil wetness      T2M       air temperature (C)
    QV2M         specific humidity (g/kg)

Pipeline per river
    1. 3x3 grid of NASA POWER points, averaged, cached on disk (re-runs are instant)
    2. STL decomposition -> trend / season / anomaly
    3. Trend test on ANNUAL means (Mann-Kendall + Theil-Sen with 95% CI). Annual means avoid the
       false confidence you get from testing a smoothed, autocorrelated monthly curve.
    4. Flood-setup months = extreme rain anomaly on already-wet soil (a proxy, not observed floods)
    5. Logistic model: chance of an extreme-rain month NEXT month, scored with expanding-window
       time-series cross-validation (never trained on the future) and compared with a base-rate guess
"""
import argparse, calendar, html, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# name: (W, S, E, N)
RIVERS = {
    "Indus":    (66, 24, 78, 37),
    "Zambezi":  (18, -21, 37, -8),
    "Yellow":   (96, 32, 119, 42),
    "Amazon":   (-75, -15, -50, 5),
    "Colorado": (-116, 31, -105, 43),
}
PARAMS = ["PRECTOTCORR", "GWETTOP", "GWETROOT", "T2M", "QV2M"]
START = 2001
END = datetime.now().year               # POWER simply returns -999 for months that are not out yet
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".earthwatch_cache")
OUT = "output"
Z_DEFAULT = 1.5
MONTH = calendar.month_name             # MONTH[1] == "January"


# ───────────────────────────── data ─────────────────────────────
def grid_points(box):
    """3x3 sample grid inside the bounding box (centre + spread) to represent the area."""
    w, s, e, n = box
    lons = np.linspace(w + (e - w) * 0.25, e - (e - w) * 0.25, 3)
    lats = np.linspace(s + (n - s) * 0.25, n - (n - s) * 0.25, 3)
    return [(round(la, 2), round(lo, 2)) for la in lats for lo in lons]


def fetch_power(lat, lon, retries=4):
    """One NASA POWER point, cached on disk (the cache key includes the month so it refreshes monthly)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"p_{lat}_{lon}_{START}_{END}_{datetime.now():%Y%m}.json")
    if os.path.exists(path):
        with open(path) as f:
            par = json.load(f)
    else:
        url = ("https://power.larc.nasa.gov/api/temporal/monthly/point?"
               f"parameters={','.join(PARAMS)}&community=AG&longitude={lon}&latitude={lat}"
               f"&start={START}&end={END}&format=JSON")
        err = None
        for i in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    par = json.load(r)["properties"]["parameter"]
                break
            except Exception as e:                      # network blip, rate limit, bad JSON
                err = e
                time.sleep(2 ** i)
        else:
            raise RuntimeError(f"NASA POWER did not answer for ({lat}, {lon}): {err}")
        with open(path, "w") as f:
            json.dump(par, f)
    rows = {}
    for p in PARAMS:
        for k, v in par[p].items():
            if k.endswith("13"):                        # annual-mean entry
                continue
            rows.setdefault(k, {})[p] = np.nan if v == -999 else v
    df = pd.DataFrame(rows).T
    df.index = pd.to_datetime(df.index, format="%Y%m")
    return df.sort_index()


def load_real(river, box, on_point=None):
    """Download the 9 sample points (3 at a time) and average them. on_point(done, total, lat, lon) drives the GUI."""
    pts = grid_points(box)
    frames, done = [], 0
    if on_point:
        on_point(0, len(pts), None, None)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_power, la, lo): (la, lo) for la, lo in pts}
        for fut in as_completed(futs):
            la, lo = futs[fut]
            try:
                frames.append(fut.result())
                print(f"  {river}: got point ({la}, {lo})")
            except Exception as e:
                print(f"  {river}: point ({la}, {lo}) failed: {e}")
            done += 1
            if on_point:
                on_point(done, len(pts), la, lo)
    if len(frames) < 3:
        raise RuntimeError("fewer than 3 of 9 sample points could be downloaded")
    if on_point:
        on_point(len(pts), len(pts), None, None)
    return pd.concat(frames).groupby(level=0).mean().sort_index()   # NaN-safe area average


def make_synthetic(river, box, seed):
    """SYNTHETIC data, only for testing the pipeline."""
    rng = np.random.default_rng(seed)
    last = pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=2)
    idx = pd.date_range(f"{START}-01-01", last, freq="MS")
    m = idx.month.values
    peak = {"Indus": 7.5, "Zambezi": 1.5, "Yellow": 7.5, "Amazon": 2.5, "Colorado": 6.5}[river]
    base = 3 + 2.5 * np.cos(2 * np.pi * (m - peak) / 12)
    trend = np.linspace(0, 0.6, len(idx))
    prec = np.clip(base + trend + rng.gamma(2, 0.6, len(idx)) - 1.2, 0, None)
    soil = np.clip(0.3 + 0.04 * prec + rng.normal(0, 0.03, len(idx)), 0, 1)
    return pd.DataFrame({
        "PRECTOTCORR": prec, "GWETTOP": soil,
        "GWETROOT": np.clip(soil + rng.normal(0, 0.02, len(idx)), 0, 1),
        "T2M": 15 + 10 * np.sin(2 * np.pi * (m - 4) / 12) + np.linspace(0, 1, len(idx)) + rng.normal(0, 0.7, len(idx)),
        "QV2M": 6 + 3 * np.sin(2 * np.pi * (m - 4) / 12) + rng.normal(0, 0.3, len(idx)),
    }, index=idx)


# ───────────────────────────── analysis ─────────────────────────────
def _annual_trend(series):
    """Mann-Kendall on annual means + Theil-Sen slope with 95% CI (per decade). Complete years only."""
    g = series.groupby(series.index.year).agg(["mean", "count"])
    ann = g[g["count"] == 12]["mean"]
    if len(ann) < 8:
        return dict(n=len(ann), tau=np.nan, p=np.nan, slope=np.nan, lo=np.nan, hi=np.nan, mean=np.nan)
    tau, p = stats.kendalltau(ann.index.values, ann.values)
    slope, _, lo, hi = stats.theilslopes(ann.values, ann.index.values, 0.95)
    return dict(n=len(ann), tau=tau, p=p, slope=slope * 10, lo=lo * 10, hi=hi * 10, mean=ann.mean())


def _features(d):
    X = pd.DataFrame({
        "rain": d["PRECTOTCORR"], "z": d["z"], "soil_top": d["GWETTOP"], "soil_root": d["GWETROOT"],
        "temp": d["T2M"], "humid": d["QV2M"],
        "msin": np.sin(2 * np.pi * d.index.month / 12), "mcos": np.cos(2 * np.pi * d.index.month / 12)})
    X["rain_lag1"] = X["rain"].shift(1)
    X["soil_change"] = d["GWETTOP"].diff()
    return X


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.3))


def analyse(river, df, z_cut=Z_DEFAULT):
    valid = df.dropna()
    df = df.sort_index().loc[valid.index[0]:valid.index[-1]].interpolate(limit_area="inside")
    d = df.copy()
    stl = STL(d["PRECTOTCORR"], period=12, robust=True).fit()
    d["trend"], d["seasonal"], d["resid"] = stl.trend, stl.seasonal, stl.resid
    d["z"] = d["resid"] / d["resid"].std()
    d["wet_soil"] = d["GWETTOP"] > d["GWETTOP"].quantile(0.75)
    d["extreme_rain"] = d["z"] > z_cut
    d["flood_risk"] = d["extreme_rain"] & d["wet_soil"]          # extreme rain on wet soil

    rt = _annual_trend(d["PRECTOTCORR"])
    tt = _annual_trend(d["T2M"])

    # ---- probability model: P(extreme rain next month), honest out-of-sample scoring ----
    X_all = _features(d)
    y_all = d["extreme_rain"].astype(float).shift(-1)
    ok = y_all.notna() & X_all.notna().all(axis=1)
    X, y = X_all[ok], y_all[ok].astype(int)
    auc = brier = base_brier = prob_next = np.nan
    base_rate = float(y.mean())
    d["p_extreme_next"] = np.nan
    if y.nunique() == 2 and len(X) > 60:
        oof, ref = pd.Series(np.nan, index=X.index), pd.Series(np.nan, index=X.index)
        for tr, te in TimeSeriesSplit(n_splits=5).split(X):
            if y.iloc[tr].nunique() < 2:
                continue
            m = _model().fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
            ref.iloc[te] = y.iloc[tr].mean()
        seen = oof.notna()
        if seen.sum() > 30 and y[seen].nunique() == 2:
            auc = roc_auc_score(y[seen], oof[seen])
            brier = brier_score_loss(y[seen], oof[seen])
            base_brier = brier_score_loss(y[seen], ref[seen])
        d.loc[oof.index, "p_extreme_next"] = oof                 # out-of-sample only: no peeking
        if not X_all.iloc[[-1]].isna().any(axis=None):
            prob_next = float(_model().fit(X, y).predict_proba(X_all.iloc[[-1]])[:, 1][0])
    bss = np.nan if np.isnan(brier) else 1 - brier / base_brier      # Brier skill vs base-rate guess

    # ---- seasonality and flood-setup patterns ----
    clim = d.groupby(d.index.month)["PRECTOTCORR"].mean()
    fl = d[d["flood_risk"]]
    peak = MONTH[int(fl.index.month.value_counts().idxmax())] if len(fl) else None
    recent, earlier = d.iloc[-120:], d.iloc[:-120]
    rate_recent = recent["flood_risk"].sum() / (len(recent) / 12)
    rate_earlier = earlier["flood_risk"].sum() / (len(earlier) / 12) if len(earlier) >= 36 else np.nan
    nm = d.index[-1] + pd.DateOffset(months=1)
    r = lambda v, k: None if pd.isna(v) else round(float(v), k)

    s = {
        "river": river, "months": len(d), "years": rt["n"], "period": f"{d.index[0]:%Y-%m} to {d.index[-1]:%Y-%m}",
        "mean_rain_mm_day": round(d["PRECTOTCORR"].mean(), 2),
        "trend_mm_day_per_decade": r(rt["slope"], 3), "trend_ci_low": r(rt["lo"], 3), "trend_ci_high": r(rt["hi"], 3),
        "trend_pct_per_decade": r(rt["slope"] / rt["mean"] * 100, 1), "trend_tau": r(rt["tau"], 3),
        "trend_p": r(rt["p"], 4), "trend_significant": bool(rt["p"] < 0.05),
        "extreme_rain_months": int(d["extreme_rain"].sum()), "flood_risk_months": int(d["flood_risk"].sum()),
        "flood_peak_month": peak, "flood_per_year_recent": r(rate_recent, 2), "flood_per_year_earlier": r(rate_earlier, 2),
        "wettest_month": MONTH[int(clim.idxmax())], "driest_month": MONTH[int(clim.idxmin())],
        "model_AUC": r(auc, 3), "model_Brier": r(brier, 3), "baseline_Brier": r(base_brier, 3), "brier_skill": r(bss, 3),
        "base_rate": r(base_rate, 3), "P_extreme_rain_next_month": r(prob_next, 3),
        "last_month": f"{d.index[-1]:%Y-%m}", "next_month": f"{nm:%B %Y}", "next_ts": f"{nm:%Y-%m-01}",
        "temp_trend_C_per_decade": r(tt["slope"], 2), "z_cut": z_cut,
    }
    return d, s


# ───────────────────────────── plain-English layer ─────────────────────────────
def skill_label(s):
    auc, bss = s["model_AUC"], s["brier_skill"]
    if auc is None:
        return "No forecast", "b-flat"
    if auc >= 0.7 and (bss or 0) > 0:
        return "Useful signal", "b-good"
    if auc >= 0.6:
        return "Weak signal", "b-weak"
    return "Barely beats guessing", "b-flat"


def risk_level(s):
    p, base = s["P_extreme_rain_next_month"], s["base_rate"]
    if p is None or not base:
        return "Not enough data", "b-flat"
    ratio = p / base
    if ratio >= 2.5 or p >= 0.35:
        return "Elevated", "b-risk"
    if ratio >= 1.3:
        return "Above usual", "b-weak"
    if ratio <= 0.7:
        return "Calmer than usual", "b-good"
    return "About usual", "b-flat"


def story_trend(s):
    n, name, pct = s["years"], s["river"], s["trend_pct_per_decade"]
    mm = s["trend_mm_day_per_decade"] * 365
    ci = f"(95% range: {s['trend_ci_low']:+.2f} to {s['trend_ci_high']:+.2f} mm/day per decade)"
    if not s["trend_significant"]:
        return (f"Over {n} complete years, rainfall in the {name} basin has bounced around without a clear direction. "
                f"The drift I measure ({pct:+.1f}% per decade) is small enough that ordinary year-to-year weather could produce it, "
                f"so I wouldn't call it a trend {ci}.")
    word = "wetter" if s["trend_mm_day_per_decade"] > 0 else "drier"
    return (f"The {name} basin is getting {word}, and it isn't a fluke. Rainfall is shifting by about {pct:+.1f}% per decade, "
            f"roughly {mm:+.0f} mm of rain per year each decade, over {n} complete years {ci}.")


def story_flood(s):
    n, m = s["flood_risk_months"], s["months"]
    if n == 0:
        return ("No month in this record paired unusually heavy rain with already-soggy ground, "
                "so I found no flood setups at the current threshold. Try lowering it in the sidebar.")
    t = (f"Out of {m} months, {n} looked like flood setups: unusually heavy rain landing on ground that was already wet. "
         f"{'Most of them cluster in ' + s['flood_peak_month'] + '. ' if s['flood_peak_month'] else ''}")
    rr, re_ = s["flood_per_year_recent"], s["flood_per_year_earlier"]
    if re_ is not None and n >= 4:
        if rr > re_ * 1.5:
            t += f"They've turned up more often in the last decade ({rr:.2f} a year vs {re_:.2f} before). "
        elif rr < re_ * 0.67:
            t += f"They've turned up less often in the last decade ({rr:.2f} a year vs {re_:.2f} before). "
        else:
            t += "Their frequency hasn't changed much between the last decade and earlier years. "
    return t + "Remember this is a proxy from satellite rain and soil moisture, not a record of actual floods."


def story_forecast(s):
    p = s["P_extreme_rain_next_month"]
    if p is None:
        return "There aren't enough extreme-rain months in this record for me to train a forecast I'd trust."
    lvl, _ = risk_level(s)
    sk, _ = skill_label(s)
    trust = {"Useful signal": "On held-out years it ranks risky months better than chance, so it's worth a look.",
             "Weak signal": "On held-out years it beats chance only slightly, so treat it as a nudge, not a warning.",
             "Barely beats guessing": "On held-out years it barely beats guessing, so I wouldn't act on this number."}.get(sk, "")
    return (f"My estimate for {s['next_month']}: about a {p:.0%} chance of an unusually wet month "
            f"(a typical month sits near {s['base_rate']:.0%}), which is {lvl.lower()}. {trust}")


def headline(results):
    S = [s for _, s in results.values()]
    sig = [s for s in S if s["trend_significant"]]
    wet = sorted([s for s in sig if s["trend_mm_day_per_decade"] > 0], key=lambda s: -s["trend_pct_per_decade"])
    dry = sorted([s for s in sig if s["trend_mm_day_per_decade"] < 0], key=lambda s: s["trend_pct_per_decade"])
    parts = []
    if wet:
        parts.append("Getting wetter: " + ", ".join(f"{s['river']} ({s['trend_pct_per_decade']:+.1f}%/decade)" for s in wet) + ".")
    if dry:
        parts.append("Getting drier: " + ", ".join(f"{s['river']} ({s['trend_pct_per_decade']:+.1f}%/decade)" for s in dry) + ".")
    quiet = [s["river"] for s in S if not s["trend_significant"]]
    if quiet and sig:
        parts.append("No clear trend: " + ", ".join(quiet) + ".")
    if not sig:
        parts.append("None of these rivers shows a rainfall trend strong enough to call at 95% confidence. That's a finding too.")
    scored = [s for s in S if s["P_extreme_rain_next_month"] and s["base_rate"]]
    if scored:
        top = max(scored, key=lambda s: s["P_extreme_rain_next_month"] / s["base_rate"])
        lvl, _ = risk_level(top)
        parts.append(f"For {top['next_month']}, the river to keep an eye on is the {top['river']} "
                     f"({top['P_extreme_rain_next_month']:.0%} chance of extreme rain, {lvl.lower()}).")
    return " ".join(parts)


# ───────────────────────────── figures (plotly) ─────────────────────────────
def _style(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=60, b=40), font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                      legend=dict(orientation="h", y=-0.08, x=0, yanchor="top"),
                      title_font=dict(color="#e2e8f0", size=16), hoverlabel=dict(bgcolor="#0f1b33"))
    fig.update_xaxes(gridcolor="#ffffff10"); fig.update_yaxes(gridcolor="#ffffff10")
    return fig


def fig_overview(S, tag=""):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Rainfall change per decade (% of average, 95% range)",
                                                        "Flood-setup months in the record"))
    lo = [(l / m * 100) if m else 0 for l, m in zip(S["trend_ci_low"], S["mean_rain_mm_day"])]
    hi = [(h / m * 100) if m else 0 for h, m in zip(S["trend_ci_high"], S["mean_rain_mm_day"])]
    col = ["#64748b" if not sg else ("#38bdf8" if v > 0 else "#fb923c")
           for v, sg in zip(S["trend_pct_per_decade"], S["trend_significant"])]
    fig.add_scatter(x=S["trend_pct_per_decade"], y=S["river"], mode="markers", marker=dict(color=col, size=14),
                    error_x=dict(type="data", symmetric=False, array=np.array(hi) - S["trend_pct_per_decade"],
                                 arrayminus=S["trend_pct_per_decade"] - np.array(lo), color="#94a3b8"),
                    row=1, col=1, name="trend")
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1, row=1, col=1)
    fig.add_bar(x=S["river"], y=S["flood_risk_months"], marker_color="#f43f5e", row=1, col=2, name="flood setups")
    fig.update_layout(showlegend=False, title=tag)
    return _style(fig, 380)


def fig_detail(d, s, z_cut, tag=""):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.5, .25, .25], vertical_spacing=.05)
    fig.add_scatter(x=d.index, y=d["PRECTOTCORR"], name="Monthly rainfall", line=dict(color="#7dd3fc", width=1.5), row=1, col=1)
    fig.add_scatter(x=d.index, y=d["trend"], name="Long-term trend (STL)", line=dict(color="#fbbf24", width=3), row=1, col=1)
    f = d[d["flood_risk"]]
    fig.add_scatter(x=f.index, y=f["PRECTOTCORR"], name="Flood-setup month", mode="markers",
                    marker=dict(color="#f43f5e", size=10, line=dict(color="#fff", width=1)), row=1, col=1)
    fig.add_bar(x=d.index, y=d["z"], name="Rain anomaly (z)", marker_color=["#f43f5e" if v else "#475569" for v in d["flood_risk"]], row=2, col=1)
    fig.add_hline(y=z_cut, line_dash="dash", line_color="#cbd5e1", row=2, col=1)
    fig.add_scatter(x=d.index, y=d["p_extreme_next"], name="P(extreme next month), out-of-sample", fill="tozeroy",
                    line=dict(color="#a78bfa"), row=3, col=1)
    if s["P_extreme_rain_next_month"] is not None:
        fig.add_scatter(x=[s["next_ts"]], y=[s["P_extreme_rain_next_month"]], name=f"Forecast: {s['next_month']}", mode="markers",
                        marker=dict(color="#fbbf24", size=13, symbol="star", line=dict(color="#fff", width=1)), row=3, col=1)
    fig.update_yaxes(title_text="mm/day", row=1, col=1); fig.update_yaxes(title_text="z", row=2, col=1)
    fig.update_yaxes(title_text="prob", range=[0, 1], row=3, col=1)
    fig.update_layout(title=f"{s['river']} {tag}")
    return _style(fig, 660)


def fig_heatmap(d, s):
    import plotly.graph_objects as go
    pv = d.assign(y=d.index.year, m=d.index.month).pivot(index="y", columns="m", values="z")
    pv = pv.reindex(columns=range(1, 13))
    fig = go.Figure(go.Heatmap(z=pv.values, x=[MONTH[m][:3] for m in pv.columns], y=pv.index, zmid=0, zmin=-3, zmax=3,
                               colorscale=[[0, "#c2410c"], [.5, "#0f1b33"], [1, "#0ea5e9"]],
                               colorbar=dict(title="z"), hovertemplate="%{y} %{x}: z=%{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{s['river']}: every month's rain anomaly (blue = wetter than normal, orange = drier)")
    return _style(fig, 560)


def fig_calendar(d, s):
    from plotly.subplots import make_subplots
    clim = d.groupby(d.index.month)["PRECTOTCORR"].mean()
    cnt = d[d["flood_risk"]].groupby(d[d["flood_risk"]].index.month).size().reindex(range(1, 13), fill_value=0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=[MONTH[m][:3] for m in range(1, 13)], y=cnt.values, name="Flood-setup months", marker_color="#f43f5e", opacity=.85)
    fig.add_scatter(x=[MONTH[m][:3] for m in range(1, 13)], y=clim.values, name="Typical rain (mm/day)",
                    line=dict(color="#7dd3fc", width=3), secondary_y=True)
    fig.update_yaxes(title_text="flood-setup months", secondary_y=False)
    fig.update_yaxes(title_text="mm/day", secondary_y=True, showgrid=False)
    fig.update_layout(title=f"{s['river']}: when the rain comes, and when flood setups happen")
    return _style(fig, 420)


# ───────────────────────────── outputs: CLI / report ─────────────────────────────
def plot_river(river, d, tag, z_cut):
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(d.index, d["PRECTOTCORR"], color="#9bb7d4", label="Monthly rainfall")
    ax[0].plot(d.index, d["trend"], color="#0b3d91", lw=2.5, label="Long-term trend (STL)")
    fr = d[d["flood_risk"]]
    ax[0].scatter(fr.index, fr["PRECTOTCORR"], color="red", zorder=3, label="Flood-setup month")
    ax[0].set_ylabel("Rain (mm/day)"); ax[0].legend(loc="upper left")
    ax[0].set_title(f"{river} - rainfall, trend and flood-setup months {tag}")
    ax[1].bar(d.index, d["z"], width=25, color=np.where(d["z"] > z_cut, "red", "#7a8a99"))
    ax[1].axhline(z_cut, ls="--", color="k", lw=0.8); ax[1].set_ylabel("Rain anomaly (z)")
    ax[2].plot(d.index, d["p_extreme_next"], color="#d9822b")
    ax[2].set_ylabel("P(extreme rain next month)"); ax[2].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(f"{OUT}/{river}_analysis.png", dpi=150); plt.close(fig)


REPORT_CSS = """body{margin:0;background:#070d1a;color:#cbd5e1;font-family:Inter,system-ui,sans-serif;line-height:1.6}
main{max-width:1050px;margin:auto;padding:36px 22px}h1{color:#fff;font-size:2.4rem;margin:0}h2{color:#f8fafc;margin-top:48px}
.lead{background:linear-gradient(120deg,#0b2447,#0e7490);border-radius:18px;padding:26px 30px;margin:20px 0;color:#e0f2fe;font-size:1.05rem}
section{background:#0f1b33;border:1px solid #ffffff14;border-radius:16px;padding:8px 26px 18px;margin:22px 0}
.foot{color:#64748b;font-size:.8rem;margin-top:40px}"""


def build_report_html(results, tag, z_cut):
    S = pd.DataFrame([s for _, s in results.values()])
    parts = [f"<main><h1>🌧️ EarthWatch report</h1><div class='foot'>{html.escape(tag)} · generated {datetime.now():%d %b %Y}</div>",
             f"<div class='lead'>{html.escape(headline(results))}</div>",
             fig_overview(S, tag).to_html(full_html=False, include_plotlyjs="cdn")]
    for r, (d, s) in results.items():
        parts.append(f"<section><h2>{r}</h2><p>{html.escape(story_trend(s))}</p><p>{html.escape(story_flood(s))}</p>"
                     f"<p>{html.escape(story_forecast(s))}</p>"
                     f"{fig_detail(d, s, z_cut, tag).to_html(full_html=False, include_plotlyjs=False)}"
                     f"{fig_calendar(d, s).to_html(full_html=False, include_plotlyjs=False)}</section>")
    parts.append("<div class='foot'>Data: NASA POWER (GPM IMERG rainfall, MERRA-2 soil moisture and climate). Flood setups are a proxy "
                 "(extreme rain on wet soil), not observed floods. The forecast is experimental: check its skill before trusting it.</div></main>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>EarthWatch report</title><style>{REPORT_CSS}</style></head><body>{''.join(parts)}</body></html>")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, for testing only")
    ap.add_argument("--cli", action="store_true", help="run without the GUI and save the output/ folder")
    ap.add_argument("--z", type=float, default=Z_DEFAULT, help="extreme-rain threshold (z-score), default 1.5")
    a = ap.parse_args()
    tag = "[SYNTHETIC DEMO - NOT REAL DATA]" if a.demo else "(NASA POWER data)"
    OUT = "output_demo" if a.demo else "output"
    os.makedirs(OUT, exist_ok=True)

    results = {}
    for i, (river, box) in enumerate(RIVERS.items()):
        print(f"== {river}")
        raw = make_synthetic(river, box, i) if a.demo else load_real(river, box)
        d, s = analyse(river, raw, a.z)
        d.to_csv(f"{OUT}/{river}_monthly_results.csv")
        plot_river(river, d, tag, a.z)
        results[river] = (d, s)

    S = pd.DataFrame([s for _, s in results.values()])
    S.to_csv(f"{OUT}/summary_all_rivers.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(S["river"], S["trend_mm_day_per_decade"], color=np.where(S["trend_significant"], "#0b3d91", "#b0b8c0"))
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("Rainfall trend (mm/day per decade)\ndark = statistically significant (p<0.05)")
    ax[1].bar(S["river"], S["flood_risk_months"], color="#c0392b")
    ax[1].set_title("Flood-setup months (extreme rain on wet soil)")
    fig.suptitle(tag); plt.tight_layout()
    fig.savefig(f"{OUT}/comparison.png", dpi=150); plt.close(fig)

    L = [f"# EarthWatch results {tag}", "", headline(results), "",
         "| River | Trend (%/decade) | Significant | Flood-setup months | Forecast AUC | P(extreme rain next month) |",
         "|---|---|---|---|---|---|"]
    for s in S.to_dict("records"):
        L.append(f"| {s['river']} | {s['trend_pct_per_decade']} | {'yes' if s['trend_significant'] else 'no'} | "
                 f"{s['flood_risk_months']} | {s['model_AUC']} | {s['P_extreme_rain_next_month']} |")
    L += ["", f"Forecast probabilities are for {S.iloc[0]['next_month']}.",
          "Trend test: Mann-Kendall + Theil-Sen on annual means. Model: logistic regression, expanding-window time-series CV. "
          "AUC 0.5 = no skill; Brier skill > 0 = better than guessing the base rate."]
    open(f"{OUT}/REPORT.md", "w", encoding="utf-8").write("\n".join(L))
    open(f"{OUT}/REPORT.html", "w", encoding="utf-8").write(build_report_html(results, tag, a.z))
    print(S[["river", "trend_pct_per_decade", "trend_significant", "flood_risk_months", "model_AUC",
             "P_extreme_rain_next_month"]].to_string(index=False))
    print(f"\nSaved to ./{OUT}/  (open REPORT.html in a browser for the shareable version)")


# ───────────────────────────── website (Streamlit) ─────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1100px 600px at 8% -10%, #12315a88, transparent), radial-gradient(900px 500px at 100% 0%, #0e749044, transparent), #070d1a; color: #e2e8f0; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #0a1224; border-right: 1px solid #ffffff12; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes fall { to { transform: translateY(320px); opacity: 0; } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.25); opacity: .6; } }
@keyframes pop { from { transform: scale(.3); } to { transform: scale(1); } }
@keyframes shimmer { from { background-position: 0 0; } to { background-position: 200% 0; } }
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px; margin-bottom: 18px;
  background: linear-gradient(120deg, #0b2447, #0e7490 65%, #155e75); border: 1px solid #ffffff22; animation: fadeUp .8s ease both; }
.hero h1 { margin: 0; font-size: 2.5rem; font-weight: 800; color: #fff !important; letter-spacing: -.02em; }
.hero p { margin: 8px 0 0; color: #cffafe !important; font-size: 1.04rem; max-width: 760px; line-height: 1.55; }
.drop { position: absolute; top: -24px; width: 2px; height: 20px; background: linear-gradient(#fff0, #7dd3fc); animation: fall linear infinite; opacity: .7; }
.chips { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip { font-size: .75rem; padding: 4px 11px; border-radius: 99px; background: #ffffff1c; border: 1px solid #ffffff2a; color: #e0f2fe !important; }
.card { background: #0f1b33cc; border: 1px solid #ffffff14; border-radius: 16px; padding: 16px 18px; animation: fadeUp .6s ease both; transition: .2s; height: 100%; }
.card:hover { transform: translateY(-3px); border-color: #38bdf866; }
.k { font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: #94a3b8 !important; }
.v { font-size: 1.85rem; font-weight: 800; margin-top: 2px; color: #f8fafc !important; }
.sub { font-size: .8rem; margin-top: 2px; color: #94a3b8 !important; }
.badge { display: inline-block; font-size: .72rem; font-weight: 600; padding: 2px 10px; border-radius: 99px; margin-top: 6px; }
.b-wet { background: #0369a133; color: #7dd3fc !important; } .b-dry { background: #c2410c33; color: #fdba74 !important; }
.b-flat { background: #47556955; color: #cbd5e1 !important; } .b-good { background: #15803d33; color: #86efac !important; }
.b-weak { background: #a1620733; color: #fde68a !important; } .b-risk { background: #be123c44; color: #fda4af !important; }
.story { background: linear-gradient(135deg, #0f1b33, #0c2a3d); border: 1px solid #22d3ee33; border-left: 4px solid #22d3ee;
  border-radius: 14px; padding: 16px 22px; margin: 10px 0 16px; animation: fadeUp .6s ease both; }
.story .who { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: #22d3ee !important; font-weight: 600; }
.story p { margin: 6px 0 0; font-size: 1rem; line-height: 1.65; color: #e2e8f0 !important; }
.gauge { position: relative; height: 8px; background: #1e293b; border-radius: 99px; margin-top: 10px; }
.gfill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22d3ee, #a78bfa); }
.gmark { position: absolute; top: -4px; width: 2px; height: 16px; background: #fbbf24; }
.collect { display: flex; gap: 30px; align-items: center; padding: 22px 26px; border-radius: 18px; background: #0f1b33; border: 1px solid #22d3ee44; animation: fadeUp .4s ease both; }
.radar { flex: none; width: 110px; height: 110px; border-radius: 50%; border: 2px solid #38bdf855; background: radial-gradient(circle, #38bdf822, transparent 70%); position: relative; }
.radar:before { content: ""; position: absolute; inset: 0; border-radius: 50%; background: conic-gradient(from 0deg, #38bdf8cc, #38bdf800 40%); animation: spin 1.2s linear infinite; }
.grid3 { display: grid; grid-template-columns: repeat(3, 30px); gap: 9px; flex: none; }
.dot { width: 30px; height: 30px; border-radius: 50%; background: #1e293b; }
.dot.done { background: #22d3ee; box-shadow: 0 0 14px #22d3ee; animation: pop .4s; }
.dot.now { background: #fbbf24; animation: pulse .8s infinite; }
.bar { height: 8px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-top: 12px; }
.bar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22d3ee, #818cf8, #22d3ee); background-size: 200% 100%; animation: shimmer 1.6s linear infinite; transition: width .4s; }
.ctitle { font-size: 1.15rem; font-weight: 800; color: #f8fafc !important; }
.cmsg { font-size: .85rem; margin-top: 4px; color: #94a3b8 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #0f1b33; border-radius: 10px 10px 0 0; padding: 8px 18px; color: #94a3b8; }
.stTabs [aria-selected="true"] { background: #0e7490; color: #fff !important; }
div.stButton > button { background: linear-gradient(90deg, #0891b2, #6366f1); color: #fff; border: 0; border-radius: 12px; font-weight: 700; padding: .6rem 1rem; transition: .2s; }
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 22px #22d3ee44; color: #fff; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color: #cbd5e1; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #f8fafc; }
div[data-baseweb="select"] div, div[data-baseweb="select"] input { background-color: #0f1b33 !important; border-color: #ffffff22 !important; }
div[data-baseweb="tag"], div[data-baseweb="tag"] div { background-color: #0e7490 !important; }
div[data-baseweb="select"] * { color: #e2e8f0 !important; }
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul { background-color: #0f1b33 !important; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * { background-color: #0f1b33 !important; color: #e2e8f0 !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li:hover *,
div[data-baseweb="popover"] li[aria-selected="true"], div[data-baseweb="popover"] li[aria-selected="true"] * { background-color: #0e7490 !important; color: #fff !important; }
:root, .stApp { color-scheme: dark; }
[data-testid="stMultiSelect"] div:not([data-testid="stWidgetLabel"]):not([data-testid="stWidgetLabel"] *),
[data-testid="stSelectbox"] div:not([data-testid="stWidgetLabel"]):not([data-testid="stWidgetLabel"] *) { background-color: #0f1b33 !important; border-color: #ffffff22 !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"], div[role="listbox"] { background-color: #0f1b33 !important; border-radius: 12px; }
[role="option"], [role="option"] * { background-color: #0f1b33 !important; color: #e2e8f0 !important; }
[role="option"]:hover, [role="option"]:hover *, [role="option"][aria-selected="true"], [role="option"][aria-selected="true"] * { background-color: #0e7490 !important; color: #fff !important; }
[data-testid="stMultiSelect"] span:not([data-testid="stWidgetLabel"] *) { background-color: #0e7490 !important; color: #fff !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { background: transparent !important; }
[data-testid="stMultiSelect"] svg, [data-testid="stSelectbox"] svg { fill: #cbd5e1 !important; }
/* every widget surface is dark with light text, whatever Streamlit theme is active */
input, textarea, [data-baseweb="input"], [data-baseweb="base-input"] { background-color: #0f1b33 !important; color: #f1f5f9 !important; }
[data-testid="stExpander"] details, [data-testid="stExpander"] details > summary { background: #0f1b33 !important; border-color: #ffffff18 !important; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #f1f5f9 !important; background: transparent !important; }
[data-testid="stExpander"] details > div, [data-testid="stExpander"] details > div * { color: #cbd5e1 !important; background: transparent !important; }
[data-testid="stToast"], [data-testid="stToast"] * { background-color: #14274a !important; color: #f1f5f9 !important; }
[data-testid="stTooltipContent"], div[data-baseweb="tooltip"] *, div[data-baseweb="tooltip"] { background-color: #14274a !important; color: #f1f5f9 !important; }
div.stButton > button p, div.stButton > button span { color: #fff !important; }
.stDownloadButton > button p, .stDownloadButton > button span { color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div { color: #e2e8f0; }
.tbl-wrap { overflow: auto; max-height: 520px; border: 1px solid #ffffff18; border-radius: 12px; background: #0f1b33; }
.tbl { width: 100%; border-collapse: collapse; font-size: .85rem; }
.tbl th { position: sticky; top: 0; background: #16294d; color: #f8fafc !important; text-align: left; padding: 9px 12px; font-weight: 600; border-bottom: 1px solid #ffffff22; white-space: nowrap; }
.tbl td { padding: 7px 12px; color: #e2e8f0 !important; border-bottom: 1px solid #ffffff0d; white-space: nowrap; }
.tbl tr:hover td { background: #ffffff0a; }
div[data-testid="stAlert"], div[data-testid="stAlertContainer"], div[data-testid="stAlert"] > div { background: #14233d !important; border: 1px solid #ffffff18; border-radius: 12px; }
div[data-testid="stAlert"] * { color: #e2e8f0 !important; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stDownloadButton > button { background: #0f1b33; border: 1px solid #22d3ee55; color: #e2e8f0; border-radius: 12px; }
.stDownloadButton > button:hover { border-color: #22d3ee; color: #fff; }
details { background: #0f1b33; border-radius: 12px; border: 1px solid #ffffff14 !important; }
.foot { color: #64748b !important; font-size: .78rem; margin-top: 26px; }
</style>
"""

WAIT_LINES = ["Asking the satellites nicely...", "Counting raindrops from orbit...", "Checking how soggy the soil is...",
              "Measuring humidity from 800 km up...", "Almost there, the clouds are cooperating..."]


def _greeting():
    h = datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 18 else "Good evening")


def _hero(demo):
    import random
    rnd = random.Random(3)
    drops = "".join(f'<div class="drop" style="left:{rnd.randint(2, 98)}%;animation-duration:{rnd.uniform(.9, 2.2):.2f}s;'
                    f'animation-delay:-{rnd.uniform(0, 2):.2f}s"></div>' for _ in range(30))
    chips = "".join(f'<span class="chip">{c}</span>' for c in
                    ["NASA POWER", "GPM IMERG rainfall", "MERRA-2 soil wetness", "STL + Mann-Kendall + Theil-Sen",
                     "5 river basins", f"{START}-{END}"])
    tag = '<span class="chip" style="background:#b4530955">SYNTHETIC DEMO DATA</span>' if demo else ""
    return (f'<div class="hero">{drops}<h1>🌧️ EarthWatch</h1>'
            f'<p>{_greeting()}. I read {END - START + 1} years of NASA satellite data for five of the world\'s great rivers, '
            f'then tell you in plain words which basins are getting wetter or drier, when flood setups tend to appear, '
            f'and what the next month might hold. I\'ll also be upfront about how far each answer can be trusted.</p>'
            f'<div class="chips">{chips}{tag}</div></div>')


def _collect_html(river, done, total, ri, nr, msg):
    dots = "".join(f'<div class="dot {"done" if i < done else ("now" if i == done else "")}"></div>' for i in range(total))
    pct = int(100 * (ri + done / total) / nr)
    return (f'<div class="collect"><div class="radar"></div><div class="grid3">{dots}</div>'
            f'<div style="flex:1"><div class="ctitle">Collecting satellite data: {river}</div>'
            f'<div class="cmsg">{msg}</div><div class="cmsg">River {ri + 1} of {nr} &middot; {done} of {total} points in</div>'
            f'<div class="bar"><div style="width:{pct}%"></div></div></div></div>')


def _card(k, v, sub="", badge="", cls="b-flat", delay=0, extra=""):
    b = f'<span class="badge {cls}">{badge}</span>' if badge else ""
    return (f'<div class="card" style="animation-delay:{delay}s"><div class="k">{k}</div>'
            f'<div class="v">{v}</div><div class="sub">{sub}</div>{extra}{b}</div>')


def _story(text, who="Detective's notes"):
    return f'<div class="story"><div class="who">{who}</div><p>{html.escape(text)}</p></div>'


def _gauge(p, base):
    axis = max(0.5, p * 1.3)
    return (f'<div class="gauge"><div class="gfill" style="width:{min(p / axis * 100, 100):.0f}%"></div>'
            f'<div class="gmark" style="left:{min(base / axis * 100, 100):.0f}%"></div></div>'
            f'<div class="sub">yellow tick = usual odds ({base:.0%})</div>')


def gui():
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="EarthWatch", page_icon="🌧️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    def plot(fig):
        try:
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.plotly_chart(fig, use_container_width=True)

    def table(df):
        """Own HTML table: st.dataframe is canvas-drawn and takes its colours from the Streamlit theme,
        which is what produced white-on-grey text. This one is always dark with light text."""
        raw = df.to_html(classes="tbl", index=False, border=0, na_rep="-", escape=True)
        st.markdown('<div class="tbl-wrap">' + "".join(l.strip() for l in raw.splitlines()) + "</div>",
                    unsafe_allow_html=True)

    def btn(fn, *a, **k):
        try:
            return fn(*a, width="stretch", **k)
        except Exception:
            return fn(*a, use_container_width=True, **k)

    @st.cache_data(show_spinner=False)
    def run_analysis(river, raw, z_cut):
        return analyse(river, raw, z_cut)

    # ---------- sidebar ----------
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        demo = st.toggle("Demo mode (SYNTHETIC data)", value=False)
        pick = st.multiselect("Rivers", list(RIVERS), default=list(RIVERS))
        z_cut = st.slider("What counts as 'extreme' rain? (z)", 1.0, 3.0, Z_DEFAULT, 0.1,
                          help="How many standard deviations above normal a month's rain must be. Lower = more months flagged.")
        run = btn(st.button, "🚀 Run analysis")
        if btn(st.button, "🗑️ Clear cached results"):
            for k in ("raw", "ran"):
                st.session_state.pop(k, None)
            st.rerun()
        st.caption("Real data comes from the NASA POWER API (no login). The first real run takes about a minute; "
                   "after that it's cached on your computer and loads instantly.")

    TAG = "[SYNTHETIC DEMO - NOT REAL DATA]" if demo else "(NASA POWER data)"
    st.markdown(_hero(demo), unsafe_allow_html=True)
    if demo:
        st.warning("Demo mode: every number is SYNTHETIC. Do not submit them as results.")

    if not (run or st.session_state.get("ran")):
        c = st.columns(3)
        for i, (t, s) in enumerate([("1 · Collect", "9 satellite sample points per river are downloaded, cached and averaged."),
                                    ("2 · Detect", "STL splits rainfall into trend, season and surprises; a Mann-Kendall test on yearly averages says whether the trend is real."),
                                    ("3 · Warn", "Extreme rain on wet soil = flood setup. A model estimates next month's odds and is graded on years it never saw.")]):
            c[i].markdown(_card(t, "", s, delay=i * .15), unsafe_allow_html=True)
        st.info("Pick rivers in the sidebar and press **Run analysis**. I'll take it from there.")
        return
    st.session_state["ran"] = True
    if not pick:
        st.info("Pick at least one river in the sidebar."); return

    # ---------- data collection with animation ----------
    raw_store = st.session_state.setdefault("raw", {})
    stage, fetched = st.empty(), False
    for ri, river in enumerate(pick):
        key = (river, demo)
        if key in raw_store:
            continue
        box, pts = RIVERS[river], grid_points(RIVERS[river])

        def on_point(done, total, lat, lon, _r=river, _ri=ri):
            msg = WAIT_LINES[(done + _ri) % len(WAIT_LINES)] if done < total else "All points received, crunching the numbers..."
            stage.markdown(_collect_html(_r, done, total, _ri, len(pick), msg), unsafe_allow_html=True)

        try:
            if demo:
                for i in range(len(pts) + 1):
                    on_point(i, len(pts), None, None); time.sleep(0.1)
                raw_store[key] = make_synthetic(river, box, list(RIVERS).index(river))
            else:
                raw_store[key] = load_real(river, box, on_point=on_point)
            fetched = True
        except Exception as e:
            stage.empty()
            st.error(f"{river}: I couldn't get the data ({e}). Check your internet connection and press Run again.")
    stage.empty()
    if fetched:
        st.toast("Data is in. Here's what I found.", icon="🌧️")

    results = {}
    for r in pick:
        if (r, demo) in raw_store:
            try:
                results[r] = run_analysis(r, raw_store[(r, demo)], z_cut)
            except Exception as e:
                st.error(f"{r}: the analysis failed ({e}).")
    if not results:
        return
    S = pd.DataFrame([s for _, s in results.values()])
    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🔎 River detail", "🗓️ Seasons & risk", "🗺️ Map", "📁 Data & method"])

    # ---------- overview ----------
    with t1:
        st.markdown(_story(headline(results), "The short version"), unsafe_allow_html=True)
        cols = st.columns(len(results))
        for i, (r, (d, s)) in enumerate(results.items()):
            if not s["trend_significant"]:
                badge, cls = "No clear trend", "b-flat"
            elif s["trend_mm_day_per_decade"] > 0:
                badge, cls = "Getting wetter", "b-wet"
            else:
                badge, cls = "Getting drier", "b-dry"
            cols[i].markdown(_card(r, f"{s['trend_pct_per_decade']:+.1f}%", "rain change per decade", badge, cls, i * .1),
                             unsafe_allow_html=True)
        st.write("")
        plot(fig_overview(S, TAG))
        st.caption("Left: dots are the best estimate, whiskers are the 95% range. Blue = significantly wetter, orange = drier, "
                   "grey = can't tell. If the whisker crosses zero, the trend could be noise.")
        show = pd.DataFrame({
            "River": S["river"], "Trend (%/decade)": S["trend_pct_per_decade"],
            "Clear trend?": np.where(S["trend_significant"], "yes", "no"),
            "Flood-setup months": S["flood_risk_months"], "Peak flood-setup month": S["flood_peak_month"].fillna("-"),
            f"Chance of extreme rain ({S.iloc[0]['next_month']})": S["P_extreme_rain_next_month"],
            "Forecast skill": [skill_label(s)[0] for _, s in results.values()]})
        table(show)

    # ---------- detail ----------
    with t2:
        river = st.selectbox("Which river?", list(results))
        d, s = results[river]
        st.markdown(_story(story_trend(s), f"{river} · rainfall trend") + _story(story_flood(s), f"{river} · flood setups")
                    + _story(story_forecast(s), f"{river} · next month"), unsafe_allow_html=True)
        p, auc = s["P_extreme_rain_next_month"], s["model_AUC"]
        lvl, lcls = risk_level(s)
        sk, scls = skill_label(s)
        c = st.columns(4)
        c[0].markdown(_card("Rain trend", f"{s['trend_mm_day_per_decade']:+.2f}", "mm/day per decade",
                            "significant" if s["trend_significant"] else "not significant",
                            "b-good" if s["trend_significant"] else "b-flat"), unsafe_allow_html=True)
        c[1].markdown(_card("Flood-setup months", s["flood_risk_months"], f"of {s['months']} months", delay=.1), unsafe_allow_html=True)
        c[2].markdown(_card(f"Extreme rain · {s['next_month']}", "n/a" if p is None else f"{p:.0%}", "chance of an unusually wet month",
                            lvl, lcls, .2, "" if p is None else _gauge(p, s["base_rate"])), unsafe_allow_html=True)
        c[3].markdown(_card("Forecast skill (AUC)", "n/a" if auc is None else f"{auc:.2f}", "0.5 = coin flip, 1.0 = perfect",
                            sk, scls, .3), unsafe_allow_html=True)
        st.write("")
        plot(fig_detail(d, s, z_cut, TAG))

        st.markdown("#### ▶ Replay the timeline")
        x, yv, tv = list(d.index), list(d["PRECTOTCORR"]), list(d["trend"])
        k0 = 12
        frames = [go.Frame(data=[go.Scatter(x=x[:k], y=yv[:k]), go.Scatter(x=x[:k], y=tv[:k])], name=str(k))
                  for k in range(k0, len(x) + 1, 4)]
        anim = go.Figure(data=[go.Scatter(x=x[:k0], y=yv[:k0], name="Rainfall", line=dict(color="#7dd3fc")),
                               go.Scatter(x=x[:k0], y=tv[:k0], name="Trend", line=dict(color="#fbbf24", width=3))], frames=frames)
        anim.update_xaxes(range=[x[0], x[-1]]); anim.update_yaxes(range=[0, max(yv) * 1.1], title_text="mm/day")
        anim.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.25, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])])
        plot(_style(anim, 340))

    # ---------- seasons ----------
    with t3:
        river = st.selectbox("Which river?", list(results), key="season_river")
        d, s = results[river]
        st.markdown(_story(f"Rain in the {river} basin peaks in {s['wettest_month']} and bottoms out in {s['driest_month']}. "
                           + (f"Flood setups mostly show up in {s['flood_peak_month']}." if s["flood_peak_month"]
                              else "I didn't find any flood setups at the current threshold."), "Seasonal rhythm"),
                    unsafe_allow_html=True)
        plot(fig_calendar(d, s))
        plot(fig_heatmap(d, s))
        st.caption("Each cell is one month. Read across a row to see a year, down a column to see how one calendar month has behaved over time.")

    # ---------- map ----------
    with t4:
        fig = go.Figure()
        palette = ["#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
        for i, (r, (d, s)) in enumerate(results.items()):
            w, so, e, n = RIVERS[r]
            fig.add_scattergeo(lon=[w, e, e, w, w], lat=[so, so, n, n, so], mode="lines", name=r,
                               line=dict(color=palette[i % 5], width=2), hoverinfo="name")
            g = grid_points(RIVERS[r])
            fig.add_scattergeo(lon=[b for a, b in g], lat=[a for a, b in g], mode="markers", showlegend=False,
                               marker=dict(color=palette[i % 5], size=8), hovertext=f"{r} sample point")
        fig.update_geos(projection_type="natural earth", showland=True, landcolor="#1e293b", showocean=True,
                        oceancolor="#0b1220", showcountries=True, countrycolor="#334155", coastlinecolor="#475569",
                        bgcolor="rgba(0,0,0,0)")
        plot(_style(fig, 520))
        st.caption("Rectangles = river bounding boxes; dots = the 9 NASA POWER sample points averaged per river.")

    # ---------- data & method ----------
    with t5:
        river = st.selectbox("River data", list(results), key="dl")
        d, _ = results[river]
        dd = d.round(3).copy()
        dd.index = dd.index.strftime("%Y-%m"); dd.index.name = "month"
        table(dd.reset_index())
        a, b, c = st.columns(3)
        a.download_button("⬇️ This river (CSV)", d.to_csv().encode(), f"{river}_monthly_results.csv")
        b.download_button("⬇️ Summary (CSV)", S.to_csv(index=False).encode(), "summary_all_rivers.csv")
        if c.button("📄 Prepare shareable report"):
            st.session_state["report"] = build_report_html(results, TAG, z_cut)
        if st.session_state.get("report"):
            c.download_button("⬇️ Download report (HTML)", st.session_state["report"].encode("utf-8"), "EarthWatch_report.html",
                              mime="text/html")
        with st.expander("How the detective works (plain English)"):
            st.markdown(
                "- **Data:** NASA POWER monthly values for rainfall (GPM IMERG), soil wetness and climate (MERRA-2) at 9 points per river, averaged.\n"
                "- **Trend:** I split rainfall into trend, season and surprise (STL), but test the trend on *yearly averages* with Mann-Kendall and Theil-Sen. "
                "Testing the smoothed monthly curve would make trends look more certain than they are.\n"
                "- **Flood setup:** a month whose rain is unusually high *and* whose topsoil is in the wettest quarter of its record.\n"
                "- **Forecast:** a logistic model predicts whether *next* month will be extreme. It is scored with expanding-window "
                "cross-validation, so it is only ever tested on months that come after the data it trained on.")
        with st.expander("What this can't tell you"):
            st.markdown(
                "- Flood setups are a proxy, not recorded floods. Rivers also flood from snowmelt, dams and upstream storms outside the box.\n"
                f"- {END - START + 1} years is short for climate trends; natural cycles like El Niño can mimic a trend.\n"
                "- With only a couple of dozen extreme months per river, forecast scores are noisy. Check the skill badge before trusting a probability.")
    st.markdown('<div class="foot">Built for the NASA Space Apps Challenge · data: NASA POWER · '
                'flood setups are a proxy from rain anomalies and soil wetness, not observed floods.</div>', unsafe_allow_html=True)


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():          # streamlit run earthwatch.py   -> GUI
    gui()
elif __name__ == "__main__":
    if len(sys.argv) == 1:    # python earthwatch.py  -> open the GUI automatically
        import subprocess
        try:
            import streamlit  # noqa: F401
        except ImportError:
            sys.exit("Streamlit is not installed. Run: pip install streamlit plotly   (or use --cli)")
        print("Opening the GUI in your browser... (press Ctrl+C here to stop)")
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
                        "--theme.base", "dark", "--theme.primaryColor", "#22d3ee",
                        "--theme.backgroundColor", "#070d1a", "--theme.secondaryBackgroundColor", "#0f1b33",
                        "--theme.textColor", "#e2e8f0"])
    else:                     # --cli / --demo -> command line
        main()
