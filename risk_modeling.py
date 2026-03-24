"""
Texas Flood Risk Modeling (2019–2024)
==================================================

Modules:
  1. Flood Frequency Analysis  — fit GEV / Log-Pearson III distributions to
                                  USGS peak flow records; estimate 10-, 50-,
                                  100-year return period flows per county
  2. Regression Model          — predict NFIP claim counts from:
                                    • precipitation anomaly  (RDS datasets from 
                                                                Robbie M Parks and Victoria Lynch)
                                    • impervious surface pct (NLCD static lookup)
                                    • county median income   (Census ACS API)
                                    • return period exceedance probability
                                  using OLS + a Random Forest for comparison
  
"""

import io
import os, time, warnings
import subprocess
warnings.filterwarnings("ignore")

import certifi
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# Keep matplotlib's cache inside the project so imports work in restricted envs.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from scipy.stats import genextreme, pearson3
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import pygris
    HAS_PYGRIS = True
except ImportError:
    HAS_PYGRIS = False

os.makedirs("outputs", exist_ok=True)

# reuse set constants

YEARS      = list(range(2019, 2025))
STATE_FIPS = "48"
CRS_WGS    = "EPSG:4326"
CRS_PROJ   = "EPSG:3083"

PLOT_BG   = "#ffffff"
PANEL_BG  = "#f7f7f7"
TEXT_COLOR = "#1f1f1f"

# Return periods to estimate (years)
RETURN_PERIODS = [2, 10, 50, 100]

print("Imports OK\n")


# HELPERS 

def get_texas_counties() -> gpd.GeoDataFrame:
    """Reuse the same county boundary loader from the main script."""
    if HAS_PYGRIS:
        gdf = pygris.counties(state="TX", cb=True, year=2022, cache=True)
    else:
        url = ("https://www2.census.gov/geo/tiger/GENZ2022/shp/"
               "cb_2022_us_county_500k.zip")
        gdf = gpd.read_file(url)
        gdf = gdf[gdf["STATEFP"] == STATE_FIPS].copy()

    gdf = gdf.rename(columns=lambda c: c.upper() if c != "geometry" else c)
    for old, new in [("NAMELSAD","COUNTY_NAME"), ("NAME","NAME")]:
        if old in gdf.columns and new not in gdf.columns:
            gdf[new] = gdf[old]
    if "COUNTY_NAME" not in gdf.columns and "NAME" in gdf.columns:
        gdf["COUNTY_NAME"] = gdf["NAME"] + " County"
    if "GEOID" not in gdf.columns:
        for cand in ["AFFGEOID","GEO_ID"]:
            if cand in gdf.columns:
                gdf["GEOID"] = gdf[cand].str[-5:]
                break
    gdf = gdf[["GEOID","COUNTY_NAME","geometry"]].copy()
    gdf["GEOID"] = gdf["GEOID"].str.zfill(5)
    return gdf


def load_panel_csv(path="outputs/texas_flood_summary.csv") -> pd.DataFrame:
    """
    Load the panel CSV produced by texas_flood_analysis.py.
    If it doesn't exist yet, return an empty DataFrame — the regression
    module will still run on whatever data it can fetch independently.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"GEOID": str})
        df["GEOID"] = df["GEOID"].str.zfill(5)
        print(f"Loaded panel from {path}  ({len(df):,} rows)")
        return df
    else:
        print(f"{path} not found — run texas_flood_analysis.py first for "
              "best results. Continuing with independently fetched data.")
        return pd.DataFrame()



# PART 1 — FLOOD FREQUENCY ANALYSIS

def fetch_all_tx_peak_flows_historical() -> pd.DataFrame:
    """
    Pull the FULL historical peak-flow record for Texas (all available years),
    not just 2019-2024.  Frequency analysis needs as many years as possible —
    ideally 30-50+ — to fit reliable distribution tails.
    Also fetches site coordinates in a second call.
    """
    # Peak flows 
    pk_url = ("https://nwis.waterdata.usgs.gov/tx/nwis/peak"
              "?format=rdb&state_cd=TX")
    print("Fetching full historical USGS peak flows for Texas (all years) …")
    r = requests.get(pk_url, timeout=180)
    r.raise_for_status()

    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    header, rows = None, []
    for line in lines:
        parts = line.split("\t")
        if header is None and "site_no" in parts:
            header = parts
        elif header and len(parts) == len(header):
            if not all(p.strip().replace("s","").replace("d","")
                         .replace("n","").isdigit() for p in parts):
                rows.append(parts)

    if not header or not rows:
        raise RuntimeError("Could not parse USGS peak-flow table.")

    pk = pd.DataFrame(rows, columns=header)
    pk["peak_va"] = pd.to_numeric(pk["peak_va"], errors="coerce")
    pk["peak_dt"] = pd.to_datetime(pk["peak_dt"], errors="coerce")
    pk["year"]    = pk["peak_dt"].dt.year
    pk = pk.dropna(subset=["peak_va","year"]).copy()

# ── Remove USGS sentinel values and physically implausible flows ──────────
# USGS encodes missing/unreliable readings as 999999 or 9999999.
# No Texas river exceeds ~5,000,000 cfs even in the most extreme events.
    before = len(pk)
    pk = pk[(pk["peak_va"] > 0) & (pk["peak_va"] < 5_000_000)].copy()
    print(f"  Removed {before - len(pk):,} sentinel/implausible values "
        f"(max remaining: {pk['peak_va'].max():,.0f} cfs)")

    print(f"  → {len(pk):,} historical peak-flow records.")

    # Site coordinates 
    site_nos = pk["site_no"].dropna().unique().tolist()
    print(f"  Fetching coordinates for {len(site_nos):,} gauges …")
    site_rows = []
    for i in range(0, len(site_nos), 100):
        chunk = site_nos[i:i+100]
        su = ("https://waterservices.usgs.gov/nwis/site/?format=rdb"
              f"&sites={','.join(chunk)}&siteOutput=expanded&siteStatus=all")
        sr = requests.get(su, timeout=60)
        if sr.status_code != 200:
            continue
        sl = [l for l in sr.text.splitlines() if not l.startswith("#")]
        sh = None
        for line in sl:
            p = line.split("\t")
            if sh is None and "site_no" in p:
                sh = p
            elif sh and len(p) == len(sh):
                if not all(x.strip().replace("s","").replace("d","")
                             .replace("n","").isdigit() for x in p):
                    site_rows.append(dict(zip(sh, p)))
        time.sleep(0.15)

    sites = pd.DataFrame(site_rows)
    lat_candidates = [
        "dec_lat_va",
        "dec_lat",
        "latitude",
        "lat",
    ]
    lon_candidates = [
        "dec_long_va",
        "dec_long",
        "longitude",
        "long",
        "lon",
        "lng",
    ]
    lat_col = next((c for c in lat_candidates if c in sites.columns), None)
    lon_col = next((c for c in lon_candidates if c in sites.columns), None)

    if lat_col is None:
        lat_col = next((c for c in sites.columns if "lat" in c.lower()), None)
    if lon_col is None:
        lon_col = next((c for c in sites.columns if any(
            token in c.lower() for token in ["long", "lon", "lng"]
        )), None)

    if lat_col and lon_col:
        sites["lat"] = pd.to_numeric(sites[lat_col], errors="coerce")
        sites["lon"] = pd.to_numeric(sites[lon_col], errors="coerce")
        sites = sites[["site_no","lat","lon"]].dropna()

    pk = pk.merge(sites, on="site_no", how="left")

    # Spatial join to county GEOID 
    has_coords = pk.dropna(subset=["lat","lon"])
    if not has_coords.empty:
        pk_gdf   = gpd.GeoDataFrame(
            has_coords,
            geometry=gpd.points_from_xy(has_coords["lon"], has_coords["lat"]),
            crs=CRS_WGS,
        )
        counties = get_texas_counties()
        joined   = gpd.sjoin(pk_gdf, counties[["GEOID","geometry"]],
                             how="left", predicate="within")
        pk.loc[has_coords.index, "GEOID"] = joined["GEOID"].values

    pk = pk.dropna(subset=["GEOID","peak_va"]).copy()
    print(f"  → {len(pk):,} records with county assignment.")
    return pk


def fit_flood_frequency(pk: pd.DataFrame) -> pd.DataFrame:
    """
    For each county, fit a GEV distribution to annual maximum peak flows
    using scipy.stats.genextreme and return T-year quantile estimates.

    Why GEV?  The Generalised Extreme Value distribution is the theoretically
    justified model for block maxima (annual peaks).  Log-Pearson III (LP3)
    is the US federal standard (Bulletin 17C) — we fit both and flag which
    fit better by AIC.
    """
    results = []

    # Use one annual maximum per county-year for frequency fitting.
    # Pooling all gauge-level annual maxima within a county would overweight
    # counties with many gauges and violate the block-maxima assumption.
    ann_max = (pk.groupby(["GEOID","year"])["peak_va"]
                 .max()
                 .reset_index())

    county_groups = ann_max.groupby("GEOID")
    total = len(county_groups)
    print(f"\nFitting flood frequency distributions for {total} counties …")
    gev_warning_count = 0

    for idx, (geoid, grp) in enumerate(county_groups):
        if idx % 50 == 0:
            print(f"  … {idx}/{total}", end="\r")

        # Fit the county-year annual maxima series
        q_series = grp["peak_va"].dropna().values
        n        = len(q_series)

        if n < 10:          # need at least 10 data points for a meaningful fit
            continue

        row = {"GEOID": geoid, "n_obs": n}

        # GEV fit 
        # Add this right before the GEV fit block, inside the county loop
        if idx < 3:   # just print the first 3 counties
            print(f"\n  {geoid} — n={n}  "
                f"min={q_series.min():,.0f}  "
                f"max={q_series.max():,.0f}  "
                f"mean={q_series.mean():,.0f} cfs")
        try:
            c_gev, loc_gev, scale_gev = genextreme.fit(q_series)
            if not all(np.isfinite([c_gev, loc_gev, scale_gev])):
                raise ValueError("Non-finite GEV parameters.")

            gev_quantiles = {}
            for T in RETURN_PERIODS:
                p_exceed = 1 / T
                gev_quantiles[f"Q{T}yr_gev"] = float(
                    genextreme.ppf(1 - p_exceed, c_gev, loc_gev, scale_gev)
                )

            if not all(np.isfinite(list(gev_quantiles.values()))):
                raise ValueError("Non-finite GEV return-period estimate.")

            row.update(gev_quantiles)
            row["gev_shape"] = c_gev

            log_lik_gev = np.sum(genextreme.logpdf(
                q_series, c_gev, loc_gev, scale_gev))
            if np.isfinite(log_lik_gev):
                row["aic_gev"] = 2*3 - 2*log_lik_gev
            else:
                gev_warning_count += 1
                print(
                    f"Warning: {geoid} GEV log-likelihood is non-finite; "
                    "keeping return levels but leaving aic_gev blank."
                )

            warning_bits = []
            if abs(c_gev) > 0.5:
                warning_bits.append(
                    f"shape={c_gev:.3f} outside the usual screening range"
                )
            if row["Q100yr_gev"] <= 0:
                warning_bits.append(
                    f"Q100={row['Q100yr_gev']:.0f} cfs is non-positive"
                )
            elif row["Q100yr_gev"] > np.nanmax(q_series) * 50:
                warning_bits.append(
                    f"Q100={row['Q100yr_gev']:.0f} cfs is very large versus "
                    f"max observed {np.nanmax(q_series):.0f} cfs"
                )

            if warning_bits:
                gev_warning_count += 1
                print(
                    f"Warning: {geoid} retained GEV fit with "
                    + "; ".join(warning_bits)
                )
        except Exception as e:
            for T in RETURN_PERIODS:
                row.pop(f"Q{T}yr_gev", None)
            row.pop("aic_gev", None)
            row.pop("gev_shape", None)
            print(f"Warning: {geoid} GEV fit failed and was skipped: {e}")

        # Log-Pearson III fit (take log and fit Pearson3)
        try:
            log_q = np.log(q_series[q_series > 0])
            skew_lp3, loc_lp3, scale_lp3 = pearson3.fit(log_q)
            for T in RETURN_PERIODS:
                p_exceed      = 1 / T
                log_q_T       = pearson3.ppf(1 - p_exceed,
                                             skew_lp3, loc_lp3, scale_lp3)
                row[f"Q{T}yr_lp3"] = float(np.exp(log_q_T))
            if (
                not np.isfinite(row["Q100yr_lp3"]) or
                row["Q100yr_lp3"] <= 0 or
                row["Q100yr_lp3"] > np.nanmax(q_series) * 100
            ):
                raise ValueError("Implausible LP3 return-period estimate.")
            log_lik_lp3 = np.sum(pearson3.logpdf(
                log_q, skew_lp3, loc_lp3, scale_lp3))
            if not np.isfinite(log_lik_lp3):
                raise ValueError("Non-finite LP3 log-likelihood.")
            row["aic_lp3"] = 2*3 - 2*log_lik_lp3
        except Exception:
            pass

        # Best-fit 100-yr estimate = whichever model has lower AIC
        if "aic_gev" in row and "aic_lp3" in row:
            row["Q100yr_best"] = (row.get("Q100yr_gev") 
                                  if row["aic_gev"] <= row["aic_lp3"]
                                  else row.get("Q100yr_lp3"))
            row["best_model"]  = ("GEV" if row["aic_gev"] <= row["aic_lp3"]
                                  else "LP3")
        elif "aic_gev" in row:
            row["Q100yr_best"] = row.get("Q100yr_gev")
            row["best_model"]  = "GEV"
        elif "aic_lp3" in row:
            row["Q100yr_best"] = row.get("Q100yr_lp3")
            row["best_model"]  = "LP3"

        results.append(row)

    print(f"  → Distributions fitted for {len(results)} counties.")
    if gev_warning_count:
        print(f"  → Retained {gev_warning_count} GEV fits with warnings.")
    df_rp = pd.DataFrame(results)
    df_rp.to_csv("outputs/texas_return_periods.csv", index=False)
    print("Saved → outputs/texas_return_periods.csv")
    return df_rp


def plot_return_period_map(df_rp: pd.DataFrame):
    """Choropleth of estimated 100-year peak flow by county."""
    counties = get_texas_counties()
    gdf = counties.merge(df_rp[["GEOID","Q100yr_best","best_model","n_obs"]],
                         on="GEOID", how="left")
    gdf = gdf.to_crs(CRS_PROJ)
    display_vmax = gdf["Q100yr_best"].dropna().quantile(0.99)
    gdf["Q100yr_plot"] = gdf["Q100yr_best"].clip(upper=display_vmax)

    fig, ax = plt.subplots(figsize=(13, 10), facecolor=PLOT_BG)
    ax.set_facecolor(PANEL_BG)

    gdf.plot(column="Q100yr_plot", ax=ax, cmap="Blues",
             edgecolor="#666", linewidth=0.3,
             legend=False,
             missing_kwds={"color":"#efefef","edgecolor":"#ccc",
                           "label":"Insufficient data"})

    sm = cm.ScalarMappable(
        cmap="Blues",
        norm=mcolors.LogNorm(
            vmin=max(1, gdf["Q100yr_plot"].dropna().min()),
            vmax=gdf["Q100yr_plot"].dropna().max(),
        )
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.7)
    cbar.set_label("Estimated 100-Year Peak Flow (cfs)", color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    ax.set_title("Texas — Estimated 100-Year Flood Peak Flow by County\n"
                 "(GEV / LP3 distribution fit to all available USGS gauge records)",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=10)
    ax.text(0.98, 0.02, "Color scale capped at 99th percentile for readability",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#666")
    ax.axis("off")

    path = "outputs/texas_return_period_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=PLOT_BG)
    print(f"Saved → {path}")
    plt.close(fig)


def plot_frequency_curves(pk: pd.DataFrame, df_rp: pd.DataFrame, n=10):
    """
    Plot GEV frequency curves for the n counties with the most gauge data.
    X-axis = return period (log scale), Y-axis = peak flow (cfs).
    This is the canonical plot used in hydrology reports.
    """
    top_counties = (df_rp.dropna(subset=["Q100yr_best"])
                         .nlargest(n, "n_obs")["GEOID"].tolist())

    counties_meta = get_texas_counties().set_index("GEOID")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor=PLOT_BG)
    fig.suptitle("Flood Frequency Curves — Top 10 Texas Counties by Data Density\n"
                 "Empirical points (Weibull plotting positions) + GEV fit",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    T_range = np.logspace(np.log10(1.01), np.log10(500), 200)
    cmap_c  = plt.get_cmap("tab10")

    ann_max = (pk.groupby(["GEOID","site_no","year"])["peak_va"]
                 .max().reset_index())

    for ax, geoid, color in zip(axes.flat, top_counties,
                                [cmap_c(i) for i in range(n)]):
        grp    = ann_max[ann_max["GEOID"] == geoid]["peak_va"].dropna().values
        if len(grp) < 5:
            ax.set_visible(False)
            continue

        # Empirical plotting positions (Weibull: p = rank / (n+1))
        sorted_q = np.sort(grp)
        n_pts    = len(sorted_q)
        emp_T    = (n_pts + 1) / np.arange(1, n_pts + 1)[::-1]

        ax.scatter(emp_T, sorted_q, color=color, s=18, zorder=5,
                   label="Observed annual max", alpha=0.8)

        # GEV fitted curve
        try:
            c, loc, scale = genextreme.fit(grp)
            fit_q = genextreme.ppf(1 - 1/T_range, c, loc, scale)
            ax.plot(T_range, fit_q, color=color, linewidth=2,
                    label="GEV fit")
        except Exception:
            pass

        # Mark T-year estimates
        rp_row = df_rp[df_rp["GEOID"] == geoid]
        for T, ls in [(10,"--"),(100,":")]:
            col_name = f"Q{T}yr_gev"
            if not rp_row.empty and col_name in rp_row.columns:
                val = rp_row[col_name].values[0]
                if pd.notna(val):
                    ax.axhline(val, color="#888", linewidth=0.8,
                               linestyle=ls, alpha=0.7)
                    ax.text(1.5, val*1.03, f"{T}-yr",
                            fontsize=7, color="#666")

        county_name = (counties_meta.loc[geoid, "COUNTY_NAME"]
                       if geoid in counties_meta.index else geoid)
        ax.set_title(county_name, fontsize=9, color=TEXT_COLOR)
        ax.set_xscale("log")
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.set_xlabel("Return Period (years)", fontsize=7, color=TEXT_COLOR)
        ax.set_ylabel("Peak Flow (cfs)", fontsize=7, color=TEXT_COLOR)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cfcfcf")

        if ax == axes.flat[0]:
            ax.legend(fontsize=6, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = "outputs/texas_flood_frequency_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    print(f"Saved → {path}")
    plt.close(fig)



# PART 2 — FEATURE ENGINEERING

#  County level daily precipitation
#  Source: PRISM, converted by Robbie M Parks and Victoria D Lynch
NOAA_TOKEN = os.environ.get("NOAA_CDO_TOKEN", "").strip()

def fetch_noaa_precip(years: list) -> pd.DataFrame:
    """
    Load county-level Texas precipitation from local daily FIPS datasets.
    Falls back to the NOAA CDO API only if the local datasets are unavailable.
    """
    dataset_dir = "datasets"
    dataset_paths = [
        os.path.join(dataset_dir, f"weighted_area_raster_fips_ppt_daily_{year}.rds")
        for year in years
        if os.path.exists(
            os.path.join(dataset_dir, f"weighted_area_raster_fips_ppt_daily_{year}.rds")
        )
    ]

    if dataset_paths:
        print("Loading county-level precipitation for Texas from local datasets …")
        r_expr = """
args <- commandArgs(trailingOnly = TRUE)
parts <- lapply(args, function(path) {
  x <- readRDS(path)
  x <- as.data.frame(x)
  x$fips <- sprintf("%05s", as.character(x$fips))
  x <- x[substr(x$fips, 1, 2) == "48", c("fips", "ppt", "year")]
  x$ppt <- as.numeric(x$ppt)
  stats::aggregate(ppt ~ fips + year, data = x, FUN = sum, na.rm = TRUE)
})
out <- do.call(rbind, parts)
utils::write.table(out, file = "", sep = ",", row.names = FALSE, col.names = TRUE)
""".strip()
        try:
            result = subprocess.run(
                ["Rscript", "-e", r_expr, *dataset_paths],
                check=True,
                capture_output=True,
                text=True,
            )
            df = pd.read_csv(io.StringIO(result.stdout), dtype={"fips": str})
            df = df.rename(columns={"fips": "GEOID", "ppt": "precip_mm"})
            df["GEOID"] = df["GEOID"].str.zfill(5)
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
            df = df.dropna(subset=["GEOID", "year", "precip_mm"]).copy()
            df["year"] = df["year"].astype(int)
            df = df[df["year"].isin(years)].copy()
            county_baseline = df.groupby("GEOID")["precip_mm"].transform("mean")
            df["precip_anom"] = df["precip_mm"] - county_baseline
            print(f"  → Loaded local precipitation for {df['GEOID'].nunique()} Texas counties.")
            return df[["GEOID", "year", "precip_mm", "precip_anom"]]
        except Exception as e:
            print(f"  Local precipitation load failed ({e}) — falling back to NOAA CDO API.")

    if not NOAA_TOKEN:
        print("NOAA_CDO_TOKEN not set or invalid after trimming whitespace — "
              "using zero precipitation anomaly.\n"
              "    Get a free token at https://www.ncdc.noaa.gov/cdo-web/token\n"
              "    Then: export NOAA_CDO_TOKEN=your_token_here")
        geoids = get_texas_counties()["GEOID"].tolist()
        rows   = [{"GEOID": g, "year": y, "precip_mm": 0, "precip_anom": 0}
                  for g in geoids for y in years]
        return pd.DataFrame(rows)

    base = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": NOAA_TOKEN}
    records = []
    print("Fetching NOAA annual precipitation for Texas …")

    for year in years:
        params = {
            "datasetid":  "GHCND",
            "datatypeid": "PRCP",
            "locationid": "FIPS:48",   # Texas
            "startdate":  f"{year}-01-01",
            "enddate":    f"{year}-12-31",
            "units":      "metric",
            "limit":      1000,
        }
        try:
            r = requests.get(base, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            data = r.json().get("results", [])
            for rec in data:
                records.append({
                    "year":     year,
                    "station":  rec.get("station",""),
                    "precip_mm":rec.get("value", 0),
                })
        except Exception as e:
            print(f"  NOAA fetch failed for {year}: {e}")
        time.sleep(0.5)

    if not records:
        geoids = get_texas_counties()["GEOID"].tolist()
        rows   = [{"GEOID": g, "year": y, "precip_mm": 0, "precip_anom": 0}
                  for g in geoids for y in years]
        return pd.DataFrame(rows)

    df = pd.DataFrame(records)
    annual = df.groupby("year")["precip_mm"].mean().reset_index()
    baseline = annual["precip_mm"].mean()
    annual["precip_anom"] = annual["precip_mm"] - baseline

    # Broadcast statewide anomaly to all counties (refineable with station mapping)
    geoids = get_texas_counties()["GEOID"].tolist()
    panel  = pd.DataFrame(
        [{"GEOID": g, "year": y} for g in geoids for y in years])
    panel  = panel.merge(annual[["year","precip_mm","precip_anom"]],
                         on="year", how="left")
    return panel


# NLCD Impervious Surface
# Use impervious surface data from National Land Cover Dataset
# Source: https://www.mrlc.gov/

def fetch_impervious_surface(
    nlcd_raster_path="datasets/Annual_NLCD_2019/Annual_NLCD_2019.tif",
    cache_dir="datasets",
):
    """
    Compute county-level impervious surface % from NLCD raster.

    This implementation avoids optional dependencies like rasterstats/fiona by
    using the project's county loader plus rasterio masking directly.
    """
    os.makedirs(cache_dir, exist_ok=True)
    county_cache_path = os.path.join(cache_dir, "texas_impervious_surface_county.csv")

    # Use cached CSV if it exists
    if os.path.exists(county_cache_path):
        print(f"  Using cached county data → {county_cache_path}")
        county_imp = pd.read_csv(county_cache_path, dtype={"GEOID": str})
        county_imp["impervious_pct"] = pd.to_numeric(
            county_imp["impervious_pct"], errors="coerce"
        )
        return county_imp.dropna(subset=["GEOID", "impervious_pct"])

    if not os.path.exists(nlcd_raster_path):
        raise FileNotFoundError(
            f"NLCD raster not found at {nlcd_raster_path}. "
            "Place the TIFF there or pass a different nlcd_raster_path."
        )

    counties = get_texas_counties().copy()

    print("  Computing county-level impervious surface from NLCD raster …")
    with rasterio.open(nlcd_raster_path) as src:
        counties = counties.to_crs(src.crs)
        nodata = src.nodata
        means = []

        for idx, row in enumerate(counties.itertuples(index=False), start=1):
            if idx % 25 == 0 or idx == len(counties):
                print(f"    processed {idx}/{len(counties)} counties …", end="\r")
            try:
                arr, _ = mask(
                    src,
                    [row.geometry.__geo_interface__],
                    crop=True,
                    indexes=1,
                    filled=False,
                )
                data = arr.compressed()
                if nodata is not None:
                    data = data[data != nodata]
                means.append(float(data.mean()) if data.size else np.nan)
            except Exception:
                means.append(np.nan)

    print(" " * 60, end="\r")
    counties["impervious_pct"] = means

    # Save CSV for future runs
    county_imp = counties[["GEOID", "impervious_pct"]].copy()
    county_imp.to_csv(county_cache_path, index=False)
    print(f"  → {len(county_imp)} counties. Cached → {county_cache_path}")
    return county_imp
    
# Census ACS Median Household Income 

def fetch_census_income() -> pd.DataFrame:
    """
    Pull median household income (B19013_001E) for Texas counties
    from the Census ACS 5-year API.
    """
    url = ("https://api.census.gov/data/2022/acs/acs5"
           "?get=B19013_001E,NAME"
           "&for=county:*"
           "&in=state:48")
    print("Fetching Census ACS median household income …")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        df   = pd.DataFrame(data[1:], columns=data[0])
        df["GEOID"]         = df["state"] + df["county"].str.zfill(3)
        df["median_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
        df = df[["GEOID","median_income"]].dropna()
        print(f"  → Income data for {len(df)} counties.")
        return df
    except Exception as e:
        print(f"Census fetch failed ({e}) — income feature will be omitted.")
        return pd.DataFrame(columns=["GEOID","median_income"])


# PART 3 — REGRESSION MODEL

def build_model_dataset(panel: pd.DataFrame,
                         df_rp: pd.DataFrame,
                         precip: pd.DataFrame,
                         impervious: pd.DataFrame,
                         income: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all features into a single modelling DataFrame.
    Target variable: log1p(claim_count) — log-transform stabilises variance.
    """
    df = panel[["GEOID","year","claim_count","total_payout"]].copy()
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)

    # Return period exceedance: P(Q > Q100) proxy per county
    # = 1 / 100 by definition, but we use Q10 as a dynamic feature:
    #   counties where Q_10yr is low are more flood-prone per unit rainfall
    rp_cols = ["GEOID"] + [c for c in df_rp.columns if c.startswith("Q")]
    df = df.merge(df_rp[rp_cols + ["best_model","n_obs"]].drop_duplicates("GEOID"),
                  on="GEOID", how="left")

    # Precipitation anomaly
    if not precip.empty:
        df = df.merge(precip[["GEOID","year","precip_mm","precip_anom"]],
                      on=["GEOID","year"], how="left")

    # Impervious surface
    if not impervious.empty:
        df = df.merge(impervious, on="GEOID", how="left")

    # Income
    if not income.empty:
        df = df.merge(income, on="GEOID", how="left")

    # derived features
    # Use LP3 Q10 as the regression hazard proxy for stability.
    if "Q10yr_lp3" in df.columns:
        df["log_Q10_lp3"] = np.log1p(df["Q10yr_lp3"].clip(lower=0))
    if "Q100yr_best" in df.columns:
        df["log_Q100_best"] = np.log1p(df["Q100yr_best"].clip(lower=0))

    # year trend
    df["year_trend"] = df["year"] - YEARS[0]

    # log-transform target
    df["log_claim_count"] = np.log1p(df["claim_count"].fillna(0).clip(lower=0))

    return df


def run_ols(df_model: pd.DataFrame) -> tuple:
    """
    OLS regression: log_claim_count ~ features.
    Returns (results, feature_list, X, y).
    """
    features = []
    if "precip_anom"    in df_model.columns: features.append("precip_anom")
    if "impervious_pct" in df_model.columns: features.append("impervious_pct")
    if "median_income"  in df_model.columns: features.append("median_income")
    if "log_Q10_lp3"    in df_model.columns: features.append("log_Q10_lp3")
    if "year_trend"     in df_model.columns: features.append("year_trend")

    df_clean = df_model[features + ["log_claim_count"]].dropna()
    X = df_clean[features]
    y = df_clean["log_claim_count"]

    X_const = sm.add_constant(X)
    ols     = sm.OLS(y, X_const).fit(cov_type="HC3")  # heteroscedasticity-robust SEs

    print("\n" + "="*60)
    print("  OLS REGRESSION RESULTS")
    print("="*60)
    print(ols.summary())

    with open("outputs/texas_regression_results.txt", "w") as f:
        f.write(str(ols.summary()))
        f.write("\n\nFeatures used: " + ", ".join(features))
    print(" Saved → outputs/texas_regression_results.txt")

    return ols, features, X, y


def run_random_forest(X: pd.DataFrame, y: pd.Series,
                      features: list) -> RandomForestRegressor:
    """
    Random Forest regressor — captures non-linear interactions between
    impervious surface, precipitation, and flood frequency.
    Reports 5-fold cross-validated R² and feature importances.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                                min_samples_leaf=5, random_state=42,
                                n_jobs=-1)
    cv_r2 = cross_val_score(rf, X_sc, y, cv=5, scoring="r2")
    print(f"\n  Random Forest  |  5-fold CV R² = "
          f"{cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    rf.fit(X_sc, y)
    importances = pd.Series(rf.feature_importances_, index=features).sort_values()

    # Append RF results to text file
    with open("outputs/texas_regression_results.txt", "a") as f:
        f.write("\n\n" + "="*60)
        f.write(f"\nRANDOM FOREST  —  5-fold CV R²: {cv_r2.mean():.3f} "
                f"± {cv_r2.std():.3f}\n")
        f.write("\nFeature importances:\n")
        f.write(importances.to_string())

    return rf, scaler, importances


def adjusted_r2_score(y_true, y_pred, n_features: int) -> float:
    """Adjusted R^2 from predictions, with a safe fallback for small samples."""
    n_obs = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n_obs <= n_features + 1:
        return r2
    return 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_features - 1)

def run_temporal_holdout(df_model: pd.DataFrame, features: list,
                          train_years=(2019, 2022), test_years=(2023, 2024)):
    """
    Train on 2019-2022, evaluate on 2023-2024.
    More honest than k-fold CV for time-series panel data because
    it never lets the model see future data during training.
    """
    df_clean = df_model[features + ["log_claim_count", "year"]].dropna()

    train = df_clean[df_clean["year"].between(*train_years)]
    test  = df_clean[df_clean["year"].between(*test_years)]

    X_train, y_train = train[features], train["log_claim_count"]
    X_test,  y_test  = test[features],  test["log_claim_count"]

    print(f"\n  Temporal holdout split:")
    print(f"    Train: {train_years[0]}–{train_years[1]}  ({len(train):,} rows)")
    print(f"    Test:  {test_years[0]}–{test_years[1]}   ({len(test):,} rows)")

    #  OLS 
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type="HC3")
    ols_pred = ols.predict(sm.add_constant(X_test))
    ols_r2   = r2_score(y_test, ols_pred)
    ols_adj_r2 = adjusted_r2_score(y_test, ols_pred, len(features))
    ols_mae  = mean_absolute_error(y_test, ols_pred)
    print(f"\n  OLS holdout     R² = {ols_r2:.3f}   MAE = {ols_mae:.3f}")

    # Random Forest 
    scaler = StandardScaler()
    rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                                min_samples_leaf=5, random_state=42,
                                n_jobs=-1)
    rf.fit(scaler.fit_transform(X_train), y_train)
    rf_pred = rf.predict(scaler.transform(X_test))
    rf_r2   = r2_score(y_test, rf_pred)
    rf_adj_r2 = adjusted_r2_score(y_test, rf_pred, len(features))
    rf_mae  = mean_absolute_error(y_test, rf_pred)
    print(f"  RF  holdout     R² = {rf_r2:.3f}   MAE = {rf_mae:.3f}")

    # Summary comparison table 
    print("\n  CV vs Holdout R² comparison:")
    print(f"    OLS            CV R² = 0.171   →   Holdout R² = {ols_r2:.3f}")
    print(f"    Random Forest  CV R² = 0.519   →   Holdout R² = {rf_r2:.3f}")
    print("  (CV R² values from earlier run — update manually if re-run)")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=PLOT_BG)
    fig.suptitle("Temporal Holdout Validation  —  Trained 2019–2022 / Tested 2023–2024",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    for ax, preds, label, color, adj_r2, mae in [
        (ax1, ols_pred, "OLS",           "#2196f3", ols_adj_r2, ols_mae),
        (ax2, rf_pred,  "Random Forest", "#4caf50", rf_adj_r2,  rf_mae),
    ]:
        ax.scatter(y_test, preds, alpha=0.35, s=14,
                   color=color, edgecolors="none")
        lims = [min(float(y_test.min()), float(preds.min())),
                max(float(y_test.max()), float(preds.max()))]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        if label == "OLS":
            ax.set_title(
                f"{label}\nHoldout R² = {ols_r2:.3f}   Adjusted R² = {adj_r2:.3f}",
                color=TEXT_COLOR,
                fontsize=11,
            )
        else:
            ax.set_title(
                f"{label}\nHoldout R² = {rf_r2:.3f}   MAE = {mae:.3f}",
                color=TEXT_COLOR,
                fontsize=11,
            )
        ax.set_xlabel("Actual  log(claim count + 1)", color=TEXT_COLOR)
        ax.set_ylabel("Predicted", color=TEXT_COLOR)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR)
        ax.legend(fontsize=8, framealpha=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cfcfcf")

    fig.text(0.5, 0.01,
             "Model trained exclusively on 2019–2022  |  "
             "2023–2024 data withheld during training",
             ha="center", color="#888", fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = "outputs/texas_temporal_holdout.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    print(f"Saved → {path}")
    plt.close(fig)

    # Append to results file
    with open("outputs/texas_regression_results.txt", "a") as f:
        f.write("\n\n" + "="*60)
        f.write("\nTEMPORAL HOLDOUT  —  Train: 2019-2022  |  Test: 2023-2024\n")
        f.write(f"OLS          R² = {ols_r2:.3f}   MAE = {ols_mae:.3f}\n")
        f.write(f"RandomForest R² = {rf_r2:.3f}   MAE = {rf_mae:.3f}\n")

    return {"ols_r2": ols_r2, "rf_r2": rf_r2,
            "ols_mae": ols_mae, "rf_mae": rf_mae}


def plot_regression_diagnostics(ols_results, rf, scaler,
                                  X: pd.DataFrame, y: pd.Series,
                                  importances: pd.Series):
    """Four-panel diagnostic figure."""
    y_pred_ols = ols_results.predict(sm.add_constant(X))
    X_sc       = scaler.transform(X)
    y_pred_rf  = rf.predict(X_sc)
    n_features = len(importances)
    ols_adj_r2 = adjusted_r2_score(y, y_pred_ols, n_features)
    rf_adj_r2 = adjusted_r2_score(y, y_pred_rf, n_features)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=PLOT_BG)
    fig.suptitle("Texas Flood Claim Count — Regression Diagnostics",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    # Panel 1: OLS actual vs predicted 
    ax = axes[0, 0]
    ax.scatter(y, y_pred_ols, alpha=0.25, s=10, color="#2196f3",
               edgecolors="none")
    lims = [min(y.min(), y_pred_ols.min()), max(y.max(), y_pred_ols.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Actual log(claim count + 1)", color=TEXT_COLOR)
    ax.set_ylabel("Predicted", color=TEXT_COLOR)
    ax.set_title(f"OLS  –  Actual vs Predicted\n"
                 f"R² = {r2_score(y, y_pred_ols):.3f}  "
                 f"Adjusted R² = {ols_adj_r2:.3f}", color=TEXT_COLOR)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for sp in ax.spines.values(): sp.set_edgecolor("#cfcfcf")

    # Panel 2: OLS residuals 
    ax = axes[0, 1]
    resid = y - y_pred_ols
    ax.scatter(y_pred_ols, resid, alpha=0.25, s=10, color="#ff7043",
               edgecolors="none")
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xlabel("Fitted values", color=TEXT_COLOR)
    ax.set_ylabel("Residuals", color=TEXT_COLOR)
    ax.set_title("OLS Residuals vs Fitted", color=TEXT_COLOR)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for sp in ax.spines.values(): sp.set_edgecolor("#cfcfcf")

    # Panel 3: RF actual vs predicted 
    ax = axes[1, 0]
    ax.scatter(y, y_pred_rf, alpha=0.25, s=10, color="#4caf50",
               edgecolors="none")
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual log(claim count + 1)", color=TEXT_COLOR)
    ax.set_ylabel("Predicted", color=TEXT_COLOR)
    ax.set_title(f"Random Forest  –  Actual vs Predicted\n"
                 f"R² = {r2_score(y, y_pred_rf):.3f}  "
                 f"MAE = {mean_absolute_error(y, y_pred_rf):.3f}",
                 color=TEXT_COLOR)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for sp in ax.spines.values(): sp.set_edgecolor("#cfcfcf")

    # Panel 4: RF feature importances 
    ax = axes[1, 1]
    colors_imp = plt.get_cmap("Blues")(
        np.linspace(0.35, 0.85, len(importances)))
    bars = ax.barh(importances.index, importances.values,
                   color=colors_imp, edgecolor="#aaa", linewidth=0.5)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)",
                  color=TEXT_COLOR)
    ax.set_title("Random Forest Feature Importances", color=TEXT_COLOR)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#cfcfcf")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = "outputs/texas_regression_actual_vs_pred.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    print(f"Saved → {path}")
    plt.close(fig)


# PART 4 — FINAL RISK SCORE TABLE

def build_final_risk_table(panel: pd.DataFrame,
                            df_rp: pd.DataFrame,
                            impervious: pd.DataFrame,
                            income: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean, portfolio-ready county risk table combining:
      - Historical exposure (FEMA claims)
      - Physical hazard (100-yr return period flow)
      - Vulnerability proxies (impervious surface, income)
    Normalise each dimension 0-100 and compute a composite Risk Index.
    """
    # Aggregate claims across all years
    hist = (panel.groupby("GEOID", as_index=False)
                 .agg(total_claims=("claim_count","sum"),
                      total_payout=("total_payout","sum"),
                      years_with_claims=("claim_count",
                                         lambda x: (x > 0).sum())))

    df = hist.copy()
    df = df.merge(df_rp[["GEOID","Q100yr_best","Q10yr_gev",
                          "best_model","n_obs"]].drop_duplicates("GEOID"),
                  on="GEOID", how="left")
    if not impervious.empty:
        df = df.merge(impervious, on="GEOID", how="left")
    if not income.empty:
        df = df.merge(income, on="GEOID", how="left")

    counties_meta = get_texas_counties()[["GEOID","COUNTY_NAME"]]
    df = df.merge(counties_meta, on="GEOID", how="left")

    def norm100(s):
        mn, mx = s.min(), s.max()
        return ((s - mn) / (mx - mn + 1e-9) * 100).round(1)

    df["exposure_idx"]     = norm100(df["total_claims"].fillna(0))
    df["hazard_idx"]       = norm100(df["Q100yr_best"].fillna(0))
    df["imperv_idx"]       = norm100(df["impervious_pct"].fillna(0)) \
                             if "impervious_pct" in df.columns else 0
    # Income: lower income = higher vulnerability → invert
    if "median_income" in df.columns:
        df["vuln_idx"] = norm100(1 / df["median_income"].replace(0, np.nan)
                                      .fillna(df["median_income"].median()))
    else:
        df["vuln_idx"] = 0

    df["risk_index"] = (
        0.40 * df["exposure_idx"] +
        0.35 * df["hazard_idx"]   +
        0.15 * df["imperv_idx"]   +
        0.10 * df["vuln_idx"]
    ).round(2)

    df = df.sort_values("risk_index", ascending=False)
    path = "outputs/texas_risk_score_final.csv"
    df.to_csv(path, index=False)
    print(f"\n Final risk table saved → {path}")
    print(f"Top 5 highest-risk counties:\n"
          + df[["COUNTY_NAME","risk_index","total_claims",
                "Q100yr_best","impervious_pct"]]
               .head(5).to_string(index=False))
    return df


# MAIN

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Texas Flood Risk Model Analysis")
    print("="*60 + "\n")

    #  1. Load base panel 
    panel = load_panel_csv()

    # 2. Flood frequency analysis 
    print("\n PART 1: Flood Frequency Analysis ")
    try:
        pk_hist = fetch_all_tx_peak_flows_historical()
        df_rp   = fit_flood_frequency(pk_hist)
        plot_return_period_map(df_rp)
        plot_frequency_curves(pk_hist, df_rp)
    except Exception as e:
        print(f"Flood frequency analysis failed: {e}")
        df_rp = pd.DataFrame(columns=["GEOID","Q10yr_gev","Q10yr_lp3",
                                       "Q100yr_best","best_model","n_obs"])

    # 3. Feature engineering 
    print("\n PART 2: Feature Engineering ")
    precip    = fetch_noaa_precip(YEARS)
    impervious = fetch_impervious_surface()
    income     = fetch_census_income()

    # 4. Regression model 
    if not panel.empty:
        print("\n PART 3: Regression Model ")
        df_model = build_model_dataset(panel, df_rp, precip, impervious, income)
        ols_res, features, X, y = run_ols(df_model)
        rf, scaler, importances = run_random_forest(X, y, features)
        plot_regression_diagnostics(ols_res, rf, scaler, X, y, importances)
        holdout = run_temporal_holdout(df_model, features)
    else:
        print("\n Skipping regression — no panel data found.\n"
              "     Run texas_flood_analysis.py first, then re-run this script.")
        ols_res = rf = scaler = importances = None

    # 5. Final risk table 
    print("\n PART 4: Final Risk Score Table ")
    if not panel.empty:
        build_final_risk_table(panel, df_rp, impervious, income)

    print("\n" + "="*60)
    print("  All outputs written to ./outputs/")
    print("  Run texas_flood_analysis.py first if panel CSV is missing.")
    print("="*60)
