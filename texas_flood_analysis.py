"""
Texas Flood & Water Exposure Analysis (2019–2024)
==================================================
Data sources:
  • FEMA OpenFEMA – NFIP flood insurance claims by county
  • USGS NWIS    – Streamgage peak-flow / flood stage data
  • Census TIGER – Texas county boundaries (via pygris / geopandas)
  • NOAA CDO     – Monthly precipitation (optional, see config)

Outputs (written to ./outputs/):
  • texas_flood_choropleth_<year>.png  – per-year static maps
  • texas_flood_animated.gif            – animated choropleth 2019-2024
  • texas_flood_interactive.html        – Folium interactive map
  • texas_flood_timeseries.png          – county-level time-series panel
  • texas_flood_summary.csv             – cleaned merged dataset
"""

# ── 0. Imports & config ────────────────────────────────────────────────────────
import warnings, os, time
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import folium
import branca.colormap as bcm
from folium.plugins import TimeSliderChoropleth
import contextily as ctx

# Try optional packages
try:
    import pygris
    HAS_PYGRIS = True
except ImportError:
    HAS_PYGRIS = False

os.makedirs("outputs", exist_ok=True)

YEARS       = list(range(2019, 2025))   # 2019–2024
STATE_FIPS  = "48"                       # Texas
CRS_PROJ    = "EPSG:3083"               # Texas Albers (meters)
CRS_WGS     = "EPSG:4326"

print("Imports are ok")

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXAS COUNTY BOUNDARIES
# ─────────────────────────────────────────────────────────────────────────────
def get_texas_counties() -> gpd.GeoDataFrame:
    """Download Texas county polygons from Census TIGER via pygris or direct URL."""
    if HAS_PYGRIS:
        print("Fetching county boundaries via pygris …")
        gdf = pygris.counties(state="TX", cb=True, year=2022, cache=True)
    else:
        # Fallback: Census TIGER shapefile (zipped)
        url = ("https://www2.census.gov/geo/tiger/GENZ2022/shp/"
               "cb_2022_us_county_500k.zip")
        print(f"Fetching county boundaries from Census TIGER …\n  {url}")
        gdf = gpd.read_file(url)
        gdf = gdf[gdf["STATEFP"] == STATE_FIPS].copy()

    # Uppercase only non-geometry columns to avoid breaking geopandas
    gdf = gdf.rename(columns=lambda c: c.upper() if c != "geometry" else c)

    # Normalise column names across pygris / direct versions
    for old, new in [("NAMELSAD", "COUNTY_NAME"), ("NAME", "NAME")]:
        if old in gdf.columns and new not in gdf.columns:
            gdf[new] = gdf[old]
    if "COUNTY_NAME" not in gdf.columns and "NAME" in gdf.columns:
        gdf["COUNTY_NAME"] = gdf["NAME"] + " County"

    # Ensure GEOID exists — pygris may use AFFGEOID or GEO_ID
    if "GEOID" not in gdf.columns:
        for candidate in ["AFFGEOID", "GEO_ID"]:
            if candidate in gdf.columns:
                gdf["GEOID"] = gdf[candidate].str[-5:]
                break

    gdf = gdf[["GEOID", "COUNTY_NAME", "geometry"]].copy()
    gdf["GEOID"] = gdf["GEOID"].str.zfill(5)
    print(f"  → {len(gdf)} Texas counties loaded.")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEMA NFIP FLOOD INSURANCE CLAIMS (OpenFEMA API)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fema_claims(years: list) -> pd.DataFrame:
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
    start    = f"{min(years)}-01-01"
    end      = f"{max(years)}-12-31"
    expected_cols = [
        "dateOfLoss",
        "state",
        "countyCode",
        "amountPaidOnBuildingClaim",
        "amountPaidOnContentsClaim",
        "buildingDamageAmount",
        "censusTract",
    ]

    params = {
        "$filter": (f"state eq 'TX' and "
                    f"dateOfLoss ge '{start}' and dateOfLoss le '{end}'"),
        "$select": ("dateOfLoss,state,countyCode,"
                    "amountPaidOnBuildingClaim,amountPaidOnContentsClaim,"
                    "buildingDamageAmount,censusTract"),
        "$top":    10000,
        "$skip":   0,
        "$format": "json",
    }

    all_records = []
    print("Fetching FEMA NFIP claims for Texas …")
    while True:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("FimaNfipClaims", data.get("fimaNfipClaims", []))
        if not batch:
            break
        all_records.extend(batch)
        print(f"  fetched {len(all_records):,} records …", end="\r")
        if len(batch) < 10000:
            break
        params["$skip"] += 10000
        time.sleep(0.3)

    df = pd.DataFrame(all_records)
    print(f"\n  → {len(df):,} NFIP claim records fetched.")

    if df.empty:
        return pd.DataFrame(columns=expected_cols + ["year", "total_payout", "GEOID"])

    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["dateOfLoss"] = pd.to_datetime(df["dateOfLoss"], errors="coerce")
    df["year"]       = df["dateOfLoss"].dt.year

    for c in ["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
              "buildingDamageAmount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["total_payout"] = (df.get("amountPaidOnBuildingClaim", 0) +
                          df.get("amountPaidOnContentsClaim",  0))

    county_code = (
        pd.to_numeric(df["countyCode"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.strip()
    )

    df["GEOID"] = np.where(
        county_code.str.fullmatch(r"\d{5}", na=False),
        county_code,
        np.where(
            county_code.str.fullmatch(r"\d{3}", na=False),
            STATE_FIPS + county_code,
            pd.NA,
        ),
    )

    df = df[df["GEOID"].str.fullmatch(r"\d{5}", na=False)].copy()
    df = df[df["year"].between(min(years), max(years))].copy()
    return df


def aggregate_claims(df_claims: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NFIP claims to county x year level."""
    if df_claims.empty:
        return pd.DataFrame(
            columns=["GEOID", "year", "claim_count", "total_payout", "buildings_hit"]
        )

    grp = df_claims.groupby(["GEOID", "year"], as_index=False).agg(
        claim_count   = ("total_payout", "count"),
        total_payout  = ("total_payout", "sum"),
        buildings_hit = ("total_payout", "count"),
    )
    return grp

# FEMA CLAIMS
def fetch_usgs_peak_flows(years: list) -> pd.DataFrame:
    # ── Step 1: Pull peak flow records ────────────────────────────────────────
    pk_url = (
        "https://nwis.waterdata.usgs.gov/tx/nwis/peak"
        "?format=rdb&state_cd=TX"
        f"&begin_date={min(years)}-01-01&end_date={max(years)}-12-31"
    )
    print("Fetching USGS TX peak flow records …")
    r = requests.get(pk_url, timeout=120)
    r.raise_for_status()

    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    header, rows = None, []
    for line in lines:
        parts = line.split("\t")
        if header is None and "site_no" in parts:
            header = parts
        elif header and len(parts) == len(header):
            if not all(p.strip().replace("s","").replace("d","").replace("n","").isdigit()
                       for p in parts):
                rows.append(parts)

    if not header or not rows:
        print("  ⚠  Could not parse USGS peak-flow table — skipping.")
        return pd.DataFrame()

    pk = pd.DataFrame(rows, columns=header)
    pk["peak_va"] = pd.to_numeric(pk["peak_va"], errors="coerce")
    pk["peak_dt"] = pd.to_datetime(pk["peak_dt"], errors="coerce")
    pk["year"]    = pk["peak_dt"].dt.year
    pk = pk[pk["year"].between(min(years), max(years))].copy()

    site_nos = pk["site_no"].dropna().unique().tolist()
    print(f"  {len(pk):,} peak flow records across {len(site_nos):,} gauges.")

    # ── Step 2: Fetch site info (lat/lon) from USGS site service ─────────────
    # The site service accepts up to 100 site numbers per request
    print("  Fetching gauge coordinates from USGS site service …")
    site_rows = []
    chunk_size = 100
    for i in range(0, len(site_nos), chunk_size):
        chunk = site_nos[i : i + chunk_size]
        site_url = (
            "https://waterservices.usgs.gov/nwis/site/"
            "?format=rdb"
            f"&sites={','.join(chunk)}"
            "&siteOutput=expanded"
            "&siteStatus=all"
        )
        sr = requests.get(site_url, timeout=60)
        if sr.status_code != 200:
            continue
        slines = [l for l in sr.text.splitlines() if not l.startswith("#")]
        sheader = None
        for line in slines:
            parts = line.split("\t")
            if sheader is None and "site_no" in parts:
                sheader = parts
            elif sheader and len(parts) == len(sheader):
                if not all(p.strip().replace("s","").replace("d","").replace("n","").isdigit()
                           for p in parts):
                    site_rows.append(dict(zip(sheader, parts)))
        time.sleep(0.2)

    if not site_rows:
        print("  ⚠  Could not fetch site coordinates — skipping USGS layer.")
        return pd.DataFrame()

    sites = pd.DataFrame(site_rows)
    print(f"  Site info columns: {list(sites.columns)}")

    county_col = next(
        (c for c in ["county_cd", "county_code"] if c in sites.columns),
        None,
    )
    lat_col = next(
        (c for c in ["dec_lat_va", "dec_lat", "latitude", "lat_va"] if c in sites.columns),
        next((c for c in sites.columns if "lat" in c.lower()), None),
    )
    lon_col = next(
        (c for c in ["dec_long_va", "dec_long", "longitude", "lon_va", "long_va"] if c in sites.columns),
        next((c for c in sites.columns if "lon" in c.lower() or "lng" in c.lower()), None),
    )

    keep_cols = ["site_no"]
    if county_col:
        keep_cols.append(county_col)
    if lat_col:
        keep_cols.append(lat_col)
    if lon_col:
        keep_cols.append(lon_col)

    sites = sites[keep_cols].copy()

    if county_col:
        sites["GEOID"] = (
            STATE_FIPS +
            pd.to_numeric(sites[county_col], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.replace("<NA>", "", regex=False)
            .str.zfill(3)
        )
        sites.loc[~sites["GEOID"].str.fullmatch(r"\d{5}", na=False), "GEOID"] = pd.NA

    if lat_col and lon_col:
        sites["lat"] = pd.to_numeric(sites[lat_col], errors="coerce")
        sites["lon"] = pd.to_numeric(sites[lon_col], errors="coerce")
    else:
        sites["lat"] = np.nan
        sites["lon"] = np.nan

    sites = sites[["site_no", "GEOID", "lat", "lon"]].drop_duplicates()

    # ── Step 3: Join coordinates onto peak flow records ───────────────────────
    pk = pk.merge(sites, on="site_no", how="left")

    if pk["GEOID"].notna().any():
        pk = pk.dropna(subset=["GEOID", "peak_va"]).copy()
        print(f"  {len(pk):,} records mapped to counties from USGS site metadata.")
    else:
        pk = pk.dropna(subset=["lat", "lon", "peak_va"]).copy()
        print(f"  {len(pk):,} records with valid coordinates after join.")

    if pk.empty:
        print("  ⚠  No records remaining after site join — skipping.")
        return pd.DataFrame()

    if pk["GEOID"].isna().any():
        spatial = pk[pk["GEOID"].isna()].dropna(subset=["lat", "lon"]).copy()
        if not spatial.empty:
            pk_gdf = gpd.GeoDataFrame(
                spatial,
                geometry=gpd.points_from_xy(spatial["lon"], spatial["lat"]),
                crs=CRS_WGS,
            )
            counties = get_texas_counties().to_crs(CRS_WGS)
            joined = gpd.sjoin(
                pk_gdf,
                counties[["GEOID", "geometry"]],
                how="left",
                predicate="within",
            )
            pk.loc[joined.index, "GEOID"] = joined["GEOID_right"].values

    pk = pk.dropna(subset=["GEOID"]).copy()

    if pk.empty:
        print("  ⚠  Spatial join fallback returned no matches — skipping USGS layer.")
        return pd.DataFrame()

    # Flag events above each gauge's historical median
    medians = pk.groupby("site_no")["peak_va"].median().rename("median_q")
    pk = pk.join(medians, on="site_no")
    pk["flood_event"] = (pk["peak_va"] > pk["median_q"]).astype(int)

    pk_grp = pk.groupby(["GEOID","year"], as_index=False).agg(
        gauge_flood_events = ("flood_event", "sum"),
        peak_q_max         = ("peak_va",     "max"),
    )
    print(f"  → Peak flow data aggregated for {pk_grp['GEOID'].nunique()} counties.")
    return pk_grp

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD PANEL DATASET (county × year)
# ─────────────────────────────────────────────────────────────────────────────
def build_panel(counties_gdf, claims_agg, usgs_agg) -> gpd.GeoDataFrame:
    """Cross-join county × year, merge all data sources."""
    geoids = counties_gdf["GEOID"].unique()
    idx    = pd.MultiIndex.from_product([geoids, YEARS], names=["GEOID","year"])
    panel  = pd.DataFrame(index=idx).reset_index()

    panel = panel.merge(counties_gdf[["GEOID","COUNTY_NAME"]], on="GEOID", how="left")
    panel = panel.merge(claims_agg, on=["GEOID","year"], how="left")

    if not usgs_agg.empty:
        panel = panel.merge(usgs_agg, on=["GEOID","year"], how="left")
    else:
        panel["gauge_flood_events"] = np.nan
        panel["peak_q_max"]         = np.nan

    # Fill NaN → 0 for count/money columns
    for col in ["claim_count","total_payout","buildings_hit",
                "gauge_flood_events","peak_q_max"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    # Composite flood exposure score (normalised 0-1 per year)
    # = 0.5 * norm(claim_count) + 0.3 * norm(total_payout) + 0.2 * norm(gauge_flood_events)
    def norm_by_year(series_name: str) -> pd.Series:
        grouped = panel.groupby("year")[series_name]
        return grouped.transform(
            lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
        )

    panel["exposure_score"] = (
        0.5 * norm_by_year("claim_count") +
        0.3 * norm_by_year("total_payout") +
        0.2 * norm_by_year("gauge_flood_events")
    )

    # Re-attach geometry
    gdf = counties_gdf.merge(panel, on=["GEOID","COUNTY_NAME"], how="right")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS_WGS)
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 5. STATIC CHOROPLETH MAPS (one per year)
# ─────────────────────────────────────────────────────────────────────────────
CMAP_BASE = plt.get_cmap("YlOrRd")
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "exposure_scale",
    CMAP_BASE(np.linspace(0.18, 0.98, 256)),
)
PLOT_BG = "#ffffff"
PANEL_BG = "#f7f7f7"
TEXT_COLOR = "#1f1f1f"
EDGE_COLOR = "#5f5f5f"
EVENTS = {
    2019: "Tropical Storm Imelda\n(Sep 2019)",
    2020: "Hurricane Laura / Marco\nHisto-flooding",
    2021: "Winter Storm Uri\n(Feb 2021)",
    2022: "Central TX flooding\n(Jul 2022)",
    2023: "Panhandle Wildfires /\nFlash flooding events",
    2024: "Houston metro flooding\n(Apr-May 2024)",
}

def plot_year(gdf_year: gpd.GeoDataFrame, year: int, ax: plt.Axes,
              vmin=0.0, vmax=1.0):
    gdf_p = gdf_year.to_crs(CRS_PROJ)
    ax.set_facecolor(PANEL_BG)

    gdf_p.plot(
        column     = "exposure_score",
        ax         = ax,
        cmap       = CMAP,
        vmin       = vmin,
        vmax       = vmax,
        edgecolor  = EDGE_COLOR,
        linewidth  = 0.3,
        legend     = False,
        missing_kwds={"color":"#efefef","edgecolor":"#c7c7c7","label":"No data"},
    )
    # Basemap
    try:
        ctx.add_basemap(ax, crs=CRS_PROJ,
                        source=ctx.providers.CartoDB.PositronNoLabels,
                        alpha=0.22)
    except Exception:
        pass

    ax.set_title(str(year), fontsize=16, fontweight="bold",
                 color=TEXT_COLOR, pad=6)
    ax.axis("off")

    # Notable event annotation
    if year in EVENTS:
        ax.text(0.02, 0.03, EVENTS[year],
                transform=ax.transAxes,
                fontsize=7.5, color=TEXT_COLOR,
                va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#d0d0d0", alpha=0.92))


def make_grid_figure(gdf_panel: gpd.GeoDataFrame):
    vmin = gdf_panel["exposure_score"].quantile(0.01)
    vmax = gdf_panel["exposure_score"].quantile(0.99)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             facecolor=PLOT_BG)
    fig.suptitle("Texas Flood & Water Exposure  2019 – 2024",
                 fontsize=22, fontweight="bold", color=TEXT_COLOR, y=0.97)

    for ax, year in zip(axes.flat, YEARS):
        subset = gdf_panel[gdf_panel["year"] == year].copy()
        # fill counties with no data
        all_counties = gdf_panel[gdf_panel["year"] == YEARS[0]][["GEOID","geometry"]].copy()
        subset = all_counties.merge(
            subset.drop(columns="geometry"), on="GEOID", how="left")
        subset = gpd.GeoDataFrame(subset, geometry="geometry", crs=CRS_WGS)
        plot_year(subset, year, ax, vmin, vmax)

    # Shared colorbar
    sm  = cm.ScalarMappable(cmap=CMAP,
                             norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.025, pad=0.03, shrink=0.6)
    cbar.set_label("Composite Flood Exposure Score  (FEMA claims + USGS peak flow)",
                   color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cbar.outline.set_edgecolor("#bdbdbd")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = "outputs/texas_flood_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✅  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANIMATED GIF
# ─────────────────────────────────────────────────────────────────────────────
def make_animation(gdf_panel: gpd.GeoDataFrame):
    vmin = gdf_panel["exposure_score"].quantile(0.01)
    vmax = gdf_panel["exposure_score"].quantile(0.99)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=PLOT_BG)
    fig.suptitle("Texas Flood Exposure", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR)

    all_counties = (gdf_panel[gdf_panel["year"] == YEARS[0]]
                    [["GEOID","geometry"]].copy())

    def update(year):
        ax.clear()
        ax.set_facecolor(PLOT_BG)
        subset = gdf_panel[gdf_panel["year"] == year].copy()
        subset = all_counties.merge(
            subset.drop(columns="geometry"), on="GEOID", how="left")
        subset = gpd.GeoDataFrame(subset, geometry="geometry", crs=CRS_WGS)
        plot_year(subset, year, ax, vmin, vmax)

        sm = cm.ScalarMappable(cmap=CMAP,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        if not hasattr(ax, "_cbar_added"):
            cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                              fraction=0.04, pad=0.01, shrink=0.7)
            cb.set_label("Flood Exposure Score", color=TEXT_COLOR, fontsize=9)
            cb.ax.tick_params(colors=TEXT_COLOR)
            cb.outline.set_edgecolor("#bdbdbd")
            ax._cbar_added = True

    ani = FuncAnimation(fig, update, frames=YEARS,
                        interval=1200, repeat=True)
    path = "outputs/texas_flood_animated.gif"
    ani.save(path, writer=PillowWriter(fps=0.85))
    print(f"  ✅  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTERACTIVE FOLIUM MAP (slider by year)
# ─────────────────────────────────────────────────────────────────────────────
def make_folium_map(gdf_panel: gpd.GeoDataFrame):
    """Create an interactive choropleth with a year slider."""
    center = [31.0, -99.5]
    m      = folium.Map(location=center, zoom_start=6,
                        tiles="CartoDB positron")

    vmin = float(gdf_panel["exposure_score"].quantile(0.02))
    vmax = float(gdf_panel["exposure_score"].quantile(0.98))
    if vmax <= vmin:
        vmax = float(gdf_panel["exposure_score"].max() or 1.0)
    exposure_scale = bcm.LinearColormap(
        colors=[mcolors.to_hex(CMAP(i)) for i in np.linspace(0, 1, 6)],
        vmin=vmin,
        vmax=vmax,
        caption="Flood Exposure Score",
    )

    for year in YEARS:
        sub = gdf_panel[gdf_panel["year"] == year].copy()
        sub = sub.to_crs(CRS_WGS)
        sub["exposure_score"] = sub["exposure_score"].fillna(0)

        def style_fn(feat):
            score = feat["properties"].get("exposure_score", 0) or 0
            color = "#efefef" if score <= 0 else exposure_scale(score)
            return {
                "fillColor": color,
                "color": "#666666",
                "weight": 0.7,
                "fillOpacity": 0.88,
                "opacity": 0.75,
            }

        folium.GeoJson(
            sub.__geo_interface__,
            style_function = style_fn,
            highlight_function = lambda feat: {
                "weight": 1.5,
                "color": "#202020",
                "fillOpacity": 0.96,
            },
            tooltip        = folium.GeoJsonTooltip(
                fields     = ["COUNTY_NAME","year",
                              "claim_count","total_payout","exposure_score"],
                aliases    = ["County","Year","# Claims",
                              "Total Payout ($)","Exposure Score"],
                localize   = True,
                sticky     = True,
            ),
            name = str(year),
            show = (year == YEARS[-1]),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    exposure_scale.add_to(m)

    # Legend note
    legend_html = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:1000;
                background:#ffffff;color:#222;padding:10px 14px;
                border-radius:8px;font-size:12px;border:1px solid #cfcfcf;
                box-shadow:0 2px 8px rgba(0,0,0,0.12);">
      <b>Texas Flood Exposure 2019-2024</b><br>
      Toggle years in the layers panel →<br>
      Color encodes <b>exposure score</b>, not raw payout<br>
      Data: FEMA NFIP claims · USGS streamflow<br>
      <i style="color:#666">Click a county for details</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    path = "outputs/texas_flood_interactive.html"
    m.save(path)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TIME-SERIES PANEL for top-hit counties
# ─────────────────────────────────────────────────────────────────────────────
def make_timeseries(gdf_panel: gpd.GeoDataFrame):
    # make a copy so you don’t mutate the original data
    gdf_panel = gdf_panel.copy()

    # Clean + convert to numeric
    gdf_panel["total_payout"] = (
        gdf_panel["total_payout"]
        .astype(str)
        .str.replace(r"[,$]", "", regex=True)  # remove $ and commas
    )
    gdf_panel["total_payout"] = pd.to_numeric(
        gdf_panel["total_payout"], errors="coerce"
    )

    # Top 12 counties by total payout across all years
    top12 = (
        gdf_panel.groupby("COUNTY_NAME")["total_payout"]
        .sum()
        .nlargest(12)
        .index.tolist()
    )

    df_top = gdf_panel[gdf_panel["COUNTY_NAME"].isin(top12)].copy()
    
    fig, axes = plt.subplots(4, 3, figsize=(17, 13),
                              facecolor=PLOT_BG, sharey=False)
    fig.suptitle("Top 12 Texas Counties — NFIP Flood Payouts by Year",
                 fontsize=17, fontweight="bold", color=TEXT_COLOR, y=0.98)

    cmap_c = plt.get_cmap("plasma")
    colors = [cmap_c(i / 11) for i in range(12)]

    for ax, county, color in zip(axes.flat, top12, colors):
        df_c = (df_top[df_top["COUNTY_NAME"] == county]
                      .sort_values("year"))
        ax.fill_between(df_c["year"], df_c["total_payout"] / 1e6,
                        alpha=0.35, color=color)
        ax.plot(df_c["year"], df_c["total_payout"] / 1e6,
                color=color, linewidth=2.2, marker="o", markersize=5)

        ax.set_title(county, fontsize=10, color=TEXT_COLOR, pad=4)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cfcfcf")
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y)[-2:] for y in YEARS])
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x,_: f"${x:.0f}M"))

        # Mark notable event years
        for ey in [2021, 2024]:
            if ey in df_c["year"].values:
                ax.axvline(ey, color="#ffaa00", linewidth=1,
                           linestyle="--", alpha=0.7)

    # Hide unused subplot(s) if any
    for ax in axes.flat[len(top12):]:
        ax.set_visible(False)

    fig.text(0.5, 0.01, "Year   |   Dashed lines = major flood years (2021 Uri, 2024 Houston)",
             ha="center", color="#666", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = "outputs/texas_flood_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✅  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9. STATEWIDE TREND BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_statewide_bar(gdf_panel: gpd.GeoDataFrame):
    annual = gdf_panel.groupby("year", as_index=False).agg(
        total_payout  = ("total_payout","sum"),
        claim_count   = ("claim_count","sum"),
        counties_hit  = ("claim_count", lambda x: (x>0).sum()),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    facecolor=PLOT_BG)
    fig.suptitle("Texas Statewide Flood Metrics  2019–2024",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR)

    bar_colors = plt.get_cmap("YlOrRd")(
        np.linspace(0.3, 0.95, len(YEARS)))

    # Payout
    bars = ax1.bar(annual["year"], annual["total_payout"]/1e9,
                   color=bar_colors, edgecolor="#a0a0a0", linewidth=0.5)
    ax1.set_title("Total NFIP Payout ($ Billion)", color=TEXT_COLOR)
    ax1.set_facecolor(PANEL_BG)
    ax1.tick_params(colors=TEXT_COLOR)
    for s in ax1.spines.values(): s.set_edgecolor("#cfcfcf")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f"${x:.1f}B"))
    for bar, val in zip(bars, annual["total_payout"]/1e9):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"${val:.2f}B", ha="center", va="bottom",
                 color=TEXT_COLOR, fontsize=8)

    # Claim count
    ax2.bar(annual["year"], annual["claim_count"],
            color=bar_colors, edgecolor="#a0a0a0", linewidth=0.5)
    ax2.set_title("Total NFIP Claim Count", color=TEXT_COLOR)
    ax2.set_facecolor(PANEL_BG)
    ax2.tick_params(colors=TEXT_COLOR)
    for s in ax2.spines.values(): s.set_edgecolor("#cfcfcf")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))

    # Annotate notable events
    for ax in [ax1, ax2]:
        ax.axvline(2021, color="#4fc3f7", linewidth=1.5,
                   linestyle="--", alpha=0.8)
        ax.text(2021.1, ax.get_ylim()[1]*0.92,
                "Uri", color="#4fc3f7", fontsize=8)

    plt.tight_layout()
    path = "outputs/texas_flood_statewide.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✅  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Texas Flood Exposure Analysis  2019-2024")
    print("="*60 + "\n")

    # Step 1: Boundaries
    counties_gdf = get_texas_counties()

    # Step 2: FEMA claims
    try:
        df_claims   = fetch_fema_claims(YEARS)
        claims_agg  = aggregate_claims(df_claims)
    except Exception as e:
        print(f"  ⚠  FEMA fetch failed ({e}) – using empty data")
        claims_agg = pd.DataFrame(columns=["GEOID","year","claim_count",
                                            "total_payout","buildings_hit"])

    # Step 3: USGS peak flows
    try:
        usgs_agg = fetch_usgs_peak_flows(YEARS)
    except Exception as e:
        print(f"  ⚠  USGS fetch failed ({e}) – skipping peak flow layer")
        usgs_agg = pd.DataFrame()

    # Step 4: Panel
    print("\nBuilding county × year panel …")
    gdf_panel = build_panel(counties_gdf, claims_agg, usgs_agg)
    gdf_panel.drop(columns="geometry").to_csv(
        "outputs/texas_flood_summary.csv", index=False)
    print("  ✅  Panel saved → outputs/texas_flood_summary.csv")
    print(f"      Shape: {gdf_panel.shape}  |  "
          f"Counties: {gdf_panel['GEOID'].nunique()}  |  "
          f"Years: {sorted(gdf_panel['year'].unique())}")

    # Step 5: Visuals
    print("\nGenerating visualisations …")
    make_grid_figure(gdf_panel)
    make_timeseries(gdf_panel)
    make_statewide_bar(gdf_panel)
    make_folium_map(gdf_panel)

    print("\nGenerating animation (this takes ~30-60 s) …")
    make_animation(gdf_panel)

    print("\n" + "="*60)
    print("  All outputs written to ./outputs/")
    print("="*60)
