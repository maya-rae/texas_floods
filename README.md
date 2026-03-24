# Texas Flood Exposure Analysis from 2019 to 2024
This project analyzes county-level damage from flooding in Texas, combining GIS spatial analysis, hydrological frequency modeling, and regression to quantify flood risk across Texas's 254 counties.

## Structure
* texas_flood_analysis.py - data pipeline and visualizations
* texa_flood_risk_model.py - frequency analysis and regression model
* outputs/ - all generated files are stored in this folder

## Data Sources
| Source | Data | Access |
|---|---|---|
| FEMA OpenFEMA API | NFIP flood insurance claims by county, 2019–2024 | Public REST API |
| USGS NWIS | Annual peak streamflow, all historical years | Public REST API |
| Census TIGER | County boundaries | pygris |
| NOAA CDO | Annual precipitation by station | Free token required |
| EPA Smart Location Database | Impervious surface % by county | Public download |
| Census ACS 5-year | Median household income by county | Public REST API |

## Methods

### Composite Exposure Score

The composite exposure score is calculated using this formula:  

$0.5 * \text{norm(claimcount)} + 0.3 * \text{norm(totalpayout)} + 0.2 * \text{norm(gaugefloodevents))}$

and stored in the variable `expoure_score.` It is first normalized within the year and uses data from OpenFEMA for insurance claims, USGS for a count of flagged flood events, and Census TIGER for Texas county boundaries. 

`claim_count` is weighted the most and represents the frequency of realized flood impacts. It recieves the largest weight because repeated claims are the clearest direct evidence that the floods are affecting both people and the property in that county.

`total_payout` represents the impact severity. It is weighted below claims frequency because the payouts can be skewed by a small number of very large losses or insurance coverage patterns.

`gauge_flood_events` represents the underlying physical flood hazard from hydrologic conditions. It is included, but recieves the smallest weight because gauge coverage is uneven across counties and are a less direct proxy for countywide damage.

This weighting scheme is not statistically estimated and the resulting score should be interpreted as a relative screening risk rather than a definitive measure of flood risk.

### Flood Frequency Analysis
Annual maximum peak flows per county are fitted to both a 
Generalised Extreme Value (GEV) distribution and a Log-Pearson III 
(LP3) distribution, which is the US federal standard per Bulletin 17C. 
The better-fitting model (by AIC) is used to estimate Q10, Q50, 
and Q100 return period flows: the flow magnitude with a 1-in-10, 
1-in-50, and 1-in-100 annual exceedance probability respectively.

### Regression Model
Target variable: log(NFIP claim count + 1) per county-year.
Features: precipitation anomaly, impervious surface %, median 
household income, log(Q10 return period flow), year trend.

Two models are compared:
* OLS with HC3 heteroscedasticity-robust standard errors (R² = 0.17)
* Random Forest with 5-fold cross-validation (CV R² = 0.52)

The OLS–RF gap suggests meaningful non-linear threshold effects 
and feature interactions, particularly between impervious surface 
and precipitation anomaly.

### Known Limitations
* Precipitation anomaly is statewide, not county-level — the 
  largest single driver of the low OLS R²
* Impervious surface and income are static (2021 vintage), not 
  time-varying
* No event-level storm footprints; all aggregation is at county-year

## Intended Outputs
* texas_flood_grid.png             – static map separated by year
* texas_flood_animated.gif         – animated choropleth 2019-2024
* texas_flood_interactive.html     – folium interactive map
* texas_flood_timeseries.png       – county-level time-series panel
* texas_flood_statewide.png        - state-wide summary chart of damage in $
* texas_flood_summary.csv          – cleaned merged dataset

## Resources
This project draws on many resesarch papers working on flood risk assessment. 

* Del-Rosal-Salido et al. (2025), “A composite index framework for compound flood risk assessment”
* Journal of Hydrology (2023), “Flood risk assessment using an indicator based approach combined with flood risk maps and grid data”
* Lee and Choi (2018), “Comparison of Flood Vulnerability Assessments to Climate Change by Construction Frameworks for a Composite Indicator”
* 
