# Texas Flood Exposure Analysis from 2019 to 2024
This project analyzes county-level damage from flooding in Texas, which is quantified using the formula  

$0.5 * \text{norm(claimcount)} + 0.3 * \text{norm(totalpayout)} + 0.2 * \text{norm(gaugefloodevents))}$

and called the `expoure_score.` It is normalized within a year first and uses data from OpenFEMA for insurance claims, USGS for a count of flagged flood events, and Census TIGER for Texas county boundaries. 

`claim_count` is weighted the most and represents the frequency of realized flood impacts. It recieves the largest weight because repeated claims are the clearest direct evidence that the floods are affecting both people and the property in that county.

`total_payout` represents the impact severity. It is weighted below claims frequency because the payouts can be skewed by a small number of very large losses or insurance coverage patterns.

`gauge_flood_events` represents the underlying physical flood hazard from hydrologic conditions. It is included, but recieves the smallest weight because gauge coverage is uneven across counties and are a less direct proxy for countywide damage.


##  Limitations of Weighting Scheme
This weighting scheme is not statistically estimated and the resulting score should be interpreted as a relative screening risk rather than a definitive measure of flood risk.

## Data Sources  
* FEMA OpenFEMA   – NFIP flood insurance claims and payout by county  
* USGS NWIS       – Streamgage peak-flow / flood stage data  
* Census TIGER    – Texas county boundaries (via pygris / geopandas)  
* NOAA CDO        – Monthly precipitation

## Intended Outputs
* texas_flood_grid.png             – static map separated by year
* texas_flood_animated.gif         – animated choropleth 2019-2024
* texas_flood_interactive.html     – folium interactive map
* texas_flood_timeseries.png       – county-level time-series panel
* texas_flood_statewide.png        - state-wide summary chart of damage in $
* texas_flood_summary.csv          – cleaned merged dataset

