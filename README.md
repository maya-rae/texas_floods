# Texas Flood Exposure Analysis from 2019 to 2024
This project explores the counties in Texas most vulnerable to flooding, which is quantified using the formula  

$0.5 * \text{norm(claimcount)} + 0.3 * \text{norm(totalpayout)} + 0.2 * \text{norm(gaugefloodevents))}$

and called the `expoure_metric` normalized per year from 2019 to 2024 using data from OpenFEMA for insurance claims, USGS for the peak 
water flows, and Census TIGER for Texas county boundaries. 

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
* texas_flood_summary.csv          – cleaned merged dataset

