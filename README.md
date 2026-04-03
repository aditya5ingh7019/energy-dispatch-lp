# Energy Dispatch Optimization — Linear Programming

A 24-hour energy dispatch optimizer using Linear Programming (PuLP) 
to minimize grid cost and carbon emissions across solar, wind, 
battery storage, and grid sources.

## Problem
Given hourly demand, solar, and wind availability, determine the 
optimal dispatch schedule for a battery storage system to minimize 
combined grid cost and emission penalties under time-of-use pricing.

## Data Sources
| Variable | Source | Details |
|----------|--------|---------|
| Demand | Kaggle — Hourly Load India (Shubham Vashisht) | Northern Region, 26-Mar-2024; underlying source: POSOCO. [Dataset link](https://www.kaggle.com/datasets/shubhamvashisht/hourly-load-india-electrical-load-forecasting) |
| Solar | Modeled (sinusoidal, lat 28°N) | Standard irradiance curve, CEA/MNRE documentation |
| Wind | Synthetic uniform 20–50 kWh/hr | Realistic NR microgrid range |
| Grid cost | DERC Time-of-Use tariff | ₹6/8/10/12 per kWh by time block |
| Emission factor | CEA CO₂ Baseline 2023-24, NR | Range 0.70–0.95 kg CO₂/kWh |

## Results
| Metric | No-Storage Baseline | LP Optimized | Change |
|--------|---------------------|--------------|--------|
| Total Grid Purchased (kWh) | 1807.3 | 1843.0 | +2.0% |
| Grid Cost (₹) | 16,883 | 16,032 | −5.0% |
| **Peak-Hour Grid (hrs 18–23)** | **621.0 kWh** | **441.0 kWh** | **−29.0%** |
| Combined Cost+Emissions (₹) | 28,653 | 27,712 | −3.3% |

The optimizer pre-charges the battery during cheap off-peak hours 
(₹6–8/kWh) and discharges during expensive peak hours (₹12/kWh), 
achieving 29% peak-hour grid reduction.

## Setup
pip install pulp numpy pandas matplotlib openpyxl kagglehub

## Data
Download the dataset automatically via Kaggle API:
import kagglehub
path = kagglehub.dataset_download("shubhamvashisht/hourly-load-india-electrical-load-forecasting")

Or download manually from:
https://www.kaggle.com/datasets/shubhamvashisht/hourly-load-india-electrical-load-forecasting

## Run
python energy_dispatch.py

> Note: Replace the Excel file path in the script with your local 
> path to hourlyLoadDataIndia.xlsx (POSOCO dataset).

## Battery Parameters
- Capacity: 200 kWh
- Max charge/discharge rate: 50 kWh/hr
- Efficiency: 95%
- Minimum end-of-day SOC: 20 kWh
