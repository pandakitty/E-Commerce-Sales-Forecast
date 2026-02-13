# 📈 E-Commerce Sales Forecasting (Python / Time Series)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview
This project develops a robust **Time Series Forecasting** model to predict daily retail sales for an e-commerce platform. By leveraging the **Facebook Prophet** library, the model identifies seasonal patterns, holiday effects, and long-term trends to provide actionable insights for inventory management, staffing, and marketing budget allocation.

## ✨ Key Features & Technical Details
* **Time Series Decomposition**: Analyzed historical data to isolate weekly, monthly, and yearly seasonality trends.
* **Prophet Modeling**: Implemented an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
* **Holiday Effect Integration**: Incorporated a custom "Holidays" regressor to account for significant sales spikes during Black Friday, Cyber Monday, and the December holiday season.
* **Evaluation Metrics**: Assessed model performance using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)** to ensure forecasting reliability.
* **Forecast Visualization**: Generated trend components and 365-day future projections with uncertainty intervals (confidence bands).

## 🚀 Business Impact
The final model provides a data-driven roadmap for operational efficiency:
* **Inventory Optimization**: Reduced the risk of "stock-outs" during peak seasons by predicting volume surges.
* **Marketing Efficiency**: Identified the highest-growth windows to maximize Ad Spend
