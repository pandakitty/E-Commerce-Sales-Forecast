import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# --- 1. Data Preparation (Synthetic Data Generation) ---

# Generate daily dates for one year
start_date = '2024-01-01'
end_date = '2024-12-31'
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate synthetic sales data
# Base sales with a general upward trend
base_sales = np.linspace(100, 150, len(dates)) 
# Add weekly seasonality (simulating weekend spikes)
weekly_seasonality = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 10 
# Add monthly seasonality
monthly_seasonality = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 15
# Add random noise
noise = np.random.normal(0, 5, len(dates))

# Combine to create the final sales figures
sales = base_sales + weekly_seasonality + monthly_seasonality + noise
sales = np.maximum(0, sales).round(2) # Ensure no negative sales

# Create the DataFrame required by Prophet (columns must be named 'ds' and 'y')
data = pd.DataFrame({'ds': dates, 'y': sales})

print("--- Data Head (Past Year) ---")
print(data.head())
print("\n--- Model Training Start ---")

# --- 2. Model Fitting and Forecasting ---

# Initialize the Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Fit the model to the historical data
model.fit(data)

# Create a DataFrame for future predictions (90 days ahead)
future = model.make_future_dataframe(periods=90)

# Generate the forecast
# 'yhat' is the predicted value
# 'yhat_lower' and 'yhat_upper' are the uncertainty intervals
forecast = model.predict(future)

# --- 3. Visualization and Results ---

print("--- Forecast Head (Next 90 Days) ---")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
# The model.plot() function is a built-in convenience feature
fig = model.plot(forecast)
plt.title('E-Commerce Sales Forecast (90 Days Ahead)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Sales Volume (USD)')

# Save the plot for your GitHub README!
plt.savefig('sales_forecast_plot.png', dpi=300)
plt.show()

print("\nSuccessfully generated forecast and saved plot to sales_forecast_plot.png")