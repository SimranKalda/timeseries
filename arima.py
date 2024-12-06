# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
from pmdarima import auto_arima


data = pd.read_csv('energy_consumption_weather.csv', parse_dates=['timestamp'], index_col='timestamp')

# Exploratory Data Analysis (EDA)
# Plotting the energy consumption over time to visualize the patterns
plt.figure(figsize=(12, 6))
plt.plot(data['consumption'], label='Energy Consumption')
plt.title('Energy Consumption Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.show()

# Check for Stationarity
# Using adf and kpss to check if the data is stationary.

adf_result = adfuller(data['consumption'])
print(f'ADF Test p-value: {adf_result[1]}')

kpss_result = kpss(data['consumption'], regression='c')
print(f'KPSS Test p-value: {kpss_result[1]}')

# Data Transformation
# data is non-stationary applying differencing stabilize the variance

# appying first-order differencing to make the series more stationary
data['diff'] = data['consumption'].diff().dropna()

# Plot the differenced data
plt.figure(figsize=(12, 6))
plt.plot(data['diff'], label='Differenced Energy Consumption')
plt.title('Differenced Energy Consumption')
plt.xlabel('Timestamp')
plt.ylabel('Differenced Consumption (kWh)')
plt.legend()
plt.show()

# Feature Engineering
data['hour'] = data.index.hour  
data['day_of_week'] = data.index.dayofweek 
data['month'] = data.index.month  

# building model ARIMA
train_size = int(len(data) * 0.8)  
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Using auto_arima to automatically find the best ARIMA parameters (p, d, q)
auto_model = auto_arima(train['consumption'], seasonal=True, m=24, 
                        exogenous=train[['hour', 'day_of_week', 'month', 'temperature', 'humidity']], 
                        trace=True, error_action='ignore', suppress_warnings=True)

print(f'Best ARIMA Model: {auto_model.summary()}')

# Fitting the ARIMA model with the best parameters
model = ARIMA(train['consumption'], exogenous=train[['hour', 'day_of_week', 'month', 'temperature', 'humidity']], 
              order=auto_model.order)  # Use ARIMA orders from auto_arima
model_fit = model.fit()

# Model Performance: Evaluating the Forecast
forecast_steps = len(test)  # Number of steps to forecast (equal to the test set length)
forecast = model_fit.forecast(steps=forecast_steps, exogenous=test[['hour', 'day_of_week', 'month', 'temperature', 'humidity']])

# Evaluating the performance of the model by calculating error metrics like RMSE and MAE 
rmse_value = rmse(test['consumption'], forecast)
mae_value = mean_absolute_error(test['consumption'], forecast)
print(f'RMSE: {rmse_value}')
print(f'MAE: {mae_value}')

# Visualizing the Forecast vs Actual Data

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['consumption'], label='Train')  # Plot the training data
plt.plot(test.index, test['consumption'], label='Test')  # Plot the actual test data
plt.plot(test.index, forecast, label='Forecast')  # Plot the forecasted data
plt.legend()
plt.title('Energy Consumption Forecast')
plt.xlabel('Timestamp')
plt.ylabel('Consumption (kWh)')
plt.show()
