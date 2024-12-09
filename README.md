# Time Series Analysis for Energy Consumption Forecasting

This project uses the ARIMA model to forecast energy consumption based on historical data, weather patterns, and time-based features.

## Libraries Used
This analysis leverages several libraries for data manipulation, visualization, statistical testing, model building, and evaluation:
- **pandas**: For data manipulation and handling.
- **numpy**: For numerical operations.
- **matplotlib** and **seaborn**: For creating visualizations.
- **statsmodels**: For performing statistical tests and building the ARIMA model.
- **sklearn**: For model evaluation through error metrics.
- **pmdarima**: For automatically selecting the best ARIMA parameters.

## Data Overview
The dataset used in this project contains energy consumption data along with timestamps and weather-related features such as temperature and humidity. 
## Process

### 1. **Exploratory Data Analysis (EDA)**
The first step involves visualizing the energy consumption over time to understand any underlying patterns, trends, or seasonality in the data. This helps provide an initial understanding of the dataset and whether there are any visible fluctuations or outliers.

### 2. **Stationarity Check**
Since ARIMA models require the data to be stationary, statistical tests like the Augmented Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test are used to check the stationarity of the series. Non-stationary data needs to be transformed before modeling.

### 3. **Data Transformation**
If the data is found to be non-stationary, differencing is applied. This technique involves subtracting the previous value from the current one to stabilize the variance and make the series more stationary.

### 4. **Feature Engineering**
Time-based features such as the hour of the day, day of the week, and month are extracted from the timestamp to capture seasonality and cyclical patterns. These features are then added as exogenous variables to the model, helping improve forecast accuracy.

### 5. **Model Building (ARIMA)**
The dataset is split into a training set and a test set (typically 80% for training and 20% for testing). The ARIMA model is then built, and the parameters for the best model are determined using the `auto_arima` function, which considers seasonal patterns and exogenous variables. The selected parameters (p, d, q) are then used to fit the ARIMA model on the training data.

### 6. **Model Evaluation**
Once the model is trained, forecasts are made on the test set, and performance is evaluated using error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). These metrics help assess how well the model is predicting energy consumption.

### 7. **Visualization of Forecast vs Actual**
To better understand the model’s performance, visualizations are created that compare the actual energy consumption data with the forecasted values. This helps in visually assessing the accuracy and trends of the model’s predictions.

## Results
The performance of the ARIMA model is evaluated based on the RMSE and MAE values, where lower values indicate better predictive accuracy. The forecasted energy consumption is compared to actual consumption data, providing insights into the model's effectiveness.

