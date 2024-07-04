import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df = pd.read_csv(r'C:\Elchanan\masters\second_year\Semester_B\IML\hackathon\HU.BER\train_bus_schedule.csv',
                 encoding="ISO-8859-8")
# Data Preprocessing
df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce')
df['arrival_time'].fillna(method='ffill', inplace=True)

# Create time-based features
df['hour'] = df['arrival_time'].dt.hour
df['minute'] = df['arrival_time'].dt.minute

# Sort by trip and station index
df.sort_values(by=['trip_id_unique', 'station_index'], inplace=True)

# Create lag features
df['lag_1'] = df.groupby('trip_id_unique')['passengers_up'].shift(1)
df['lag_2'] = df.groupby('trip_id_unique')['passengers_up'].shift(2)

# Fill NaN values in lag features
df.fillna(0, inplace=True)

# Select relevant features
features = ['hour', 'minute', 'lag_1', 'lag_2']
target = 'passengers_up'

# Train-test split based on trip_id_unique
unique_trips = df['trip_id_unique'].unique()
train_trips = unique_trips[:int(len(unique_trips) * 0.8)]
test_trips = unique_trips[int(len(unique_trips) * 0.8):]

train_df = df[df['trip_id_unique'].isin(train_trips)]
test_df = df[df['trip_id_unique'].isin(test_trips)]

# Prepare training and testing data
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, random_state=60)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Random Forest - MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# # Plot the actual vs predicted values for Random Forest
# plt.figure(figsize=(14, 7))
# plt.plot(y_test.values, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.legend()
# plt.xlabel('Sample')
# plt.ylabel('Passengers Up')
# plt.title('Actual vs Predicted Number of Passengers Boarding the Bus (Random Forest)')
# plt.show()
#
# # Optional: Time Series Forecasting with ARIMA
# # Prepare the data for ARIMA
# series = df.set_index('arrival_time')['passengers_up']
#
# # Fit the ARIMA model
# arima_model = ARIMA(series, order=(5, 1, 0))
# arima_model_fit = arima_model.fit()
#
# # Make future predictions
# forecast = arima_model_fit.forecast(steps=24)
#
# # Plot the forecast
# plt.figure(figsize=(14, 7))
# plt.plot(series.index, series, label='Actual')
# plt.plot(forecast.index, forecast, label='Forecast')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Passengers Up')
# plt.title('ARIMA Forecast')
# plt.show()
