import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# Function to create dataset by grouping past 100 days of data together
def create_dataset(data, time_step=1):
  X, y = [], []

  for i in range(100, data.shape[0]):
    X.append(data[i-time_step:i])
    y.append(data[i, 0])
  
  return np.array(X), np.array(y)


# Gather and Display Data from Yahoo Finance
st.title("Stock Price Trend Predictor App")

ticker = st.text_input("Enter the ticker symbol", "MSFT")

end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
data = yf.download(ticker, start, end)

st.subheader("Stock Data")
st.write(data)

# Plot closing price for a Stock over past 10 years
st.subheader(f"Closing Price for {ticker}'s stock over past 10 years")
fig1 = plt.figure(figsize=(12, 5))
plt.plot(data.index, data["Close"])
plt.xlabel("Years")
plt.ylabel("Price")
plt.title(f"Closing Price for {ticker}'s stock", weight="bold")
plt.legend(["Close"], loc="upper left")
st.pyplot(fig1)

# Plot closing price vs Moving Average of 100 and 200 days
st.subheader(f"Closing Price vs Moving Average of 100 and 200 days for {ticker}'s stock over last 10 years")
data["MA_100_days"] = data["Close"].rolling(window=100).mean()
data["MA_200_days"] = data["Close"].rolling(window=200).mean()
fig2 = plt.figure(figsize=(12, 5))
plt.plot(data.index, data["MA_100_days"], 'r')
plt.plot(data.index, data["MA_200_days"], 'g')
plt.plot(data.index, data["Close"])
plt.xlabel("Years")
plt.ylabel("Price")
plt.title(f"Closing Price vs Moving Average of 100 and 200 days for {ticker}'s stock", weight="bold")
plt.legend(["MA_100_days", "MA_200_days", "Close"], loc="upper left")
st.pyplot(fig2)

# Plot percentage change in closing price
st.subheader(f"Percentage Change in Closing Price for {ticker}'s stock over last 10 years")
data["percentage_change_close"] = data["Close"].pct_change()*100
fig3 = plt.figure(figsize=(12, 5))
plt.plot(data.index, data["percentage_change_close"])
plt.xlabel("Years")
plt.ylabel("Percentage (%)")
plt.title(f"Percentage Change in Closing Price for {ticker}'s stock", weight="bold")
plt.legend(["Percentage Change"], loc="lower right")
st.pyplot(fig3)

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Create and Reshape Data
time_step = 100
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)  # reshape input to the format expected by the LSTM model

# Divide data into training (80%) and testing (20%)
split_len = int(len(X) * 0.8)
X_train, X_test = X[:split_len], X[split_len:]
y_train, y_test = y[:split_len], y[split_len:]

# Load the saved model
model = load_model("Stock_Price_Trend_Predictor.keras")

# Get the predictions from loaded model
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Inverse transforma and reshape predicted data
final_predictions_train = scaler.inverse_transform(predictions_train.reshape(-1, 1))
final_y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

final_predictions_test = scaler.inverse_transform(predictions_test.reshape(-1 ,1))
final_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Actual vs Predicted closing price (Training Data)
st.subheader(f"Actual vs Predicted Closing Price for {ticker}'s stock (Training Data)")
fig4 = plt.figure(figsize=(12, 5))
plt.plot(data.index[:split_len], final_predictions_train, 'r')
plt.plot(data.index[:split_len], final_y_train, 'g')
plt.xlabel("Years")
plt.ylabel("Price")
plt.title(f"Actual vs Predicted Closing Price for {ticker}'s stock (Training Data)", weight="bold")
plt.legend(["Predicted Closing Price", "Actual Closing Price"], loc="upper left")
st.pyplot(fig4)

# Plot Actual vs Predicted closing price (Testing Data)
st.subheader(f"Actual vs Predicted Closing Price for {ticker}'s stock (Test Data)")
fig5 = plt.figure(figsize=(12, 5))
plt.plot(data.index[split_len+time_step:], final_predictions_test, 'r')
plt.plot(data.index[split_len+time_step:], final_y_test, 'g')
plt.xlabel("Years")
plt.ylabel("Price")
plt.title(f"Actual vs Predicted Closing Price for {ticker}'s stock (Test Data)", weight="bold")
plt.legend(["Predicted Closing Price", "Actual Closing Price"], loc="upper left")
st.pyplot(fig5)

# Calculate and Diaplay the Accuracy Matrix
mape_train = mean_absolute_percentage_error(final_y_train, final_predictions_train) * 100
mape_test = mean_absolute_percentage_error(final_y_test, final_predictions_test) * 100

st.write(f"Mean Absolute Percentage Error (MAPE) for Training Data: {mape_train:.2f}%")
st.write(f"Mean Absolute Percentage Error (MAPE) for Testing Data: {mape_test:.2f}%")

# Predict next day's closing price
last_100_days = scaled_data[-time_step:]
last_100_days = last_100_days.reshape(1, time_step, 1)

next_day_prediction = model.predict(last_100_days)
next_day_prediction = scaler.inverse_transform(next_day_prediction)

last_closing_price = data["Close"].values[-1]
if next_day_prediction[0][0] > last_closing_price:
  trend = f"The {ticker}'s closing price is predicted to go up on the next trading day."
elif next_day_prediction[0][0] < last_closing_price:
  trend = f"The {ticker}'s closing price is predicted to go down on the next trading day."
else:
  trend = f"The {ticker}'s closing price is predicted to remain the same on the next trading day."

st.write("***")
st.subheader(f"{trend}")
st.subheader(f"Next trading day's estimated closing price for {ticker}: {next_day_prediction[0][0]: 0.2f}")
st.write("***")

# Disclaimer
st.subheader("Disclaimer")
st.write("The predictions provided by this app are for educational purposes only and should not be considered as financial advice. Please consult with a financial advisor before making any investment decisions.")