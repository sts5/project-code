from flask import Flask, render_template
from pandas_datareader import data as pdr
import yfinance as yfin
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from dateutil.relativedelta import relativedelta
# Fetch stock data - SPY using yfinance
start_date = dt.datetime(2021, 1, 1)    # Define the date range
end_date = dt.datetime.now()
yfin.pdr_override()  # This overrides the behavior of pandas_datareader so that it fetches stock data using yfinance. This function is required because pandas_datareader does not directly support Yahoo Finance anymore, and yfinance must replace it.
spy = pdr.get_data_yahoo('SPY', start=start_date, end=end_date).reset_index() #This fetches historical stock data for the SPY from Yahoo Finance between the start and end dates. Then reset the index of the DataFrame to ensure that the date becomes a regular column instead of the index.
labe = spy['Date'].values     #Extract the date and close price data
values = spy['Close'].values
data = []
spy = pdr.get_data_yahoo('SPY', start=start_date, end=end_date)
for i in range(0, len(labe)):
    ts = pd.to_datetime(str(labe[i])) #cpnvert date to string and append to list
    d = ts.strftime('%Y-%m-%d')
    t = (d, values[i])
    data.append(t)
# The the list data now contains tuples where each tuple consists of a date (formatted as a string) and the corresponding closing price for that date.

scaler = MinMaxScaler(feature_range=(0, 1))
transformedData = scaler.fit_transform(spy['Close'].values.reshape(-1, 1))
testDays = 60

x_train = []
y_train = []
for i in range(testDays, len(transformedData)):
    x_train.append(transformedData[i - testDays:i, 0])
    y_train.append(transformedData[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build The Model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'cosine_proximity'], )
history = model.fit(x_train, y_train, epochs=25, batch_size=32)

test_end = dt.datetime.now()
test_start = test_end - relativedelta(months=13)

test_data = pdr.get_data_yahoo('SPY', start=test_start, end=test_end)
actual_data = test_data['Close'].values
total_dataset = pd.concat((spy['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - testDays:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)
x_test = []
for i in range(testDays, len(model_inputs)):
    x_test.append(model_inputs[i - testDays:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
predicted_data = pdr.get_data_yahoo('SPY', start=test_start, end=test_end).reset_index()
labe = predicted_data['Date'].values
pred_data_plot = []
for i in range(0, len(labe)):
    ts = pd.to_datetime(str(labe[i]))
    d = ts.strftime('%Y-%m-%d')
    t = (d, predictions[i][0])
    pred_data_plot.append(t)


next_data = [model_inputs[len(model_inputs) + 1 - testDays:len(model_inputs) + 1, 0]]
next_data = np.array(next_data)
next_data = np.reshape(next_data, (next_data.shape[0], next_data.shape[1], 1))
nextDayPrediction = model.predict(next_data)
nextDayPrediction = scaler.inverse_transform(nextDayPrediction)
nextDayPrediction = nextDayPrediction[0][0]

meanSquareError = []
for i in range(0, len(history.history['mse'])):
    meanSquareError.append({'x': str(i + 1), 'y': history.history['mse'][i]})


app = Flask(__name__)


@app.route("/")
def home():
    values = [row[1] for row in data]
    values = []

    meanAbsoluteError = []
    for i in range(0, len(history.history['mae'])):
        meanAbsoluteError.append({'x': str(i + 1), 'y': history.history['mae'][i]})

    cosine_proximity = []
    for i in range(0, len(history.history['cosine_proximity'])):
        cosine_proximity.append({'x': str(i + 1), 'y': history.history['cosine_proximity'][i]})

    for i in data:
        values.append({'x': i[0], 'y': i[1]})
    predicted = []
    for i in pred_data_plot:
        predicted.append({'x': i[0], 'y': i[1]})
    return render_template("chart.html", predicted=predicted, values=values, nextPrediction=nextDayPrediction, meanSquareError=meanSquareError, meanAbsoluteError=meanAbsoluteError, cosine=cosine_proximity)