import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
stock = "GOOG"
google_data = yf.download(stock, start, end)
#print(google_data.head())
#print(google_data.shape)
#print(google_data.describe())
#print(google_data.info())
#print(google_data.isna().sum())

"""
plt.figure(figsize = (15,5))
google_data['Adj Close'].plot()
plt.xlabel("years")
plt.ylabel("Adj Close")
plt.title("Closing price of Google data")
plt.show()
"""

def plot_graph(figsize, values, column_name):
    plt.figure()
    values.plot(figsize = figsize)
    plt.xlabel("years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of Google data")
    plt.show()

#print(google_data.columns)

#for column in google_data.columns:
#    plot_graph((15,5),google_data[column], column)

temp_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#print(sum(temp_data[1:6])/5)

data = pd.DataFrame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#print(data.head())

data['MA'] = data.rolling(5).mean()
#print(data)

#for i in range(2004,2025):
#    print(i,list(google_data.index.year).count(i))

google_data['MA_for_250_days'] = google_data['Adj Close'].rolling(250).mean()
#print(google_data['MA_for_250_days'][0:250].tail())

#plot_graph((15,5), google_data['MA_for_250_days'], 'MA_for_250_days')

#plot_graph((15,5), google_data[['Adj Close','MA_for_250_days']], 'MA_for_250_days')

google_data['MA_for_100_days'] = google_data['Adj Close'].rolling(100).mean()
#plot_graph((15,5), google_data[['Adj Close','MA_for_100_days']], 'MA_for_100_days')

#plot_graph((15,5), google_data[['Adj Close','MA_for_100_days', 'MA_for_250_days']], 'MA')

google_data['percentage_change_cp'] = google_data['Adj Close'].pct_change()
#print(google_data[['Adj Close','percentage_change_cp']].head())

#plot_graph((15,5), google_data['percentage_change_cp'], 'percentage_change')

Adj_close_price = google_data[['Adj Close']]
#print(max(Adj_close_price.values),min(Adj_close_price.values))




scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Adj_close_price)
#print(scaled_data)
#print(len(scaled_data))

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
    

x_data, y_data = np.array(x_data), np.array(y_data)
#print(x_data[0],y_data[0])

#print(int(len(x_data)*0.7))
#print(5028-100-int(len(x_data)*0.7))

splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs = 2)

model.summary()
predictions = model.predict(x_test)

#print(predictions)

inv_predictions = scaler.inverse_transform(predictions)
print(inv_predictions)

inv_y_test = scaler.inverse_transform(y_test)
print(inv_y_test)

rmse = np.sqrt(np.mean( (inv_predictions - inv_y_test)**2))
print("RMSE=",rmse)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
print(ploting_data.head())

plot_graph((15,6), ploting_data, 'test data')
plot_graph((15,6), pd.concat([Adj_close_price[:splitting_len+100],ploting_data], axis=0), 'whole data')

model.save("Latest_stock_price_model.keras")