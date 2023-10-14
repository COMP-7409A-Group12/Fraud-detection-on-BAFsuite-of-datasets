import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt

data = pd.read_csv('zgpa_train.csv')
price = data.loc[:, 'close']
price_norm = price / max(price)

time_step = 8

def extract_data(data, time_step):
    x = []
    y = []
    for i in range(len(data) - time_step):
        x.append([a for a in data[i:i+time_step]])
        y.append(data[i+time_step])
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = np.array(y)
    return x, y

x, y = extract_data(price_norm, time_step)

model = Sequential()
model.add(LSTM(units=5, input_shape=(time_step, 1), activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.fit(x, y, batch_size=30, epochs=200)

data_test = pd.read_csv('zgpa_test.csv')
price_test = data_test.loc[:, 'close']
price_test_norm = price_test / max(price)

x_test_norm, y_test_norm = extract_data(price_test_norm, time_step)

y_test_predict = model.predict(x_test_norm) * max(price)
y_test = y_test_norm * max(price)

fig = plt.figure(figsize=(8, 5))
plt.plot(y_test, label='real test price')
plt.plot(y_test_predict, label='predict test price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()