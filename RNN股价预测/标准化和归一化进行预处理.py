import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt

data = pd.read_csv('zgpa_train.csv')
price = data.loc[:, 'close']

# 使用标准化进行特征缩放
mean = np.mean(price)
std_dev = np.std(price)
price_norm = (price - mean) / std_dev

# # 使用归一化进行特征缩放
# min_value = np.min(price)
# max_value = np.max(price)
# price_norm = (price - min_value) / (max_value - min_value)

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

# 使用标准化进行特征缩放
price_test_norm = (price_test - mean) / std_dev

# # 使用归一化进行特征缩放
# price_test_norm = (price_test - min_value) / (max_value - min_value)

x_test_norm, y_test_norm = extract_data(price_test_norm, time_step)

y_test_predict = model.predict(x_test_norm) * std_dev + mean
y_test = y_test_norm * std_dev + mean

fig = plt.figure(figsize=(8, 5))
plt.plot(y_test, label='real test price')
plt.plot(y_test_predict, label='predict test price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()