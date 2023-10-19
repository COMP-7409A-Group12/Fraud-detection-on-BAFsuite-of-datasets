import pandas as pd
# Load Base.csv
df = pd.read_csv('Variant.csv')

# Remove "device_fraud_count", it's 0 for all entries
print(df['device_fraud_count'].value_counts()) # It's 0 for all rows
df = df.drop(['device_fraud_count'], axis=1, errors='ignore')


'''用于将数据分为特征和目标变量。

首先，通过将'fraud_bool'列从DataFrame中删除，创建了一个名为X的新DataFrame，其中包含除了目标变量外的所有特征。

然后，创建了一个名为y的Series，其中只包含目标变量'fraud_bool'。

接下来，根据月份对数据进行了训练集和测试集的划分。月份0-5被划分为训练数据，而月份6-7被划分为测试数据。所以，X_train和y_train包含训练集数据，而X_test和y_test包含测试集数据。

最后，使用drop函数删除了X_train和X_test中的'month'列，因为该列已经不再需要作为特征变量。
'''

# Split data into features and target
X = df.drop(['fraud_bool'], axis=1)
y = df['fraud_bool']

# Train test split by 'month', month 0-5 are train, 6-7 are test data as proposed
X_train = X[X['month'] < 6]
X_test = X[X['month'] >= 6]
y_train = y[X['month'] < 6]
y_test = y[X['month'] >= 6]

X_train = X_train.drop('month', axis=1)
X_test = X_test.drop('month', axis=1)


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

"""首先，通过 X_train.dtypes == 'object' 判断每一列是否包含分类特征。这样得到一个布尔类型的 Series，其中索引是列名，值表示是否为分类特征。

接着，通过 list(s[s].index) 得到了所有包含分类特征的列的列表，并赋值给 object_cols。

然后，使用 OneHotEncoder 初始化一个独热编码器 ohe ，其中的参数 sparse=False 表示生成的编码结果不是稀疏矩阵，handle_unknown='ignore' 表示在测试集中出现未在训练集中出现的特征时忽略它。

接下来，分别用 ohe.fit_transform(X_train[object_cols]) 和 ohe.transform(X_test[object_cols]) 对训练集和测试集中的分类特征进行独热编码，分别得到 ohe_cols_train 和 ohe_cols_test，这里用 DataFrame 来存储编码结果。

再将 ohe_cols_train 和 ohe_cols_test 的索引设置为原始数据的索引，以便后续拼接回原始数据。

然后，从训练集和测试集中删除原始的分类特征列，得到只包含数值特征的 num_X_train 和 num_X_test。

最后，使用 pd.concat 将 num_X_train 和 ohe_cols_train 按列拼接，得到最终的训练集 X_train。同样地，将 num_X_test 和 ohe_cols_test 拼接为测试集 X_test。

为了确保新的特征列的名称与原始数据一致，通过 X_train.columns.astype(str) 将特征列的名字转换为字符串类型。"""


s = (X_train.dtypes == 'object') # list of column-names and wether they contain categorical features
object_cols = list(s[s].index) # All the columns containing these features
print(X[object_cols])

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore') # ignore any features in the test set that were not present in the training set

# Get one-hot-encoded columns
ohe_cols_train = pd.DataFrame(ohe.fit_transform(X_train[object_cols]))
ohe_cols_test = pd.DataFrame(ohe.transform(X_test[object_cols]))

# Set the index of the transformed data to match the original data
ohe_cols_train.index = X_train.index
ohe_cols_test.index = X_test.index

# Remove the object columns from the training and test data
num_X_train = X_train.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

# Concatenate the numerical data with the transformed categorical data
X_train = pd.concat([num_X_train, ohe_cols_train], axis=1)
X_test = pd.concat([num_X_test, ohe_cols_test], axis=1)

# Newer versions of sklearn require the column names to be strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

drop_cols = ['zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w']

X_train.drop(columns=drop_cols, inplace=True)
X_test.drop(columns=drop_cols, inplace=True)


# Feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#.........................................................................................
#Train and Test the model
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Build LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
model.add(LSTM(64))  # Stack Layer 2 LSTM
model.add(Dense(1, activation='sigmoid'))

#  Compilation Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)


#.........................................................................................
#Calculate the accuracy
from sklearn.metrics import accuracy_score

# Converts probabilities to category labels
threshold = 0.5  # set threshold
y_pred_labels = [1 if prob >= threshold else 0 for prob in y_pred]

# Calculation accuracy
accuracy = accuracy_score(y_test, y_pred_labels)

# Print accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))