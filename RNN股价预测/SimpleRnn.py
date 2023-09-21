#任务：基于zgpa_train.csv数据，建立RNN模型，预测股价
#1.完成数据预处理，将序列数据转化为可用于RNN输入的数据
#2.对新数据zgpa_test.csv进行预测，可视化结果
#3.存储预测结果，并观察局部预测结果
#备注：模型结构：单层RNN，输出有五个神经元：每次使用千把个数据预测第九个数据

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from matplotlib import pyplot as plt

data = pd.read_csv('zgpa_train.csv')
#显示文件的前几行
# print(data.head())
price = data.loc[:,'close'] #data.loc[row_label, column_label] 选择close这一列的内容
# price.head()
price_norm = price/max(price) #除以最大的数，让所有数值在[0,1]范围内
print(price_norm)

#----------可视化----------

fig1 = plt.figure(figsize = (8,5)) # 创建一个具有指定大小的图形对象
plt.plot(price_norm) #归一化之后的样子，若要归一化之前的样子则(price)
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()

#---------数据预处理------------

#define x,y
#difine method to extract X and Y 让他适合RNN输入序列
#0,1,2,3,4,5,6,7,8,9:10个样本; time_step = 8; 0,1,2,3,4,5,6,7(第一组); 1,2,3,4,5,6,7,8(第二组);
#2,3,4,5,6,7,8,9(没有需要预测的数据，因此不算做一组样本); 因此一共只有2组样本
#y是目标值，是一维数组，但是x需要进行转化
def extract_data(data,time_step):
    x = []
    y = []
    for i in range(len(data) - time_step): #731 - 8 = 723 一共两组样本
        x.append([a for a in data[i:i+time_step]])
        y.append(data[i+time_step])
    x = np.array(x) #将x转换为数组
    x = x.reshape(x.shape[0],x.shape[1],1)   # x有三个维度：样本数, time_step, 数组本身的维度
    y = np.array(y)
    return x,y

time_step = 8
x,y = extract_data(price_norm,time_step)
# print(x.shape)
#打印第一个样本，选择了第一个样本的所有时间步和所有特征维度的值。
#print(x[0,:])
#print(y)

#---------Set up model------------

model = Sequential() #实例
# add RNN layer
model.add(SimpleRNN(units = 5, input_shape = (time_step, 1),activation = 'relu'))
# add output layer (output:y)
model.add(Dense(units = 1, activation = 'linear')) # 输出的维度为1; 线性激活函数就是 f(x) = x
#config model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

#train model
model.fit(x,y,batch_size = 30,epochs = 200)

#----------可视化预测结果-----------

# predict on training data
y_train_predict = model.predict(x) * max(price) #之前x归一化过，则需要反归一化
y_train = y * max(price)
print(y_train_predict) #y的预测值
print(y_train) #y在训练集中的真实值

fig2 = plt.figure(figsize = (8,5))
plt.plot(y_train, label = 'real price')
plt.plot(y_train_predict, label ='predict price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

#预测测试数据
data_test = pd.read_csv('zgpa_test.csv')
# data_test.head()
price_test = data_test.loc[:,'close']
# price_test.head()

#Extract x_test, y_test
price_test_norm = price_test/max(price) #与之前用同样的归一化方法，防止预测不准
x_test_norm, y_test_norm = extract_data(price_test_norm, time_step)
#打印维度
# print(x_test_norm.shape,len(y_test_norm))

# predict on test data
y_test_predict = model.predict(x_test_norm)*max(price)
y_test = y_test_norm*max(price)
fig3 = plt.figure(figsize = (8,5))
plt.plot(y_test, label = 'real test price')
plt.plot(y_test_predict, label ='predict test price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

#----------可视化预测结果-----------

#将两个数组 y_test 和 y_test_predict 沿着轴 Axis=1 进行拼接。
result_y_test = y_test.reshape(-1,1) #-1自动计算行数
result_y_test_predict = y_test_predict
result = np.concatenate((result_y_test,result_y_test_predict),axis = 1)
result = pd.DataFrame(result,columns = ['real test price','predict test price'])
result.to_csv('zgpa_predict_test.csv')