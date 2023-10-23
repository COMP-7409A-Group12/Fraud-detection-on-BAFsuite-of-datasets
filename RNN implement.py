import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import database as db
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K

def f1_score(y_true, y_pred):
    y_pred_binary = K.round(y_pred)
    tp = K.sum(K.round(y_true * y_pred_binary))
    fp = K.sum(K.round(K.clip(y_pred_binary - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred_binary, 0, 1)))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1

def RNN_train(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics= [f1_score])

    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1, batch_size=32)

    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    y_pred_binary = np.round(y_pred)

    return y_pred

"""
def cross_validation(model, x, y, cv):
    skf = StratifiedKFold(n_splits=cv)
    f1_list = []

    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        f1 = RNN_train(X_train, y_train, X_test, y_test)
        f1_list.append(f1)

    return np.array(f1_list)
"""


if __name__ == "__main__":
    '''For testing purpose only'''
    X = db.data_lanudry(sample_portion=0.1)

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(RNN_train(X_train, y_train, X_test, y_test))


''' 
    # draw cross validation graph
    cv = 5
    f1_scores = cross_validation(model=None, x=X_train, y=y_train, cv=cv)
    print("F1 Scores:", f1_scores)

    plt.plot(range(1, cv+1), f1_scores, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('Cross Validation - F1 Score')
    plt.show()
'''
