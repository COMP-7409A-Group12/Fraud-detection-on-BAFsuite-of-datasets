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

import numpy as np

def RNN_train(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=10, batch_size=32)

    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    y_pred_binary = np.round(y_pred).astype(int)

    f1 = f1_score(y_test, y_pred_binary)
    return f1


def cross_validation(model, x, y, cv):
    skf = StratifiedKFold(n_splits=cv)
    f1_list = []

    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # scaler会把index也scale掉，所以一定是分完data再scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        f1 = RNN_train(X_train, y_train, X_test, y_test)
        f1_list.append(f1)

    return f1_list



if __name__ == "__main__":
    '''For testing purpose only'''
    X = db.data_lanudry(sample_portion=0.01)

    ### 你可以用一小簇数据来快速debug
    # X.sample(frac=0.001)
    #####################################

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    # print(RNN_train(X_train, y_train, X_test, y_test))

    # draw cross validation graph
    cv = 5
    f1_scores = cross_validation(model=None, x=X_train, y=y_train, cv=cv)
    f1_scores = np.array(f1_scores)
    print(f1_scores)

    plt.plot(range(1, cv + 1), f1_scores, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('Cross Validation - F1 Score')
    plt.show()