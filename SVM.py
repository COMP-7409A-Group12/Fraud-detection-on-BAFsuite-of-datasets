from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import plot

import database as db

# train SVM  //rbf>linear>poly>sigmoid
def SVM_train(X_train, y_train, X_test, y_test,kernel:str):
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)



    y_pred = svm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    FPR = fp / (tn + fp)
    error_rate = (fp+fn) / (tn + fp + fn + tp)
    # return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
    #     y_test, y_pred)
    return FPR,error_rate


#
def svm_fit():
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

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    FPR ,error= [],[]


    for kernel in kernels:
        fpr,svm_error = SVM_train(X_train, y_train, X_test, y_test, kernel=kernel)
        print(fpr,svm_error)
        FPR.append(fpr)
        error.append(svm_error)
    kernel_numlist = [i + 1 for i in range(len(FPR))]
    plot.draw(kernel_numlist, FPR, 'SVM', 'kernel', 'FPR')
    plot.draw(kernel_numlist, error, 'SVM', 'kernel', 'FPR')

    return FPR,error
    #accuracy, precision, recall, f1_score = SVM_train(X_train, y_train, X_test, y_test)

    # print(
    #     "accuracy:" + str(accuracy) + "\nprecision:" + str(precision) + "\nrecall:" + str(recall) + "\nf1_score:" + str(
    #         f1_score))





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
    print(SVM_train(X_train, y_train, X_test, y_test,'linear'))




