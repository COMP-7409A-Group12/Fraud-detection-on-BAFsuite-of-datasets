import pandas
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import plot
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

import database as db


# train SVM  //rbf>linear>poly>sigmoid
def SVM_train(X_train, y_train, X_test, y_test, C,gamma,kernel: str,cv):
    svm = SVC(C=C,gamma=gamma,kernel=kernel)
    cross_list = []
    cross_list,cross_average = cross_validation(svm, X_train, y_train, cv, 'f1', cross_list)
    cross_numlist = [i + 1 for i in range(len(cross_list[0]))]
    plot.draw(cross_numlist, cross_list[0], cross_numlist, 'SVM', f'{kernel}_fold', 'cross_result_f1')

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    FPR = fp / (tn + fp)
    error_rate = (fp + fn) / (tn + fp + fn + tp)
    # return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
    #     y_test, y_pred)
    return FPR, error_rate, cross_list,cross_average


#
def svm_fit(X: pandas.DataFrame, portion, cv):
    # X_train, y_train = shuffle(X_train, y_train)



    X_train, X_test = train_test_split(X, test_size=1-portion, train_size=portion)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    gammas = [10,1,0.1,0.01,0.001]
    FPR, f1s = [], []


    # for kernel in kernels:
    #     param_grid = {'C': [0.1], 'gamma': [1], 'kernel': [kernel]}
    #     svc = svm.SVC()
    #     # 创建GridSearchCV实例
    #     grid = GridSearchCV(svc, param_grid, refit=True, verbose=2)
    #     # 使用训练数据拟合GridSearchCV
    #     grid.fit(X_train, y_train)
    #     # 打印最优的参数组合
    #     print(grid.best_params_)
    #
    #
    #     fpr, svm_error, cross_result,f1 = SVM_train(X_train, y_train, X_test, y_test, kernel=kernel,cv=cv)
    #     #print(fpr, svm_error)
    #     FPR.append(fpr)
    #     f1s.append(f1)
    param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    svc = svm.SVC()
    # 创建GridSearchCV实例
    grid = GridSearchCV(svc, param_grid, refit=True, verbose=2)
    # 使用训练数据拟合GridSearchCV
    grid.fit(X_train, y_train)
    # 打印最优的参数组合
    print(grid.best_params_)

    fpr, svm_error, cross_result, f1 = SVM_train(X_train, y_train, X_test, y_test,C=grid.best_params_['C'],gamma=grid.best_params_['gamma'],kernel=grid.best_params_['kernel'], cv=cv)
    #print(fpr, svm_error)
    # FPR.append(fpr)
    # f1s.append(f1)

    # kernel_numlist = [i + 1 for i in range(len(FPR))]
    # plot.draw(kernel_numlist, FPR, kernels, 'SVM', 'kernel', 'FPR')
    # plot.draw(kernel_numlist, f1s, kernels, 'SVM', 'kernel', 'f1s')

    #the_best(FPR, f1s, kernels)

    return fpr, f1
    # accuracy, precision, recall, f1_score = SVM_train(X_train, y_train, X_test, y_test)

    # print(
    #     "accuracy:" + str(accuracy) + "\nprecision:" + str(precision) + "\nrecall:" + str(recall) + "\nf1_score:" + str(
    #         f1_score))

def the_best(FPR, f1, kernels):
    '''
    Print out hyperparameter that gives the best fpr and f1 score
    '''
    print(
        f"The best kernel for SVM is {kernels[f1.index(max(f1))]} based on f1 score:\navg:{sum(f1) / len(f1)}")
    print(
        f"The best kernel for SVM is {kernels[FPR.index(min(FPR))]} based on False Positive rate:\navg:{sum(FPR) / len(FPR)}")

def cross_validation(model, x, y, cv, score_type: str, cross_result):
    scores = cross_val_score(model, x, y, cv=cv, scoring=score_type)
    cross_result.append(scores)

    # print(scores)
    return cross_result,sum(scores) / len(scores)


if __name__ == "__main__":
    '''For testing purpose only'''
    name = 'base.csv'
    portion=0.8
    X = db.data_lanudry(0.1,1, name=name)
    X_train, X_test = train_test_split(X, test_size=1 - portion, train_size=portion)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    gammas = [10, 1, 0.1, 0.01, 0.001]
    FPR, f1s = [], []

    param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    svc = svm.SVC()
    # 创建GridSearchCV实例
    grid = GridSearchCV(svc, param_grid, refit=True, verbose=2)
    # 使用训练数据拟合GridSearchCV
    grid.fit(X_train, y_train)
    # 打印最优的参数组合
    print(grid.best_params_)

    # fpr, svm_error, cross_result, f1 = SVM_train(X_train, y_train, X_test, y_test, C=grid.best_params_['C'],
    #                                              gamma=grid.best_params_['gamma'], kernel=grid.best_params_['kernel'],
    #                                              cv=cv)
    #
