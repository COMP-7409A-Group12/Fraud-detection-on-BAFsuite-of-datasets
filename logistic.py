import pandas
from sklearn.metrics import confusion_matrix

import database as db
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import plot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def logistic_regression_train(X_train, y_train, X_test, y_test, solver, penalty,cv):
    lr = LogisticRegression(solver=solver, penalty=penalty,max_iter=1100)


    cross_list = []
    cross_list, cross_average_f1 = cross_validation(lr, X_train, y_train, cv, 'f1', cross_list)
    cross_numlist = [i + 1 for i in range(len(cross_list[0]))]
    plot.draw(cross_numlist, cross_list[0], cross_numlist, 'lr', f'{solver}_fold', 'cross_result_f1')


    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    # 创建一个评分器fpr
    fpr_scorer = make_scorer(false_positive_rate)
    cross_list = []
    cross_list, cross_average_fpr = cross_validation(lr, X_train, y_train, cv, fpr_scorer, cross_list)
    plot.draw(cross_numlist, cross_list[0], cross_numlist, 'lr', f'{solver}_fold', 'cross_result_fpr')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fpr = fp / (tn + fp)
    # error_rate = (fp + fn) / (tn + fp + fn + tp)
    return cross_average_fpr,  cross_list,cross_average_f1


def false_positive_rate(y_test, y_pred):
    # 计算FPR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (tn + fp)
    return fpr

def lr_fit(X: pandas.DataFrame, portion, cv):
    X_train, X_test = train_test_split(X, test_size=1 - portion, train_size=portion)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    FPR, f1s = [], []

    # for solver in solvers:
    #     fpr, cross_result,f1 = logistic_regression_train(X_train, y_train, X_test, y_test, solver=solver, cv=cv)
    #
    #     # print(fpr, lr_error)
    #     FPR.append(fpr)
    #     f1s.append(f1)
    # solver_numlist = [i + 1 for i in range(len(FPR))]
    #
    # plot.draw(solver_numlist, FPR, solver_numlist, 'LR', 'solver', evaluation_method='FPR')
    # plot.draw(solver_numlist, f1s, solver_numlist, 'LR', 'solver', evaluation_method='f1')
    #
    # the_best(FPR, f1s, solvers)
    '''Best parameter combination: solver = saga, penelty = l1'''

    fpr, cross_result, f1 = logistic_regression_train(X_train, y_train, X_test, y_test, solver='saga', penalty='l1', cv=cv)
    print(f"average f1 score from k-cross validation is:{f1}")
    print(f"average fpr score from k-cross validation is:{fpr}")

    return fpr, f1


def the_best(FPR, f1, solvers):
    '''
    Print out hyperparameter that gives the best error and f1 score 
    '''
    print(
        f"The best solver for logistic regression is {solvers[f1.index(max(f1))]} based on f1 score:\navg:{sum(f1) / len(f1)}")
    print(
        f"The best solver for logistic regression is {solvers[FPR.index(min(FPR))]} based on False Positive rate:\navg:{sum(FPR) / len(FPR)}")


def cross_validation(model, x, y, cv, score_type: str,cross_result) -> float:
    '''
    return a mean value of f1 score
    '''
    scores = cross_val_score(model, x, y, cv=cv, scoring=score_type)
    cross_result.append(scores)
    # print(scores)
    return cross_result,sum(scores) / len(scores)


if __name__ == "__main__":
    '''For testing purpose only'''
    print("Testing result for logisitc.py : ")
    X = db.data_lanudry(0.1,1)

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # print(logistic_regression_train(X_train, y_train, X_test, y_test))

    param_grid = [
        {'solver': ['lbfgs'], 'penalty': ['l2', None]},
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
        {'solver': ['newton-cg'], 'penalty': ['l2', None]},
        {'solver': ['newton-cholesky'], 'penalty': ['l2', None]},
        {'solver': ['sag'], 'penalty': ['l2', None]},
        {'solver': ['saga'], 'penalty': ['elasticnet', 'l1', 'l2', None]}
    ]
    lr = LogisticRegression(max_iter=999)
    # 创建GridSearchCV实例
    grid = GridSearchCV(lr, param_grid, refit=True, verbose=2)
    # 使用训练数据拟合GridSearchCV
    grid.fit(X_train, y_train)
    # 打印最优的参数组合
    print(grid.best_params_)
