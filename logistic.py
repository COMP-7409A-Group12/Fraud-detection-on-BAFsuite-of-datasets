import pandas
from sklearn.metrics import confusion_matrix

import database as db
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import plot
from sklearn.model_selection import train_test_split, cross_val_score


def logistic_regression_train(X_train, y_train, X_test, y_test,solver):

    lr = LogisticRegression(solver=solver,max_iter=999)
    #########################################################
    cross_validation(lr, X_train, y_train,10,'f1')
    #########################################################


    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    FPR = fp / (tn + fp)
    error_rate = (fp + fn) / (tn + fp + fn + tp)
    return FPR, error_rate



def lr_fit(X:pandas.DataFrame):
    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    solvers = ['lbfgs','liblinear','newton-cg', 'newton-cholesky','sag', 'saga']
    FPR, error = [], []

    for solver in solvers:
        fpr, lr_error = logistic_regression_train(X_train, y_train, X_test, y_test, solver=solver)

        print(fpr, lr_error)
        FPR.append(fpr)
        error.append(lr_error)
    solver_numlist = [i + 1 for i in range(len(FPR))]
    plot.draw(solver_numlist, FPR, 'LR', 'solver', 'FPR')
    plot.draw(solver_numlist, error, 'LR', 'solver', 'error')

    return FPR, error

#########################################################
def cross_validation(model, x, y,cv,score_type:str):
    scores = cross_val_score(model, x, y, cv=cv, scoring=score_type)
    print(scores)
#########################################################
if __name__ == "__main__":
    '''For testing purpose only'''
    X = db.data_lanudry(sample_portion=0.1)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train, X_test = train_test_split(X,test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'],axis=1),True,ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'],axis=1),False, ohe)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(logistic_regression_train(X_train, y_train, X_test, y_test))