import pandas
from sklearn.metrics import confusion_matrix

import database as db
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import plot
from sklearn.model_selection import train_test_split, cross_val_score


def logistic_regression_train(X_train, y_train, X_test, y_test,solver,cv):

    lr = LogisticRegression(solver=solver,max_iter=100)
    f1 = cross_validation(lr, X_train, y_train,cv,'f1')


    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fpr = fp / (tn + fp)
    # error_rate = (fp + fn) / (tn + fp + fn + tp)
    return fpr, f1



def lr_fit(X:pandas.DataFrame, portion, cv):
    X_train, X_test = train_test_split(X, test_size=1-portion, train_size=portion)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    solvers = ['lbfgs','liblinear','newton-cg', 'newton-cholesky','sag', 'saga']
    FPR, f1s = [], []

    for solver in solvers:
        fpr, f1 = logistic_regression_train(X_train, y_train, X_test, y_test, solver=solver,cv=cv)

        # print(fpr, lr_error)
        FPR.append(fpr)
        f1s.append(f1)
    solver_numlist = [i + 1 for i in range(len(FPR))]

    plot.draw(solver_numlist, FPR, 'LR', 'solver', 'FPR')
    plot.draw(solver_numlist, f1s, 'LR', 'solver', 'f1')

    the_best(FPR, f1s, solvers)
    return FPR, f1s

def the_best(FPR, f1, solvers):
    '''
    Print out hyperparameter that gives the best error and f1 score 
    '''
    print(f"The best solver for logistic regression is {solvers[f1.index(max(f1))]} based on f1 score:\navg:{sum(f1)/len(f1)}")
    print(f"The best solver for logistic regression is {solvers[FPR.index(min(FPR))]} based on False Positive rate:\navg:{sum(FPR)/len(FPR)}")


def cross_validation(model, x, y,cv,score_type:str)->float:
    '''
    return a mean value of f1 score
    '''
    scores = cross_val_score(model, x, y, cv=cv, scoring=score_type)
    # print(scores)
    return sum(scores)/len(scores)

if __name__ == "__main__":
    '''For testing purpose only'''
    print("Testing result for logisitc.py : ")
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
    # print(logistic_regression_train(X_train, y_train, X_test, y_test))