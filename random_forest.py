import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plot
import database as db
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, make_scorer

def random_forest_train(X_train, y_train, X_test, y_test, n_estimators, random_state, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, cv=10):
    rF_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
    cross_list = []
    cross_list, average_f1 = cross_validation(rF_model, X_train, y_train, cv, 'f1', cross_list)
    cross_numlist = [i + 1 for i in range(len(cross_list[0]))]
    plot.draw(cross_numlist, cross_list[0], cross_numlist, 'RF', f'{criterion}_fold', 'cross_result_f1')



    fpr_scorer = make_scorer(false_positive_rate)
    cross_list = []
    cross_list, average_fpr = cross_validation(rF_model, X_train, y_train, cv, fpr_scorer, cross_list)
    plot.draw(cross_numlist, cross_list[0], cross_numlist, 'RF', f'{criterion}_fold', 'cross_result_fpr')


    rF_model.fit(X_train, y_train)
    y_pred = rF_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    FPR = fp / (tn + fp)
    error_rate = (fp + fn) / (tn + fp + fn + tp)
    return average_fpr, error_rate, cross_list, average_f1

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (tn + fp)

def rf_fit(X: pd.DataFrame, portion, cv):
    X_train, X_test = train_test_split(X, test_size=1-portion, train_size=portion)    # split data into train and test

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, OHE)   # one hot encoding
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, OHE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)    # standardize data
    X_test = scaler.transform(X_test)

    # The function to measure the quality of a split.
    criterions = ['gini', 'entropy', 'log_loss']
    FPR, f1s = [], []

    for criterion in criterions:
        fpr, lr_error, cross_val_score, f1 = random_forest_train(X_train, y_train, X_test, y_test, n_estimators=100, random_state=100, criterion=criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, cv=cv)

        print(fpr, f1)
        FPR.append(fpr)
        f1s.append(f1)
    criterion_numlist = [i + 1 for i in range(len(FPR))]
    plot.draw(criterion_numlist, FPR, criterions, 'RF', 'criterion', 'FPR')
    plot.draw(criterion_numlist, f1s, criterions, 'RF', 'criterion', 'f1s')

    the_best(FPR, f1s, criterions)
    return FPR, f1s

def the_best(FPR, f1, criterions):
    '''
    Print out hyperparameter that gives the best error and f1 score 
    '''
    print(f"The best solver for random forest is {criterions[f1.index(max(f1))]} based on f1 score:\navg:{sum(f1)/len(f1)}")
    print(f"The best solver for random forest is {criterions[FPR.index(min(FPR))]} based on False Positive rate:\navg:{sum(FPR)/len(FPR)}")

#------------------------------------------------------------

def cross_validation(model, x, y, cv, score_type: str, cross_result):
    scores = cross_val_score(model, x, y, cv=cv, scoring=score_type)
    cross_result.append(scores)
    # print(f'Cross validation scores: {scores}')
    return cross_result,sum(scores) / len(scores)

#------------------------------------------------------------

if __name__ == '__main__':
    X = db.data_lanudry(sample_portion=0.1)
    OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)    # split data into train and test

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, OHE)   # one hot encoding
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, OHE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)    # standardize data
    X_test = scaler.transform(X_test)
    print(random_forest_train(X_train, y_train, X_test, y_test, n_estimators=100, random_state=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)) 

'''
# random test for regular random forest
# load datatset here
data = pd.read_csv(f'./database/base.csv')
print(data.head(5))

# check for any missing values
print(data.isnull().sum())

y = data['fraud_bool']
X = data.drop(['fraud_bool'], axis=1)
print(f'X : {X.shape}')

# divide data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=100)
print(f'X_train: {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_test{y_test.shape}')

# build basic random forest model
rF_model = RandomForestClassifier(n_estimators=100, random_state=100)
rF_model.fit(X_train, y_train)

# check accuracy
#ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
rF_model.oob_score_(X_train, y_train)

'''