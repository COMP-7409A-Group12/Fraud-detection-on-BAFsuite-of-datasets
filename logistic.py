import database as db
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

def logistic_regression(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train,y_train)

    
    from sklearn.metrics import accuracy_score
    
    return accuracy_score(y_test, lr.predict(X_test))


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
    print(logistic_regression(X_train, y_train, X_test, y_test))