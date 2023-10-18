import database as db
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

def logistic_regression(X, y):
    lr = LogisticRegression()
    lr.fit(X,y)
    
    pass

if __name__ == "__main__":
    '''For testing purpose only'''

    X, y = train_test_split(db.data_lanudry(),test_size=0.3, train_size=0.7)
