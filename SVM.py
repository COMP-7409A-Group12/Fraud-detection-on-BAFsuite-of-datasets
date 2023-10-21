from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import database as db


# train SVM  //rbf>linear>poly>sigmoid
def SVM(X_train, y_train, X_test, y_test):
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
        y_test, y_pred)


#
def svm_fit():
    X = db.data_lanudry(sample_portion=0.1)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    accuracy, precision, recall, f1_score = SVM(X_train, y_train, X_test, y_test)
    print(
        "accuracy:" + str(accuracy) + "\nprecision:" + str(precision) + "\nrecall:" + str(recall) + "\nf1_score:" + str(
            f1_score))


if __name__ == "__main__":
    '''For testing purpose only'''
    X = db.data_lanudry(sample_portion=0.1)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(SVM(X_train, y_train, X_test, y_test))

# FPR

# # 加载数据集
# with open(f'./database/base.csv') as f:
#     data = f.readlines() # 将数据集加载到data变量中
#     data = np.array([data])
#     # 划分特征和标签
#     X = data[:, 1:]  # 特征
#     y = data[:, 0]  # 标签
#
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 创建SVM分类器对象
#     clf = svm.SVC(kernel='linear')
#
#     # 在训练集上训练SVM分类器
#     clf.fit(X_train, y_train)
#
#     # 在测试集上进行预测
#     y_pred = clf.predict(X_test)
#
#     # 计算准确率
#     accuracy = accuracy_score(y_test, y_pred)
#     print("准确率：", accuracy)
