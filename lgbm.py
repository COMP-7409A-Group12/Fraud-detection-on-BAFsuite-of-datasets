from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import database as db
from plot import savefig
from sklearn.metrics import confusion_matrix
from sklearn.base import clone

def LGBM_train(X_train, y_train, X_test, y_test):
    model = LGBMClassifier()
    # 现在使用编码后的数据训练模型
    model.fit(X_train, y_train)
    parameters = {'num_leaves': [10, 15, 31], 'n_estimators': [10, 20, 30], 'learning_rate': [0.05, 0.1, 0.2]}
    model = LGBMClassifier()
    grid_search = GridSearchCV(model, parameters, scoring='f1', cv=5)
    grid_search.fit(X_train, y_train)
    best_params=grid_search.best_params_
    # 打印最佳参数和最佳分数
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model using f1 score.
    """
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print("F1 Score on Test Data:", score)


def plot_cross_val_f1(model, X, y):
    # 使用KFold作为交叉验证策略
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 使用f1作为评分，并执行交叉验证
    f1_scores = cross_val_score(model, X, y, scoring="f1", cv=cv)

    # 绘制F1分数
    fig=plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, '-o')
    plt.title('Cross Validation - F1 Score')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.xticks(range(1, len(f1_scores) + 1))
    plt.ylim(0, 1)  # F1分数的范围为0到1
    plt.grid(True)
    # 当绘制F1分数的条形图后:

    # ... [rest of the plotting code for the F1 score]
    savefig("cross_val_f1", "LGBM",fig)
    plt.show()



def cross_val_fpr(model, X, y, n_splits=5):
    """
    计算交叉验证的假正率。
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fpr_values = []

    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        model_clone = clone(model)
        model_clone.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_clone.predict(X_test_fold)

        tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred_fold).ravel()
        fpr = fp / (fp + tn)
        fpr_values.append(fpr)

    return fpr_values


def plot_cross_val_fpr(model, X, y):
    """
    计算交叉验证的假正率并绘制条形图。
    """
    fpr_values = cross_val_fpr(model, X, y)

    # 画条形图展示每次折叠的假正率
    fig=plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(fpr_values) + 1), fpr_values)
    plt.xlabel('Fold Number')
    plt.ylabel('False Positive Rate (FPR)')
    plt.title('Cross Validation - FPR')
    plt.xticks(range(1, len(fpr_values) + 1))
    plt.ylim(0, 1)  # FPR的范围为0到1
    plt.grid(True)
    # 当绘制fpr分数的条形图后:

    # ... [rest of the plotting code for the F1 score]
    savefig("cross_val_fpr", "LGBM",fig)
    plt.show()




if __name__ == "__main__":
    '''For testing purpose only'''
    X = db.data_lanudry(sample_portion=0.1)

    ### 你可以用一小簇数据来快速debug
    # X.sample(frac=0.001)
    #####################################

    X_train, X_test = train_test_split(X, test_size=0.3, train_size=0.7)

    y_train = X_train['fraud_bool']
    y_test = X_test['fraud_bool']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = db.one_hot(X_train.drop(['fraud_bool'], axis=1), True, ohe)
    X_test = db.one_hot(X_test.drop(['fraud_bool'], axis=1), False, ohe)


    num_train_samples = len(X_test)
    print(f"训练集包含 {num_train_samples} 个数据样本。")
    # 对训练集中的fraud_bool列的值计数
    train_fraud_counts = y_train.value_counts()
    print("训练集中的fraud_bool计数：")
    print(train_fraud_counts)
    # 对测试集中的fraud_bool列的值计数
    test_fraud_counts = y_test.value_counts()
    print("\n测试集中的fraud_bool计数：")
    print(test_fraud_counts)

    model=LGBM_train(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    plot_cross_val_f1(model, X_test, y_test)
    plot_cross_val_fpr(model, X_test, y_test)

