from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}

# 创建一个SVM分类器实例
svc = svm.SVC()

# 创建GridSearchCV实例
grid = GridSearchCV(svc, param_grid, refit=True, verbose=2)

# 使用训练数据拟合GridSearchCV
grid.fit(X_train, y_train)

# 打印最优的参数组合
print(grid.best_params_)