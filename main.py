# import logistic_regression
# import knn
import database as db
import SVM
import logistic as lr

visualize = True # If you want to see all attribute in terminal, False by default
sample_method = None # Choose between None and 'random'
sample_portion = 0.1 # Expecting a float number for example: 0.3
name = 'base.csv'
# Lenght attribute print data with full details or in brief.
detail = 'short' # choose between 'short' and 'long'; 'short' by default

def main():
    '''
    This is main logic of our project
    TODO: implement database.py to correctly load data and display with pandas library
    TODO: implement models namely, XXX.py and we import them on top
    TODO: After we get results from each model(s) we need to draw them and display results, if any.
    '''
    X = db.data_lanudry(sample_portion=sample_portion)

    # lr_FPR, lr_error = lr.lr_fit(X)
    # print(lr_FPR, lr_error)
    svm_FPR,svm_error = SVM.svm_fit(X)
    print(svm_FPR,svm_error)

    if visualize:
        db.display_data(sample_method,sample_portion,name, detail)




main()

