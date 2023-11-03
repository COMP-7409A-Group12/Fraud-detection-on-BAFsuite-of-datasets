# import logistic_regression
# import knn
import database as db

def main():
    '''
    Main logic of our project
    '''
    name = 'base.csv'

    while 1:
        print("**************************************\nData Preperation:\n**************************************")
        sample_portion1 = float(input("sample portion for non-fraud class:"))
        sample_portion2 = float(input("sample portion for fraud class:"))

        option = int(input("1.Visulize data\n2.algorithm\n3.quit\n"))
        assert option in [1,2,3], "Invalid option"

        if option == 1:
            detail = input("Detail of data? (long/short):").lower()
            print(detail)
            assert detail in ['long','short'], "Invalid option"
            db.display_data(sample_portion1, sample_portion2, name, detail)
            continue

        elif option == 3:
            print("Program exit with code 0")
            exit(0)
        
        print("**************************************\nOpertaion Selection:\n**************************************")

        option = int(input("Which alogrithm you wish to use?\n1.Logistic Regression\n2.SVM\n3.Random Forest\n4.RNN\n5.Lightbgm\n"))
        assert option in [1,2,3,4,5], "Invalid option"
        X = db.data_lanudry(sample_portion1=sample_portion1,sample_portion2=sample_portion2,name=name)

        portion = float(input("Portion for training data? Enter a float <1 :"))
        cv = int(input("Define k for kfold: "))
        
        match option:
            case 1:
                import logistic as lr
                lr_FPR, lr_error = lr.lr_fit(X, portion=portion, cv=cv)
                # print(lr_FPR, lr_error)

            case 2:
                import SVM
                svm_FPR,svm_error = SVM.svm_fit(X, portion=portion, cv=cv)
                #print(svm_FPR,svm_error)

            case 3:
                import random_forest
                rF_FPR,rF_error = random_forest.rf_fit(X)
                print(rF_FPR,rF_error)

            case 4:
                import RNN_implement as rnn
                pass

            case 5:
                import lgbm
                pass

        option = input("Go back (any key) or Quit (q)? :").lower()
        exit(0) if option == 'q' else ''
        print("\n\n")
main()

