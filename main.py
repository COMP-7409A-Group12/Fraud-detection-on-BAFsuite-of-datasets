# import logistic_regression
# import knn
import database as db

visualize = True # If you want to see all attribute in terminal, False by default
sample_method = 'random' # Choose between None and 'random'
sample_portion = 0.01 # Expecting a float number for example: 0.3
name = 'base.csv'
# Lenght attribute print data with full details or in brief.
detail = 'long' # choose between 'short' and 'long'; 'short' by default

def main():
    '''
    This is main logic of our project
    TODO: implement database.py to correctly load data and display with pandas library
    TODO: implement models namely, XXX.py and we import them on top
    TODO: After we get results from each model(s) we need to draw them and display results, if any.
    '''


    if visualize:
        db.display_data(sample_method,sample_portion,name, detail)


main()

