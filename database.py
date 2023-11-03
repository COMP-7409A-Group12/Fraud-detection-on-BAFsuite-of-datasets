import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def display_data(sample_method: str = None, sample_portion: str = None, name: str = 'base.csv', detail='short'):
    '''
    Display data from csv
    Display each type and first 5 rows per attribute

    Input:
    name: the name of csv you want to load
    sample_method: how do you want to sample from non-fraund records
    sample_portion: how much we need sample from non-fraud records
    detail: Print data with 'long': All detail for each column in csv or 'short': a brief summary of data.
    '''
    data = data_lanudry(sample_method, sample_portion, name)

    if detail == 'short':
        print(data)
    elif detail == 'long':
        for col in data.columns:
            print(f'Column: {col}')
            print(data[col].head(5))


def data_lanudry( sample_portion1: float = None,sample_portion2: float = None, name: str = 'base.csv'):
    '''
    Input:
    name: the name of csv you want to load
    sample_portion1: how much we need sample from non-fraud records
    sample_portion2: how much we need sample from fraud records

    This method should load data from database directory then load assigned
    amount of non-fraud cases together with fraud cases (since fraud cases are tiny in comparison)

    Output:
    A DataFrame object should returned after being landuried.
    '''

    # load data:
    try:
        data = pd.read_csv(f'./database/{name}')
    except FileNotFoundError:
        print(f"file {name} is missing!")
        exit(-1)

    # we should let pandas to load data as DataFrame object and process the way you want it
    data = data_selection(data, sample_portion1, sample_portion2)

    # return DF object
    return data


def data_selection(data, sample_portion1: float = None, sample_portion2: float = None):
    '''
    A method to select data, especially to select from a non-fraud class
    Since we've seen non-fraud records are way more than fraud records

    Input:
    name: the name of csv you want to load
    
    sample_portion1: how much we need sample from non-fraud records
    sample_portion2: how much we need sample from fraud records
    '''
    seed = None  # the seed you want to replicate result, None by default
    class_non_fraud = data[data['fraud_bool'] == 0]
    class_fraud = data[data['fraud_bool'] == 1]

    
    print(f"Randomly sample {sample_portion1 * 100}% of non-fraud cases")
    print(f"Randomly sample {sample_portion2 * 100}% of fraud cases")

    sample_non_fraud = class_non_fraud.sample(frac=sample_portion1, random_state=seed)
    sample_fraud = class_fraud.sample(frac=sample_portion2, random_state=seed)
    data = pd.concat([sample_non_fraud, sample_fraud])
    return data



def one_hot(data, train, ohe):
    '''
    One hot encoding for columns that are categorical
    Input:
    data: X train or X test data with columns you want to encoder
    train: A boolean value, true if passing train data false otherwise
    ohe: One hot encoder to encode your dataset

    Output:
    a Dataframe includes both encoded columns and other numerical columns
    '''
    X_categorical = data.loc[:, data.dtypes == 'object']
    if train:
        X_categorical_encoded = pd.DataFrame(ohe.fit_transform(X_categorical))
    else:
        X_categorical_encoded = pd.DataFrame(ohe.transform(X_categorical))
    X_categorical_encoded.index = X_categorical.index
    X_categorical_encoded.columns = X_categorical_encoded.columns.astype(str)
    return pd.concat([X_categorical_encoded, data.drop(columns=X_categorical.columns.tolist())], axis=1)


if __name__ == "__main__":
    '''For testing purpose'''
    print("This is testing result for database.py: ")
