import pandas as pd

def display_data(sample_method:str=None, sample_portion:str=None,name:str='base.csv', detail= 'short'):
    '''
    Display data from csv
    Display each type and first 5 rows per attribute

    Input:
    name: the name of csv you want to load
    sample_method: how do you want to sample from non-fraund records
    sample_portion: how much we need sample from non-fraud records
    detail: Print data with 'long': All detail for each column in csv or 'short': a brief summary of data.
    '''
    data = data_lanudry( sample_method,sample_portion, name)

    if detail == 'short':
        print(data)
    elif detail =='long':
        for col in data.columns:
            print(f'Column: {col}')
            print(data[col].head(5))

def data_lanudry(sample_method:str=None, sample_portion:float=None,name:str='base.csv'):
    '''
    Input:
    name: the name of csv you want to load
    sample_method: how do you want to sample from non-fraund records
    sample_portion: how much we need sample from non-fraud records
    
    This method should load data from database directory then load assigned
    amount of non-fraud cases together with fraud cases (since fraud cases are tiny in comparison)

    Output:
    A DataFrame object should returned after being landuried.
    '''

    # load data:
    data = pd.read_csv(f'./database/{name}')

    # we should let pandas to load data as DataFrame object and process the way you want it
    data = data_selection(data, sample_method, sample_portion)

    # return DF object
    return data

def data_selection(data,sample_method:str=None, sample_portion:float=None):
    '''
    A method to select data, especially to select from a non-fraud class
    Since we've seen non-fraud records are way more than fraud records

    Input:
    name: the name of csv you want to load
    sample_method: how do you want to sample from non-fraund records
    sample_portion: how much we need sample from non-fraud records
    '''
    class_label = 0 # sample non-fraund cases by default
    seed = None # the seed you want to replicate result, None by default

    if sample_method == None:
        if sample_portion == None:
            return data
        
        class_non_fraud = data[data['fraud_bool'] == class_label][:int(len(data)*sample_portion)]
        data = pd.concat([class_non_fraud, data[data['fraud_bool'] == reverse_label(class_label)]])
        print(f"No sample method applys to {sample_portion*100}% of {'non-fraud' if class_label == 0 else 'fraund'} cases")

    elif sample_method=='random':
        print(f"Randomly sample {sample_portion*100}% of {'non-fraud' if class_label == 0 else 'fraund'} cases")
        # data lanudry e.g. how do you want to sample data?
        class_non_fraud = data[data['fraud_bool'] == class_label]
        sample_non_fraud = class_non_fraud.sample(frac=sample_portion,random_state=seed)
        data = pd.concat([sample_non_fraud, data[data['fraud_bool'] == reverse_label(class_label)]])
    return data

def reverse_label(class_label:int)->int:
    '''
    Return opposite class label
    '''
    if class_label == 1:
        return 0
    return 1


if __name__ == "__main__":
    print("Main?")