import pandas

def data_lanudry(sample_method:str, sample_portion:str,name:str='base.csv'):
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
    with open(f'./database/{name}') as f:
        data = f.readlines()
    # we should let pandas to load data as DataFrame object and process the way you want it
    some_data = data_selection()

    # return DF object
    return None

def data_selection():
    '''
    A method to select data, especially to select from a non-fraud class
    Since we've seen non-fraud records are way more than fraud records
    '''
    import random as rd # maybe we can do it with pandas but I have no idea at the moment
    rd.seed() # we probably need to set a seed at some point in order to replicate some results we want to share
    
    # data lanudry e.g. how do you want to sample data?

    return None