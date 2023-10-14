import matplotlib

def savefig(graph_name:str, model_name, fig):
    '''
    Input:
    graph_name: the name of the graph you want to save
    model_name: the name of your model
    fig: the graph you draw using matplotlib

    This method is intent to draw pictures and save them
    you probably want to import this method in each model.
    Therefore, in your_model.py, you could import this file
    and plot whatever you want.
    '''
    path=f'./plots/{model_name}'

    return None