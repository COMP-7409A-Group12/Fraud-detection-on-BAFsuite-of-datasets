import matplotlib.pyplot as plt
import os


def savefig(graph_name: str, model_name, fig) -> None:
    '''
    Input:
    graph_name: the name of the graph you want to save
    model_name: the name of your model
    fig: the graph you draw using matplotlib

    This method is intent to save the picture
    you probably want to import this method in each model.
    Therefore, in your_model.py, you could import this file
    and plot whatever you want.
    '''
    path = f'./plots/{model_name}'

    # Create directory if it doesn't exist
    # if not os.path.exists(path):
    #     os.makedirs(path)

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{model_name}_{graph_name}.png")
    fig.savefig(filename)
    print("Graph saved as", filename)


def draw(x: list, y: list,model_name:str,parameter:str,evaluation_method:str) -> None:
    """
    Input:
    x:select of the parameter value,which is represented in a number
    y:accuracy(or recall) of a model under different parameter
    parameter: which parameter you are compared
    evaluation_method: like accuracy

    This method is intent to plot the model
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制图形
    ax.plot(x, y)

    # 设置图形属性
    ax.set_xlabel(parameter)
    ax.set_ylabel(evaluation_method)

    # 设置横坐标刻度和标签
    xticks = x
    ax.set_xticks(xticks)

    graph_name=f'{evaluation_method} of different {parameter}'
    # 显示图形
    # plt.show()
    savefig(graph_name, model_name, fig)


if __name__ == "__main__":
    x=[1,2,3,4,5]
    y=[9,8,7,6,5]
    draw(x,y,'random_forest','criterion','accuracy')
