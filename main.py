import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron


def plot(X, y, flower_label1=1, flower_label2=2):
    plt.figure(1)
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label=flower_label1)
    plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label=flower_label2)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend(loc='upper left')


def train(X, y):
    model = Perceptron(learning_rate=0.1, n_iterations=10)
    model.fit(X, y)
    plt.figure(2)
    plt.plot(range(1, len(model.errors) + 1), model.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of weight updates') # we only update when there is an error in prediction
    print('Number of errors per epoch (should be decreasing)')
    plt.show()


def start(df, flower_type1=1, flower_type2=2):
    """
    Select the types flowers you want to classify
    * initially classifying setosa and versicolor
    """
    if flower_type1 == 2 and flower_type2 == 3:
        y = df.iloc[50:150, 4].values
        y = np.where(y == 'Iris-versicolor', -1, 1)
        X = df.iloc[50:150, [0, 2]].values  # extract only sepal_length and petal length
        flower_label1 = 'versicolor'
        flower_label2 = 'virginica'
    elif flower_type1 == 1 and flower_type2 == 3:
        # join labels from start and end of dataframe
        y_ft1 = df.iloc[:50, 4] 
        y_ft2 = df.iloc[100:150, 4]
        y = pd.concat([y_ft1, y_ft2]).values
        y = np.where(y == 'Iris-setosa', -1, 1)
        # join features from start and end of dataframe
        X_ft1 = df.iloc[:50, [0, 2]]
        X_ft2 = df.iloc[100:150, [0, 2]]
        X = pd.concat([X_ft1, X_ft2]).values
        flower_label2 = 'virginica'
    else:
        y = df.iloc[:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[:100, [0, 2]].values
        flower_label1 = 'setosa'
        flower_label2 = 'versicolor'

    
    #print('features: \n{} ...'.format(X[:10]))
    #print('labels: \n{} ...'.format(y[:10]))
    print('Separation of 2D features per flower type...')
    plot(X, y, flower_label1, flower_label2)
    train(X, y)


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print('DataFrame shape {}'.format(df.shape))
    print('Select which two types of flowers you would like to classify split by ",":')
    ft1, ft2 = (int(x) for x in input('1 - setosa\n2 - versicolor\n3 - versinica\n~> ').split(','))
    start(df, ft1, ft2)


if __name__ == '__main__':
    main()

