"""
File: prepare.py
Authors: Pol Monroig and Josep Maria Olivé
Description: The train script loads the data and applies a cross validation
             Grid search to different models to find out which is best and which
             hyper-parameters work best with it. It generates conclusive plots
             that will help in the selection between them

"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import utils

def main():
    X = utils.ResourceManager.load_data('X')
    y = utils.ResourceManager.load_data('y')
    models = {'LinearRegression' : LinearRegression(), 'KNN' : KNeighborsRegressor(),
               'LinearSVM' : LinearSVR(), 'MLP' : MLPRegressor(), 'RandomForest' : RandomForestRegressor()}

    parameters = [None, {'n_neighbors': np.linspace(2, 50, 49).astype(np.int32)},
                  {'C' : [0.1, 10, 100, 500, 800, 1000, 1500,  2000, 5000], 'epsilon' : [0, 0.05, 0.1, 10, 100, 1000, 2500, 5000]},
                  {'hidden_layer_sizes': [[100, 100, 100], [5000], [50, 50, 50, 50]], 'alpha' : [0.001], 'max_iter' : [100, 200, 500],
                   'learning_rate_init' : [0.0001, 0.001, 0.1]},
                  {'max_depth' : [1, 2, 5, 10, 100, 500, None], 'n_estimators' : [10, 25, 50, 100],
                         'min_samples_leaf' : [1, 5, 10, 15]}]

    history = {}

    # Search best parameters
    best_params = {}
    print('Finding best hyper-parameters')
    for p, key in zip(parameters, models.keys()):
        print('Searching best parameters', key)
        if p:
            best_params[key] = utils.train_grid(models[key], p, X, y)
        else:
            best_params[key] = None
    print('Training models')
    for key in models.keys():
        print('Training model', key)
        if best_params[key]:
            models[key].set_params(**best_params[key])
        history[key] = utils.train_model(models[key], X, y)

    for key in models.keys():
        print('Best params for', key, best_params[key])
        utils.ResourceManager.save_model(key, models[key])

    utils.ResourceManager.save_history('history', history)





if __name__ == '__main__':
    main()
