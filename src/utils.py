"""
File: utils.py
Authors: Pol Monroig and Josep Maria Olivé
Description: The utils script is a utility script that contains parse functions
             for each other scriipt, each parse script is made to parse the
             command line arguments of the script, it also contains information
             about the resources in the project, and any extra function that
             you might need.
"""
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import argparse
import sys
import os




categorical_features = ['make', 'fuel-type', 'aspiration',
                        'num-of-doors', 'body-style',
                        'drive-wheels', 'engine-location',
                        'fuel-system', 'engine-type',
                        'num-of-cylinders']

prediction_col = 'price'
numerical_features = ['symboling', 'normalized-losses',
                     'wheel-base', 'length', 'width',
                     'height', 'curb-weight', 'engine-size',
                     'bore', 'stroke', 'compression-ratio',
                     'horsepower', 'peak-rpm', 'city-mpg',
                     'highway-mpg']




class ResourceManager:

    RESOURCES_PATH = '../resources/'
    DATA_PATH = os.path.join(RESOURCES_PATH, 'data')
    MODELS_PATH = os.path.join(RESOURCES_PATH, 'models')
    PLOTS_PATH = os.path.join(RESOURCES_PATH, 'plots')
    PLOT_EXT = '.png'
    DATA_EXT = '.npy'

    @staticmethod
    def save_data(name, data):
        np.save(os.path.join(ResourceManager.DATA_PATH, name + ResourceManager.DATA_EXT), data)

    @staticmethod
    def load_data(name):
        return np.load(os.path.join(ResourceManager.DATA_PATH, name + ResourceManager.DATA_EXT))

    @staticmethod
    def load_table(name):
        return pd.read_csv(os.path.join(ResourceManager.DATA_PATH, name))

    @staticmethod
    def save_plot(name):
        plt.savefig(os.path.join(ResourceManager.PLOTS_PATH, name) + ResourceManager.PLOT_EXT)
        plt.clf()

    @staticmethod
    def save_model(name, model):
        dump( model, os.path.join(ResourceManager.MODELS_PATH, name))

    @staticmethod
    def load_model(name):
        return load(os.path.join(ResourceManager.MODELS_PATH, name))

    @staticmethod
    def save_history(name, history):
        outfile = open(os.path.join(ResourceManager.MODELS_PATH, name),'wb')
        pickle.dump(history, outfile)
        outfile.close()

    @staticmethod
    def load_history(name):
        infile = open(os.path.join(ResourceManager.MODELS_PATH, name),'rb')
        history = pickle.load(infile)
        infile.close()
        return history



class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


def apply_pipeline(data):
    categorical_pipeline = Pipeline([('DataframeSelector', DataFrameSelector(categorical_features)),
                                 ('Inputer', SimpleImputer(strategy='most_frequent')),
                                 ('OrdinalEncoder', OrdinalEncoder()),
                                 ('MinMaxScaling', MinMaxScaler())])

    numerical_pipeline = Pipeline([ ('DataframeSelector', DataFrameSelector(numerical_features)),
                                    ('Inputer', SimpleImputer()),
                                    ('MinMaxScaling', MinMaxScaler())])

    full_pipeline = FeatureUnion(transformer_list=[
                                ('num_pipeline', numerical_pipeline),
                                ('cat_pipeline', categorical_pipeline)
    ])

    X = np.array(full_pipeline.fit_transform(data))
    y = np.array(data[prediction_col].astype(float))
    return X, y

def dimensionality_reduction(X, n_components=0.8):
    pca = PCA(n_components=0.8)# 0.8 variance explained
    return pca.fit_transform(X)

def to_numerical(data):
    """
    Converts each of the numerical columns in the DataFrame
    into a numerical type, to prevent using th Object type
    """
    data = data[data['price'] != '?']
    # change ? for Nan
    data = data.replace('?', np.nan)
    for col in numerical_features:
        data[col] = pd.to_numeric(data[col])
    return data


def train_model(model, X, y, k=5):
    history = {}

    # KFOLD Cross Validation
    kfold = KFold(n_splits=k, shuffle=True)
    history['mse_train'] = 0
    history['mse_test'] = 0
    history['mae_train'] = 0
    history['mae_test'] = 0
    history['y_pred_train'] = {}
    history['y_pred_test'] = {}
    history['y_train'] = {}
    history['y_test'] = {}

    for fold, indices in enumerate(kfold.split(X)):
        train_index, test_index = indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        history['mse_train'] += mean_squared_error(y_train, y_pred_train)
        history['mse_test'] += mean_squared_error(y_test, y_pred_test)
        history['mae_train'] += mean_absolute_error(y_train, y_pred_train)
        history['mae_test'] += mean_absolute_error(y_test, y_pred_test)
        history['y_pred_train'][fold] = y_pred_train
        history['y_pred_test'][fold] = y_pred_test
        history['y_train'][fold] = y_train
        history['y_test'][fold] = y_test

    history['mse_train'] /= k
    history['mse_test'] /= k
    history['mae_train'] /= k
    history['mae_test'] /= k

    return history

def train_grid(model, parameters, X, y, k=5):
    grid = GridSearchCV(model, param_grid=parameters, n_jobs=4, cv=k)
    grid.fit(X, y)
    best_params = grid.best_params_

    return best_params

def plot_error(history, error, title):
    mae_train = np.array([[key, history[key][error + '_train']] for key in history.keys()])
    mae_test = np.array([[key, history[key][error + '_test']] for key in history.keys()])

    x_length = range(len(mae_test))
    plt.title(title)
    plt.plot(x_length, mae_train[:, 1].astype(np.float32), '.', label='Train Error')
    plt.plot(x_length, mae_test[:, 1].astype(np.float32), '.', label='Test Error')
    plt.legend(loc='lower left')
    plt.xticks(x_length, mae_test[:, 0])


def get_parser():
    """
    Based on the file that request the parser
    this function returns the adequate one
    """
    type = sys.argv[0]
    if type == 'prepare.py':
        return get_prepare()
    elif type == 'train.py':
        return get_train()
    elif type == 'results.py':
        return get_history()



def get_prepare():
    """
    Returns the argument parser for the prepare script
    """
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--data', type=str, default='imports-85.data')
    return parser

def get_history():
    """
    Returns the argument parser for the history script
    """
    parser = argparse.ArgumentParser(description='Save the results and plot them')
    return parser

def get_train():
    """
    Returns the argument parser for the train script
    """
    parser = argparse.ArgumentParser(description='Train different models')
    return parser
