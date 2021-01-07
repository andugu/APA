"""
File: prepare.py
Authors: Pol Monroig and Josep Maria Olivé
Description: The results script, collects the results made by training
             and plots them for visualization

"""
from sklearn.metrics import r2_score
import utils

def main():
    history = utils.ResourceManager.load_history('history_definitive')
    history = utils.calculate_error(history, r2_score, 'r2_score')


    utils.plot_error(history=history, error='mse', title='Mean Squared Error')
    utils.ResourceManager.save_plot('mse_comparison')
    utils.plot_error(history=history, error='mae', title='Mean Absolute Error')
    utils.ResourceManager.save_plot('mae_comparison')
    utils.plot_error(history=history, error='r2_score', title='R2 Score')
    utils.ResourceManager.save_plot('r2_score')






if __name__ == '__main__':
    main()
