"""
File: prepare.py
Authors: Pol Monroig and Josep Maria Olivé
Description: The results script, collects the results made by training
             and plots them for visualization

"""
import utils

def main():
    parser = utils.get_parser()
    history = utils.ResourceManager.load_history('history')
    utils.plot_error(history=history, error='mse', title='Mean Squared Error')
    utils.ResourceManager.save_plot('mse_comparison')
    utils.plot_error(history=history, error='mae', title='Mean Absolute Error')
    utils.ResourceManager.save_plot('mae_comparison')




if __name__ == '__main__':
    main()
