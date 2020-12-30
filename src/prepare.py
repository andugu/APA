"""
File: prepare.py
Authors: Pol Monroig and Josep Maria Olivé
Description: The prepare script is responsible of loading the data, applying
             a dimensionality reduction, removing outliers, imputing missing values
             and saving the data ready for training.

"""
import utils

def main():
    parser = utils.get_parser()
    args = parser.parse_args()
    data = utils.ResourceManager.load_table(args.data)
    data = utils.to_numerical(data)
    X, y = utils.apply_pipeline(data)
    X = utils.dimensionality_reduction(X)
    utils.ResourceManager.save_data('X', X)
    utils.ResourceManager.save_data('y', y)




if __name__ == '__main__':
    main()
