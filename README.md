# Machine Learning regression project
The project is divided into multiple files
- data_analysis.ipynb: represents the notebook where we made the data analysis, and all the visualizations.
- src/prepare.py: this file reads the raw data and applies any preprocessing to it (i.e., dimensionality reduction, imputer, normalization), it saves the processed data into a file
- src/train.py: it trains the multiple models, finding out the best parameters and saving them, as well as the training history.
- src/utils.py: contains any utility functions to run any code.
- src/results.py: it reads the training history and it generates any relevant plots.

## Execution
The code is made as modular as possible, this enables us to separate different tasks and execute them independently. The proper way to execute the files is
`prepare -> train -> results`, without parameters

## Authors
Josep Maria Olivé
Pol Monroig 
