# Machine Learning Regression Project: Automobile Price Prediction
The project is divided into multiple files
- **src/data_analysis.ipynb:** represents the notebook where we made the data analysis, and all the visualizations.
- **src/prepare.py:** this file reads the raw data and applies any preprocessing to it (i.e., dimensionality reduction, imputer, normalization), it saves the processed data into a file
- **src/train.py:** it trains the multiple models, finding out the best parameters and saving them, as well as the training history.
- **src/utils.py:** contains any utility functions to run any code.
- **src/results.py:** it reads the training history and it generates any relevant plots.
- **docs/Docs.pdf:** project documentation
- **docs/APAProjectGuide.pdf:** project guide

## Execution
The code is made as modular as possible, this enables us to separate different tasks and execute them independently. The proper way to execute the files is in the following order
`requirements -> prepare -> train -> results`, without parameters. Although we provide the trained models and the definitve history of those models, so you can spare the execution entirely.

`pip install -r requirements.txt`
`python prepare.py`
`python train.py`
`python results.py`

## Authors
* Josep Maria Olivé <br>
* Pol Monroig
