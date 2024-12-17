# MLSP-final_project
This repository contains all the code necessary to reproduce the results of the paper _Investigating vision neural 
networks as financial quantitative models_. It is separated into two parts: **deep** and **evaluation**. All the code 
necessary to reproduce the results are in the deep folder. [Click here](./deep/README.md) to see more details, and 
to know how to run the code to reproduce the results. The repository also 
contains the code used to make the project proposal, which was different from the project in the paper. The next 
section explain each part of the repository.

## Root
All elements in the root directory are associated to the project proposal, and are not directly relevant to reproduce 
the results, but can show the progression of the project.
- ```annotate_stock.py```: A script used to automatically annotate trends using an oracle approach. Used in the 
proposition, but not in the project.
- ```annotate_stockV2.py```: Same thing, but more efficient because it is using numpy arrays.
- ```get_trade_dist.py```: Compute the expected gain distribution obtain from a predictor having 100% accuracy at predicting 
annotation provided by the algorithms defined in the previous files.
- ```Paper_figs.ipynb```: A notebook containing figures used in the project proposition.
- ```stock2image.py```: Our first prototype to convert stock price fluctuations into images
- ```stock2music.py```: A prototype to turn stock price fluctuation into sound (Not used).
- ```Workflow_fig.png```: The figure showing the workflow of the project proposition.
- ```Workflow_fig.pptx```: The powerpoint file containing the figure.

## Evaluation
This folder contains an unfinished prototype that could have been used to backtest a financial strategy using the deep 
learning model as the predictor engine. Due to time constraints, we did not have time to finish and test the strategy.
- ```annotate.py```: A script that uses a gpu and a model to annotate multiple charts in parallel. It then saved the 
annotated charts in a file. 
- ```backtesting.py```: A script using the annotated charts from the previous script and run a backtest using a strategy 
that buys the stock when the model predicted a uptrend, and sells it if the model predicted a downtrend.
- ```dist.py```: A script that draws a distribution of the trades performed by the model, given an annotation file 
provided by the first script.
- ```configFile.py```: A util file copied from this repository: [torchbuilder](https://github.com/anthol42/torchbuilder)

## deep
This is the folder containing the code used to make the paper. [Click here](./deep/README.md) to see more details, and 
to know how to run the code to reproduce the results.