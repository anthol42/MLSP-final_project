# Experiment 2

## Description
The goal of this experiment is to perform the same task as Experiment 2, but with 1D models instead of vision models. 
We will try the 1D analog of the model from the paper and a residual LSTM.

## Hypothesis
The model should perform slightly better since they are designed to work on time series, compared to vision models. 
On the other side, the LSTM might be more sensible to noise (And the data is really noisy). This could make the LSTM
perform worst.

## Methods
Same code and task as experiment 2 except for the following changes:
1. Data processing:
   - We normalize the data by dividing each features (Open, High, Low, Close) by the first non-null item of the features
to have relative time series.
   - To avoid extreme values that make the training unstable, we take the log10 of the relative TS with a epsilon of 1e-4.
2. We implement and test two 1D temporal models. 
   - CNN: We changed the paper's model for a 1D CNN
   - LSTM: We implement a residual LSTM with a similar number of parameters

## Runs
| RunID | Command | Objective                                                                                                         |
|-------|---------|-------------------------------------------------------------------------------------------------------------------|
| 28    | ```python main.py --experiment=experiment3 --config=configs/paper1D_2.yml --watch=accuracy --dataset="small"``` | Train the paper1D model on the TW50 dataset |
| 29    | ```python main.py --experiment=experiment3 --config=configs/paper1D_2.yml --watch=accuracy --dataset="huge" --fract=0.1``` | Train the paper1D model on 10% of US stocks |
| 



## Results
Checkout the jupyter notebook for the results visualization [here](../notebooks/ablation.ipynb)

## Short discussion

