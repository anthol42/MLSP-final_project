# Experiment 2

## Description
The goal of this experiment is to try to reproduce the results from the paper [Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market](https://arxiv.org/abs/1903.12258)
on US stocks. (We will train on an order of magnitude bigger dataset, and try more complex models.)
## Hypothesis
We should get similar results to them. They got 92% accuracy at predicting whether the next day will go up or down from 
the current close. This being said, because we train on a bigger dataset, we should get better results. However, the US
stock market is more liquid, which could make prediction harder, but more profitable. (We'll see how it goes).

## Methods
1. We reimplement the paper using the provided metho. 
   - Predict whether the next day is up or down
   - TW50 companies
   - Train: From 2000-01-01 to 2016-12-31; Valid: from 2017-01-01 to 2018-06-14=3; Test: from 2018-06-14 to 2020-01-01
2. We try with another visual representation (Our custom renderer about 1000x faster)
3. We try different non-predictive tasks to verify if our architectures are able to recognise pattern in the images 
(See if it is the model the problem, or the task. *i.e. predicting the future*)
4. Test on US stocks (50 biggest stocks by market cap)
   - Note: We need a 5x smaller learning rate for the model (paper) to converge
5. Test on US50 and TW50 with another, bigger, model.
   - Note: It over fits shit ton
6. Try on a bigger dataset (randomly sample 10% of US stocks that existed before 2017-01-01). To keep the task fair, we scaled the valid and test set accordingly

### Task 1: Up or Down (*ud*)
This first task aim to test if the model can understand basic patterns. The goal of this task is to accurately affirm 
whether the price of current day is higher than the previous day. To accurately perform this task, the model needs to 
check if the end of the candle (top if green and bottom if red) is higher or lower than the one of the previous candle.
This task is analogue to classical ML image classification because the class is linked to patterns in the image. Any 
human would be really good at this task, like a human is really good at classifying ImageNet images.  
(0: The price is lower; 1: The price is higher)

**Failure case**: When there is really low or no volatility, the prices might stay the same, or almost the same. Because 
the visual representation of the image consist of quantized prices, small prices movement are aligned on the same pixel 
row, meaning that the model cannot see the difference. In those cases, the model cannot guess if the price went up or 
down.

### Task 2: Count candles (*count*)
This task is a little bit harder than the previous one, and aims at testing *reasoning* capabilities of the model.
The goal of this class is to determine if there are more green candles than red one in the chart. To do so, the model 
needs to count both type of candles, than compares them. We think this task is harder, because it requires more 
*mental computing* from a human than task one. In task 1, we could easily determine the class without really thinking 
about it. In this case, we need to count and make some calculation in our head to get the good answer. We believe that 
most human would get really good results at this task too.
(0: There is more red candles; 1: There is more green candles)

**Failure case**: We do not think of any failure cases for this task *a priori*

## Runs
| RunID | Command | Objective                                                                                                         |
|------|-|-------------------------------------------------------------------------------------------------------------------|
| 10   |```python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --config.data.plt_fig=True```| Try to reproduce the paper's results. We have the same model, task and dataset AKA: same conditions               |
| 11   |```python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small"```| Implement the paper's metho, but with our image representation (Faster)                                           |
| 13   |```python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --task="ud"``` | Test the up/down task with the paper model on the taiwan dataset                                                  |
| 15   |```python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --task="count"```| Test the count task with the paper model on the taiwan dataset                                                    |
| 16   |```python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="smallUS" --config.training.lr=0.0001 --watch=accuracy``` | Test the paper model on the 50 most valuable US compagnies. (Analog to #11, but with us stocks instead of taiwan) |
| 17   |```python main.py --experiment=experiment2 --config=configs/E_s2.yml --dataset="smallUS" --config.training.lr=0.0001 --watch=accuracy``` | Test the EfficientNetV2 small model on the 50 most valuable US compagnies (Analog to #16)                         |
| 18   | ```python main.py --experiment=experiment2 --config=configs/E_s2.yml --dataset="small" --config.training.lr=0.0001 --watch=accuracy``` | Test the EfficientNetV2 small model on TW50 stocks. (Analog to #11, but with EfficientNet)                        |
| 20   | ```python main.py --experiment=experiment2 --config=configs/E_s2.yml --config.training.lr=0.0001 --watch=accuracy --fract=0.1``` | Test the EfficientNetV2 model on 10% of US stock market to see if scaling the dataset size improves performances  |
| 21   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --config.training.lr=0.0001 --watch=accuracy --fract=0.1```| Test the paper model on 10% of US stock market to see if scaling the dataset size improves performances           |
| 22   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=2 --config.data.window_len=40 --fract=0.1``` | We group the datapoints in group of two days                                                                      |
| 23   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=5 --config.data.window_len=100 --fract=0.1``` | We group in group of 5 days                                                                                       |
| 24   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=10 --config.data.window_len=200 --fract=0.1``` | We group in groups of 10 days                                                                                     |
| 25   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=2 --config.data.window_len=40 --fract=0.2``` | We group in groups of 2 days, and scale the dataset accordingly                                                   |
| 26   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=5 --config.data.window_len=100 --fract=0.5``` | We group in groups of 5 days, and scale the dataset accordingly                                                   |
| 27   | ```python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=10 --config.data.window_len=200``` | We group in groups of 10 days, and scale the dataset accordingly |



## Results
Checkout the jupyter notebook for the results visualization [here](../notebooks/ablation.ipynb)

## Short discussion

