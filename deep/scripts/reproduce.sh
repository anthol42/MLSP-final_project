#!/bin/bash
source venv/bin/activate

# Experiment 2
python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --config.data.plt_fig=True --sample_inputs
python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small"
python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --task="ud"
python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --task="count"
python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="smallUS" --config.training.lr=0.0001 --watch=accuracy
python main.py --experiment=experiment2 --config=configs/E_s2.yml --dataset="smallUS" --config.training.lr=0.0001 --watch=accuracy
python main.py --experiment=experiment2 --config=configs/E_s2.yml --dataset="small" --config.training.lr=0.0001 --watch=accuracy
python main.py --experiment=experiment2 --config=configs/E_s2.yml --config.training.lr=0.0001 --watch=accuracy --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --config.training.lr=0.0001 --watch=accuracy --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=2 --config.data.window_len=40 --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=5 --config.data.window_len=100 --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=10 --config.data.window_len=200 --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=2 --config.data.window_len=40 --fract=0.2
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=5 --config.data.window_len=100 --fract=0.5
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=10 --config.data.window_len=200
python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset=small --config.training.lr=0.0001 --watch=accuracy --split="stocks"
python main.py --experiment=experiment2 --config=configs/V_32.yml --watch=accuracy --dataset="small" --sample_inputs
python main.py --experiment=experiment2 --config=configs/V_16.yml --watch=accuracy --dataset="small" --sample_inputs

# Experiment 3
python main.py --experiment=experiment3 --config=configs/paper1D_2.yml --watch=accuracy --dataset="small"
python main.py --experiment=experiment3 --config=configs/paper1D_2.yml --watch=accuracy --dataset="huge" --fract=0.1
python main.py --experiment=experiment3 --config=configs/resLSTM.yml --watch=accuracy --dataset="small"
python main.py --experiment=experiment3 --config=configs/resLSTM.yml --watch=accuracy --dataset="huge" --fract=0.1
python main.py --experiment=experiment3 --config=configs/LSTM.yml --watch=accuracy --dataset="huge" --fract=0.1 --sample_inputs
