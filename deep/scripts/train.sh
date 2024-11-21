#!/bin/bash
source venv/bin/activate

# Launch training
echo "--------------------------------------------"
echo "           Starting training...              "
echo "--------------------------------------------"

#python main.py --experiment=experiment1 --config=configs/E_s.yml --fract=0.1
#python main.py --experiment=experiment1 --config=configs/E_m.yml --fract=0.1
#python main.py --experiment=experiment1 --config=configs/C_s.yml --fract=0.1
#python main.py --experiment=experiment1 --config=configs/C_t.yml --fract=0.1
#python main.py --experiment=experiment1 --config=configs/V_16.yml --fract=0.1
#python main.py --experiment=experiment1 --config=configs/V_32.yml --fract=0.1
# The same dataset size
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=2 --config.data.window_len=40 --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=5 --config.data.window_len=100 --fract=0.1
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=10 --config.data.window_len=200 --fract=0.1

# Increase the dataset size (About the same number of samples)
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=2 --config.data.window_len=40 --fract=0.2
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=5 --config.data.window_len=100 --fract=0.5
python main.py --experiment=experiment2 --config=configs/paper_2.yml --watch=accuracy --sample_inputs --config.data.group_size=10 --config.data.window_len=200
echo "--------------------------------------------"
echo "                  Done!                     "
echo "--------------------------------------------"

