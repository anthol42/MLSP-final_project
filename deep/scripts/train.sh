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
python main.py --experiment=experiment2 --config=configs/E_s2.yml --config.training.lr=0.0001 --watch=accuracy
python main.py --experiment=experiment2 --config=configs/paper_2.yml --config.training.lr=0.0001 --watch=accuracy

echo "--------------------------------------------"
echo "                  Done!                     "
echo "--------------------------------------------"

