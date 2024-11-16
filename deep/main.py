import argparse
from utils.color import TraceBackColor, Color
import sys
from datetime import datetime
import os
sys.excepthook = TraceBackColor()
parser = argparse.ArgumentParser()

# ######################################################################################################################
# ------------------------------------------- Import here your experiments ------------------------------------------- #
# ######################################################################################################################
from experiments.experiment1 import experiment1
from experiments.experiment2 import experiment2


# ######################################################################################################################
# --------------------------------- Add here your arguments to pass to the experiment -------------------------------- #
# ######################################################################################################################
parser.add_argument("--experiment", required=True, type=str)
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--dataset", required=False, type=str, default="huge")
parser.add_argument("--comment", required=False, type=str, default=None)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--noscaler", action="store_true", default=False)
parser.add_argument("--fract", required=False, type=float, default=1.) # Fraction of dataset


# ######################################################################################################################
# ------------------------------------------ Register you experiments here ------------------------------------------- #
# ######################################################################################################################
experiments = {
    "experiment1":experiment1,
    "experiment2":experiment2,
}


def type_value(val: str):
    """
    Convert the value to int or float if it is convertible.
    :param val: The string value
    :return: The converted value
    """
    if val.isdigit():
        return int(val)
    if val.count(".") == 1:
        try:
            return float(val)
        except ValueError:
            pass
    if val == "True":
        return True
    if val == "False":
        return False
    return val

def parse_kwargs(kwargs_list):
    kwargs = {}
    for item in kwargs_list:
        if item.startswith('--'):
            key_value = item.lstrip('--').split('=', 1)  # Remove '--' and split key=value
            if len(key_value) == 2:
                key, value = key_value
                kwargs[key] = type_value(value)
            else:
                raise ValueError(f"Invalid format for argument '{item}'. Expected --key=value.")
    return kwargs



if __name__ == "__main__":
    start = datetime.now()
    args, unknown_args = parser.parse_known_args()
    kwargs = parse_kwargs(unknown_args)
    experiment = experiments.get(args.experiment)
    if experiment is None:
        raise ValueError(f"Invalid experiment name!  Available experiments are: \n{list(experiments.keys())}")
    os.environ['TORCH_HOME'] = f'{os.getcwd()}/.cache'
    experiment(args, kwargs)
    end = datetime.now()
    print(f"Done!  Total time: {(end - start)}")

