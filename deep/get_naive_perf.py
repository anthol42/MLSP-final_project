import torch

from data import  make_dataloader
from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute
from pipes import Finviz, RmTz, FromFile
from datetime import datetime
import argparse
import sys
from utils import TraceBackColor
import utils
from tqdm import tqdm

sys.excepthook = TraceBackColor()
parser = argparse.ArgumentParser()

parser.add_argument("--config", required=True, type=str)
parser.add_argument("--dataset", required=False, type=str, default="huge")
parser.add_argument("--split", required=False, type=str, default="test")
parser.add_argument("--fract", required=False, type=float, default=1.)
parser.add_argument("--task", required=False, type=str, default="predictive") # Predictive, ud or count

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
    args, unknown_args = parser.parse_known_args()
    kwargs = parse_kwargs(unknown_args)

    if args.dataset == "huge":
        pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=ipodate_more10", True) | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
    elif args.dataset == "small":
        pipe = FromFile("tw50.json") | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
        pipe.set_id(10)
    elif args.dataset == "smallUS":
        pipe = FromFile("us50.json") | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
        pipe.set_id(20)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    config = utils.ConfigFile(args.config, verify_path=True)
    config.override_config(kwargs)

    train_loader, val_loader, test_loader = make_dataloader(config=config, pipe=pipe,
                                                            start=datetime(2000, 1, 1),
                                                            train_end=datetime(2016, 12, 31),
                                                            val_end=datetime(2018, 6, 13),
                                                            test_end=datetime(2020, 1, 1),
                                                            fract=args.fract, annotation_type="change", task=args.task)
    # if args.split == "train":
    #     dataloader = train_loader
    # elif args.split == "valid":
    #     dataloader = val_loader
    # elif args.split == "test":
    #     dataloader = test_loader
    # else:
    #     raise ValueError("Invalid split: ", args.split)

    targets = []
    for X, y in tqdm(train_loader, desc="Iterating training set"):
        targets.append(y)
    targets = torch.cat(targets, dim=0)
    _, counts = torch.unique(targets, return_counts=True)
    best_class = torch.argmax(counts)

    targets = []
    for X, y in tqdm(test_loader, desc="Iterating test set"):
        targets.append(y)

    targets = torch.cat(targets, dim=0)
    print(targets.shape)

    _, counts = torch.unique(targets, return_counts=True)
    print(f"Naive accuracy for {args.dataset}[{args.fract}] = {100*(counts[best_class] / torch.sum(counts)).round(decimals=4)}")