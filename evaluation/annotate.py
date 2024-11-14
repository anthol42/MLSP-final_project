import matplotlib.pyplot as plt
from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute, Process, PipeOutput
from deep.pipes import Finviz, RmTz
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Callable, List, Tuple
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import torch
from deep.data import gen_image, split_data
from torch.utils.data import Dataset, DataLoader
from deep import models
from evaluation.configFile import ConfigFile
import h5py
import torch.nn.functional as F

N_STOCKS = 500
@Process
def Sample(frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Dict[str, pd.DataFrame]]], **kwargs) -> Dict[str, pd.DataFrame]:
    keys = np.array(list(po.value.keys()))
    np.random.shuffle(keys)
    return {k: po.value[k] for k in keys[:N_STOCKS]}

def annotate(data: List[Dict[str, pd.DataFrame]], fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.int8]], batch_size=128,
              window_size: int = 128, mode: str = 'iterative', **kwargs):
    out = {}
    for ticker, chart in tqdm(data[0].items(), desc="Annotating"):
        if len(chart) > 2 * window_size:
            annot = _annotate(chart, fn, batch_size, window_size, mode=mode, **kwargs)
            assert len(annot) == len(chart), f"Length of annotation is not the same as the one from the chart. Annot: {len(annot)}, chart: {len(chart)}"
            out[ticker] = (chart, annot)
    return out

class AnnotateDataset(Dataset):
    def __init__(self, data, p_quant=128, mode='substract', decay=20.):
        super().__init__()
        self.data = data
        self.p_quant = p_quant
        self.mode = mode
        self.decay = decay

    def __getitem__(self, item):
        return gen_image(self.data[item], p_quant=self.p_quant, mode=self.mode, decay=self.decay)

    def __len__(self):
        return len(self.data)

def _annotate(chart: pd.DataFrame, fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.int8]], batch_size=128,
              window_size: int = 128, mode: str = 'iterative', **kwargs):
    """
    Annotate a chart using an annotation function fn
    :param chart: The chart to annotate
    :param fn: The annotation function
    :return: The annotated chart
    """
    # Batching
    data = chart.values
    n = data.shape[0]
    out = []
    ncols = data.shape[1]
    if mode == 'iterative':
        for i in range(0,
                   (n - batch_size - window_size + 1),
                   batch_size):
            b = data[i:i + batch_size + window_size - 1]
            b = np.lib.stride_tricks.sliding_window_view(b, window_shape=(window_size, ncols), axis=(0, 1))[:, 0, :, :]
            out.append(fn(b))

        # Do remainder
        exploitable_len = len(data) - window_size
        missing = exploitable_len - (exploitable_len // batch_size) * batch_size
        b = data[-missing - window_size:]
        b = np.lib.stride_tricks.sliding_window_view(b, window_shape=(window_size, ncols), axis=(0, 1))[:, 0, :, :]
        out.append(fn(b))
    else:
        b_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size, ncols), axis=(0, 1))[:, 0, :, :]
        dataset = AnnotateDataset(b_data, **kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)
        for b in loader:
            out.append(fn(b.numpy()))

    return np.concatenate(([[np.nan, np.nan, np.nan] for _ in range(window_size - 1)], *out))

def ma(data: npt.NDArray[np.float32]) -> npt.NDArray[np.int8]:
    """
    Annotate a chart using the moving average
    :param data: The chart to annotate (Shape(B, L, 5))
    :param window: The window size
    :return: The annotations of the last day: Shape (B, )
    """
    window = 14
    series = pd.DataFrame(data[:, :, 3])
    mas = series.rolling(window).mean().values # Shape(B, L)
    return (data[:, :, 3] >= mas).astype(np.int8)[:,-1]


def save_annotations(annotations: Dict[str, pd.DataFrame], filename: str):
    with h5py.File(filename, 'w') as f:
        for key, (df, array) in annotations.items():
            # Store DataFrame as a table
            group = f.create_group(key)

            # Convert DataFrame to numpy array and store as dataset
            group.create_dataset('dataframe', data=df.to_numpy())
            group.create_dataset('index', data=df.index.values.astype('datetime64[ns]').astype(np.float64))

            # Store numpy array as a dataset
            group.create_dataset('array', data=array)

def load_annotations(filename: str) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    annotations = {}
    # Open the HDF5 file in read mode
    with h5py.File(filename, 'r') as f:
        # Iterate over each group in the file
        for key in f.keys():
            group = f[key]

            # Load the DataFrame (as a numpy array) and convert it back to DataFrame
            df_data = group['dataframe'][:]
            # Load the index and convert it back to a DatetimeIndex if necessary
            index_data = group['index'][:]
            df_index = pd.to_datetime(index_data)

            # Reconstruct the DataFrame
            df = pd.DataFrame(df_data, index=df_index,
                              columns=[f"col{i}" for i in range(df_data.shape[1])])  # Recreate DataFrame

            # Load the numpy array
            array = group['array'][:]

            # Store the DataFrame and array as a tuple in the dictionary
            annotations[key] = (df, array)

    return annotations

def prep_image(b, p, m):
    image = gen_image(b, p, mode=m)
    image = image.permute(2, 0, 1)

    # Interpolate 2x
    image = F.interpolate(image.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
    return image.float()

def prep_batch(batch, mode: str = 'subtract', p_quant: int = 128, decay: float = 20.):
    B, L, _ = batch.shape
    out = torch.stack([prep_image(batch[i], p_quant, mode) for i in range(B)]).float()
    return out

# Annotate from saved model
config = ConfigFile("../deep/configs/E_s.yml")
model = models.from_name(config)
weights = torch.load(f"../deep/{config['model']['model_dir']}/1/{config['model']['name']}.pth")
model.load_state_dict(weights["model_state_dict"])
model.cuda()
model.eval()

@torch.inference_mode()
def tmp(batch):
    data = prep_batch(batch, p_quant=config["data"]["p_quant"]).cuda()
    pred = model(data)
    out = torch.softmax(pred, dim=1).cpu().numpy()
    return out

if __name__ == "__main__":
    pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=ipodate_more5", True) | JSONCache() | \
           FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
           Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
    print(pipe)
    data = pipe.get(datetime(2000, 1, 1), datetime(2024, 1, 1))
    data = split_data(data, datetime(2019, 12, 13), datetime(2024, 1, 1))

    # keys = list(data.keys())[:5]
    # data = {key:data[key] for key in keys}

    annotated_data = annotate([data], tmp, window_size=config["data"]["window_len"], batch_size=config["data"]["batch_size"], mode='iterative')
    print("Saving annotations")
    save_annotations(annotated_data, "annotations/E_s.anno")