from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute, ToTSData, PadNan, Process, PipeOutput
from deep.pipes import Finviz, RmTz
from datetime import datetime
from backtest import Strategy, Backtest, RecordsBucket, TSData, Record
import pandas as pd
from typing import Optional, Dict, Callable, List
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import torch
from deep.data import gen_image
from torch.utils.data import Dataset, DataLoader

N_STOCKS = 500
@Process
def Sample(frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Dict[str, pd.DataFrame]]], **kwargs) -> Dict[str, pd.DataFrame]:
    keys = np.array(list(po.value.keys()))
    np.random.shuffle(keys)
    return {k: po.value[k] for k in keys[:N_STOCKS]}

def annotate(data: List[Dict[str, TSData]], fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.int8]], batch_size=128,
              window_size: int = 128, mode: str = 'iterative', **kwargs):
    out = {}
    for ticker, ts in tqdm(data[0].items(), desc="Annotating"):
        chart = ts.data
        tmp = _annotate(chart, fn, batch_size, window_size, mode=mode, **kwargs)
        chart["Anno"] = tmp
        out[ticker] = chart
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
                   (n - batch_size - window_size),
                   batch_size):
            b = data[i:i + batch_size + window_size - 1]
            b = np.lib.stride_tricks.sliding_window_view(b, window_shape=(window_size, ncols), axis=(0, 1))[:, 0, :, :]
            out.append(fn(b))

        # Do remainder
        exploitable_len = len(data) - window_size
        missing = exploitable_len - (exploitable_len // batch_size) * batch_size
        b = data[-missing - window_size:]

        b = np.lib.stride_tricks.sliding_window_view(b, window_shape=(window_size, ncols), axis=(0, 1))[:, 0, :, :]
        out.append(ma(b))
    else:
        b_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size, ncols), axis=(0, 1))[:, 0, :, :]
        dataset = AnnotateDataset(b_data, **kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)
        for b in loader:
            out.append(fn(b.numpy()))

    return np.concatenate(([np.nan for _ in range(window_size - 1)], *out))

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
    with pd.HDFStore(filename, 'w') as store:
        for ticker, annotation in annotations.items():
            store.put(ticker, annotation)

def load_annotations(filename: str) -> Dict[str, pd.DataFrame]:
    annotations = {}
    with pd.HDFStore(filename, 'r') as store:
        for key in store.keys():
            annotations[key[1:]] = store[key]
    return annotations

def prep_batch(batch, mode: str = 'substract', p_quant: int = 128, decay: float = 20.):
    B, L, _ = batch.shape
    out = torch.empty(B, p_quant, L, 3)
    # for i in range(B):
    #     out[i] = gen_image(batch[i], p_quant, mode, decay)
    [gen_image(batch[i], p_quant, mode, decay) for i in range(B)]
    return out

def tmp(batch):
    _ = prep_batch(batch)
    return np.zeros((batch.shape[0]))

if __name__ == "__main__":
    pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=cap_largeover%2Cexch_nyse%2Cidx_sp500%2Cipodate_more5"
                  ,True) | JSONCache() | \
           FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
           Cache() | FilterNoneCharts() | Sample() | CausalImpute() | PadNan() | ToTSData()
    print(pipe)
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    annotated_data = annotate(data, ma, window_size=64, batch_size=256, mode='iterative')
    print("Saving annotations")
    save_annotations(annotated_data, "annotations.anno")
    print("Loading annotations")
    loaded_annotations = [load_annotations("annotations.anno")]
    print(loaded_annotations)