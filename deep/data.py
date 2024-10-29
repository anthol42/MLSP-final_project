from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from backtest.data import DataPipe
import random

WINDOW_SIZE = 14

def gen_image(chart: np.ndarray, p_quant = 128, mode='substract', decay: float = 20.) -> torch.Tensor:
    """
    Generate an image from a chart
    :param chart: A tensor in the format [Open, High, Low, Close, Volume, LABEL]
    :param p_quant: The quantification size of the price
    :param mode: The mode of the spectrum (add or subtract)
    :param decay: The decay of the spectrum (Only used when mode = subtract)
    :return: The image
    """
    # Step 1: Quantify prices
    prices = np.linspace(chart[:, 2].min(), chart[:, 1].max(), p_quant)
    quant_open = np.argmin((chart[:, 0] - prices[:, None]) ** 2, axis=0)
    quant_close = np.argmin((chart[:, 3] - prices[:, None]) ** 2, axis=0)
    # Step 2: Color the image with the 'candlesticks' like representation
    _, t_idx = np.meshgrid(np.arange(p_quant), np.arange(len(chart)))

    # Positive days
    length = 1 + quant_close - quant_open
    length[length <= 0] = 0    # Keep only positives days
    range_array = np.arange(np.max(length))
    mask = range_array < length[:, None]
    q_pos_days = (quant_open[:, None] + range_array)[mask]
    idx_pos_days = t_idx[:, :len(range_array)][mask]
    # Negative days
    length = 1 + quant_open - quant_close
    length[length < 2] = 0    # Keep only negative days
    range_array = np.arange(np.max(length))
    mask = range_array < length[:, None]
    q_neg_days = (quant_close[:, None] + range_array)[mask]
    idx_neg_days = t_idx[:, :len(range_array)][mask]

    # Generate the image
    image = np.zeros((p_quant, len(chart), 3))    # RGB
    image[q_pos_days, idx_pos_days, 1] = 1    # Green
    image[q_neg_days, idx_neg_days, 0] = 1    # Red

    # Step 3: Make the spectrum
    quant_high = np.argmin((chart[:, 1] - prices[:, None]) ** 2, axis=0)
    quant_low = np.argmin((chart[:, 2] - prices[:, None]) ** 2, axis=0)
    if mode == 'subtract':
        mask_down = np.log(1 + np.exp((quant_low[None, :] - np.arange(p_quant)[:, None]) / p_quant)) - np.log(2)
        mask_down[mask_down < 0] = 0
        mask_up = np.log(1 + np.exp((np.arange(p_quant)[:, None] - quant_high[None, :]) / p_quant)) - np.log(2)
        mask_up[mask_up < 0] = 0
    else:
        price_max = np.maximum(chart[:, 0], chart[:, 3])   # Max between open and close
        price_min = np.minimum(chart[:, 0], chart[:, 3])   # Min between open and close
        quant_max = np.argmin((price_max - prices[:, None]) ** 2, axis=0) # Index of top candle
        quant_min = np.argmin((price_min - prices[:, None]) ** 2, axis=0) # Index of bottom candle
        mask_up = np.log(1 + np.exp(-decay * np.abs(np.arange(p_quant)[:, None] - quant_max -1) / p_quant)) - np.log(2)
        mask_up += -mask_up.min()
        mask_up /= mask_up.max()
        mask_up[np.arange(p_quant)[:, None] <= quant_max] = 0.
        mask_up[np.arange(p_quant)[:, None] > quant_high] = 0.
        mask_down = np.log(1 + np.exp(-decay * np.abs(quant_min - np.arange(p_quant)[:, None] - 1) / p_quant)) - np.log(2)
        mask_down += -mask_down.min()
        mask_down /= mask_down.max()
        mask_down[np.arange(p_quant)[:, None] >= quant_min] = 0.
        mask_down[np.arange(p_quant)[:, None] < quant_low] = 0.
    mask = mask_up + mask_down
    mask = mask / (mask.max(axis=0) + 1e-8)

    # Step 4: Get volume
    volume = np.tile(np.log(1 + chart[:, 4][None, :] / (np.max(chart[:, 4] + 1))), (p_quant, 1))
    volume_image = plt.cm.coolwarm(volume)[:, : :, :3]    # Or plasma
    volume_image = mask[:, :, None] * volume_image

    # Step 5: Combine the images and clamp 0-1
    image = np.clip(image + volume_image, 0, 1)

    # Flip vertically the image
    image = image[::-1, :, :].copy()
    return torch.from_numpy(image)

def annotate_tickers(chart: np.ndarray, WINDOW_SIZE = 14):
    # Get close prices
    close_price = pd.Series(chart[:, 3])

    # Generate all windows
    moving_average = close_price.rolling(WINDOW_SIZE).mean()
    close_price = close_price[len(close_price) - len(moving_average):]

    # Split the data point on which are over or under the mean curve
    over = close_price >= moving_average

    t = np.array(over, dtype=np.int8)
    i = 0
    inflection_pts = []

    while(i < len(t)):
        val = t[i]

        if(val == 0):
            idx = np.argmax(t[i:]) + i if np.all(~t[i:]) else len(t)
            if(i != idx):
                arg = close_price[i:idx].argmin() + i + 1
                t[i:arg] = 0
                t[arg:idx] = 1

        else: # Value = 1
            idx = np.argmin(t[i:]) + i if not np.all(t[i:]) else len(t)
            arg = close_price[i:idx].argmax() + i + 1
            t[i:arg] = 1
            t[arg:idx] = 0

        inflection_pts.append(arg)
        if i != idx: i = idx
        else: break

    return t

class ImageDataset(Dataset):
    LABELS = ["DOWN", "UP", "NEUTRAL"]
    def __init__(self, data: Dict[str, pd.DataFrame], p_quant: int = 128, window_len: int = 256, mode: str = 'subtract', fract: float = 1.,
                 random_seed: Optional[int] = None):
        """
        :param data: The data fetched from the data pipeline
        :param p_quant: The precision of the quantification (Number of bins or height of the image)
        :param window_len: The length of the window that the image will represent (Width of the image)
        :param mode: The mode: add or subtract
        :param fract: The fraction of the dataset to use.
        :param random_seed: The random seed to use in the dataset. NOTE: It will set the random seed globally in
        python's random module
        """
        self.p_quant = p_quant
        self.window_len = window_len
        self.mode = mode
        if fract < 1.:
            data = self.sample_data(data, fract, seed=random_seed)
        self.offsets, self.data = self.process_data(data, window_len)

    @staticmethod
    def sample_data(data: Dict[str, pd.DataFrame], frac, seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Sample a fraction of the given data. Sample across stocks (Ex: keep 42 stocks out of 400)
        :param data: The data to be sampled from
        :param frac: The fraction of the original data to keep
        :param seed: The random seed to use. If None, no random seed is set.  NOTE: Setting the random seed here will
        set the random seed globally
        :return: The sampled data
        """
        if seed is not None:
            random.seed(seed)
        keys = list(data.keys())
        # Shuffle
        random.shuffle(keys)
        threshold = int(frac * len(data))
        return {ticker: data[ticker] for ticker in keys[:threshold]}


    @staticmethod
    def process_data(data: Dict[str, pd.DataFrame], window_len) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """
        Process the data into the format we want and generate the index
        :param data: The raw data to process
        :param window_len: The length of the window
        :return: The index, the processed data
        """
        offsets = [-1]
        out_data = []
        for name, chart in tqdm(data.items()):
            offsets.append(len(chart) - window_len + 1)
            out_data.append((torch.from_numpy(chart.values), annotate_tickers(chart.values, WINDOW_SIZE)))
        offsets.append(offsets[-1] + 1)    # To avoid an overflow in the indexing algorithm where all offset are passed
        offsets = np.cumsum(offsets)
        return offsets, out_data

    def __getitem__(self, idx):
        # Get coordinates
        chart_idx = np.argmin(idx > self.offsets[1:])
        i = idx - self.offsets[chart_idx] - 1
        # Return the processed sample and the label
        chart, labels = self.data[chart_idx]
        window = chart[i:i + self.window_len]
        return self.process(window, self.p_quant, self.mode), torch.tensor(labels[i + self.window_len - 1]).long()

    @staticmethod
    def process(window: torch.Tensor, p_quant: int, mode: str) -> torch.Tensor:
        """
        Process the window into an image and a label
        :param window: The window to process
        :param p_quant: The quantification of the price
        :return: The image, the label
        """
        image = gen_image(window.numpy(), p_quant, mode=mode)
        image = image.permute(2, 0, 1)

        # Interpolate 2x
        image = F.interpolate(image.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
        return image.float()

    def __len__(self):
        return self.offsets[-2] + 1   # Last one is just padding to avoid overflow in indexing algorithm

def split_data(data: Dict[str, pd.DataFrame], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """
    Extract the date range from the data; range: [start,end]
    :param data: The data to split
    :param start: The start of split to extract (Included)
    :param end: The end of the split to extract (Included)
    :return: The cropped data
    """
    out = {}
    for name, chart in data.items():
        out[name] = chart.loc[start.date():end.date()]
    return out
def make_dataloader(config, pipe: DataPipe, start: datetime, train_end: datetime, val_end: datetime, test_end: datetime, fract: float = 1.):
    # Step 1: Fetch the data
    data = pipe.get(start, test_end)

    # Step 2: Split the data
    train_data = split_data(data, start, train_end)
    val_data = split_data(data, train_end + timedelta(days=1), val_end)
    test_data = split_data(data, val_end + timedelta(days=1), test_end)

    # Step 3: Make the datasets
    train_ds = ImageDataset(train_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                            window_len=config["data"]["window_len"], fract=fract,
                            random_seed=config["data"]["random_seed"])
    val_ds = ImageDataset(val_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                          window_len=config["data"]["window_len"], fract=fract,
                            random_seed=config["data"]["random_seed"])
    test_ds = ImageDataset(test_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                           window_len=config["data"]["window_len"], fract=fract,
                            random_seed=config["data"]["random_seed"])

    # Step 4: Initialize the dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"],
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)
    val_dataloader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)
    test_dataloader = DataLoader(test_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    from backtest.data import FetchCharts, Cache
    from datetime import datetime
    TICKERS = ['AAPL', 'NVDA', 'META', "AMZN"]
    pipe = FetchCharts(TICKERS) | Cache()
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    dataset = ImageDataset(data, mode='add')
    print(len(data["AAPL"]), len(data["NVDA"]), len(data["META"]))
    # print(dataset[11214])
    print(len(dataset))
    k = 0
    for i, (image, label) in enumerate(tqdm(dataset)):
        if i % 256 == 0:
            plt.imshow(image)
            plt.tight_layout()
            plt.axis("off")
            plt.show()
            # plt.savefig("/Users/alavertu/Downloads/viz_repr_no_spectrum.png", dpi=400)
