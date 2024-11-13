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
import numpy.typing as npt

WINDOW_SIZE = 14

def gen_image(chart: np.ndarray, p_quant = 128, mode='substract', decay: float = 20., space_between: int = 0,
              enlarge_factor: int = 1) -> torch.Tensor:
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
    elif mode == "add":
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
    elif mode == "basic":
        price_max = np.maximum(chart[:, 0], chart[:, 3])   # Max between open and close
        price_min = np.minimum(chart[:, 0], chart[:, 3])   # Min between open and close
        quant_max = np.argmin((price_max - prices[:, None]) ** 2, axis=0) # Index of top candle
        quant_min = np.argmin((price_min - prices[:, None]) ** 2, axis=0) # Index of bottom candle
        mask_up = np.full((p_quant, len(chart)), fill_value=1.)
        mask_up[np.arange(p_quant)[:, None] <= quant_max] = 0.
        mask_up[np.arange(p_quant)[:, None] > quant_high] = 0.
        mask_down = np.full((p_quant, len(chart)), fill_value=1.)
        mask_down[np.arange(p_quant)[:, None] >= quant_min] = 0.
        mask_down[np.arange(p_quant)[:, None] < quant_low] = 0.
    else:
        raise ValueError(f"Unknown mode: {mode}")
    mask = mask_up + mask_down
    mask = mask / (mask.max(axis=0) + 1e-8)

    # Step 4: Get volume
    if mode == "basic":
        volume_image = np.full_like(image, fill_value=0.6)  # Or plasma
        volume_image = mask[:, :, None] * volume_image
    else:
        volume = np.tile(np.log(1 + chart[:, 4][None, :] / (np.max(chart[:, 4] + 1))), (p_quant, 1))
        volume_image = plt.cm.coolwarm(volume)[:, : :, :3]    # Or plasma
        volume_image = mask[:, :, None] * volume_image

    # Step 5: Combine the images and clamp 0-1
    image = np.clip(image + volume_image, 0, 1)

    # Flip vertically the image
    image = image[::-1, :, :].copy()

    if space_between > 0:
        positions = np.arange(image.shape[1] - 1) + 1
        if space_between > 1:
            positions = np.concatenate([positions for _ in range(space_between)])
        image = np.insert(image, positions, 0, axis=1)

    if enlarge_factor > 1 and space_between > 0:
        gap_idx = np.arange(image.shape[1])[(image == 0).all(axis=(0, 2))]
        gap_idx = np.array([idx - i for i, idx in enumerate(gap_idx)]) # In the case we want to reintroduce them
        nonnul_image = np.repeat(image[:, (image != 0).any(axis=(0, 2)), :], enlarge_factor, axis=1)
        image = np.insert(nonnul_image, enlarge_factor * gap_idx, 0, axis=1)
    else:
        image = np.repeat(image, enlarge_factor, axis=1)
    return torch.from_numpy(image)

def add_neutral_v2(anot: npt.NDArray[bool], dur : int = 7):
    # Compute the length of each consecutive sequence of True values
    change_points = np.diff(anot.astype(int)).nonzero()[0] + 1
    segments = np.split(anot, change_points) # Segmented array

    # Calculate the lengths of each segment and repeat them to form the output array
    durations = np.concatenate([np.full(len(segment), len(segment)) for segment in segments])

    out = anot.astype(np.int32)
    out[durations < dur] = 2
    return out

def annotate_tickers(chart: np.ndarray, WINDOW_SIZE = 14):
    """
    Annotate the chart based on the closing prices
    :param chart: The chart as a numpy array
    :param WINDOW_SIZE: The window size for the moving average
    :return: The annotations
    """
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
                arg = close_price.iloc[i:idx].argmin() + i + 1
                t[i:arg - 1] = 0
                t[arg - 1:idx] = 1

        else:
            idx = np.argmin(t[i:]) + i if not np.all(t[i:]) else len(t)
            arg = close_price.iloc[i:idx].argmax() + i + 1
            t[i:arg - 1] = 1
            t[arg - 1:idx] = 0

        inflection_pts.append(arg)
        if i != idx: i = idx
        else: break

    annotations = t.astype(bool)
    # Mise à jour des annotations
    annotations = add_neutral_v2(annotations, dur=7)
    return annotations

def annotate_chart_change(chart: np.ndarray, offset = 1):
    today = chart[:-offset]
    next_day = chart[offset:]
    classes = (next_day[:, 3] > today[:, 3]).astype(int)
    return np.concatenate([classes, [np.nan] * offset])

class ImageDataset(Dataset):
    LABELS = ["DOWN", "UP", "NEUTRAL"]
    def __init__(self, data: Dict[str, pd.DataFrame], p_quant: int = 128, window_len: int = 256, mode: str = 'subtract', fract: float = 1.,
                 random_seed: Optional[int] = None, space_between: int = 0, enlarge_factor: int = 1,
                 interpolation_factor: int = 2, annotation_type: str = 'default', offset: int = 1):
        """
        :param data: The data fetched from the data pipeline
        :param p_quant: The precision of the quantification (Number of bins or height of the image)
        :param window_len: The length of the window that the image will represent (Width of the image)
        :param mode: The mode: add or subtract
        :param fract: The fraction of the dataset to use.
        :param random_seed: The random seed to use in the dataset. NOTE: It will set the random seed globally in python's random module
        :param space_between: The number of pixel between each candle
        :param enlarge_factor: Width in pixels of the candles
        :param interpolation_factor: The factor to interpolate the image. If 1, no interpolation is done
        :param annotation_type: The type of annotation to use (default or change)
        :param offset: The number of day in the future to to use in the change annotation (Used only if annotation_type is change)
        """
        self.p_quant = p_quant
        self.window_len = window_len
        self.mode = mode
        self.space_between = space_between
        self.enlarge_factor = enlarge_factor
        self.interpolation_factor = interpolation_factor
        if fract < 1.:
            data = self.sample_data(data, fract, seed=random_seed)

        if annotation_type == 'default':
            self.offsets, self.data = self.process_data(data, window_len)
        elif annotation_type == 'change':
            self.offsets, self.data = self.process_data_change_annotation(data, window_len, offset)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")

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
            if len(chart) > window_len + 4 * WINDOW_SIZE + 1:
                offsets.append(len(chart) - window_len + 1)
                out_data.append((torch.from_numpy(chart.values[2 * WINDOW_SIZE:-2 * WINDOW_SIZE]),
                                 annotate_tickers(chart.values, WINDOW_SIZE)[2 * WINDOW_SIZE:-2 * WINDOW_SIZE]))
        offsets.append(offsets[-1] + 1)    # To avoid an overflow in the indexing algorithm where all offset are passed
        offsets = np.cumsum(offsets)
        return offsets, out_data

    @staticmethod
    def process_data_change_annotation(data: Dict[str, pd.DataFrame], window_len, offset: int = 1) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """
        Process the data into the format we want and generate the index
        :param data: The raw data to process
        :param window_len: The length of the window
        :return: The index, the processed data
        """
        offsets = [-1]
        out_data = []
        for name, chart in tqdm(data.items()):
            if len(chart) > window_len + 4 * WINDOW_SIZE + 1:
                offsets.append(len(chart) - window_len + 1)
                out_data.append((torch.from_numpy(chart.values[:-offset]),
                                 annotate_chart_change(chart.values, offset)[:-offset]))
        offsets.append(offsets[-1] + 1)
    def __getitem__(self, idx):
        # Get coordinates
        chart_idx = np.argmin(idx > self.offsets[1:])
        i = idx - self.offsets[chart_idx] - 1
        # Return the processed sample and the label
        chart, labels = self.data[chart_idx]
        window = chart[i:i + self.window_len]
        return (self.process(window, self.p_quant, self.mode, self.space_between, self.enlarge_factor, self.interpolation_factor),
                torch.tensor(labels[i + self.window_len - 1]).long())

    @staticmethod
    def process(window: torch.Tensor, p_quant: int, mode: str, space_between: int, enlarge_factor: int, interpolation_factor: int) -> torch.Tensor:
        """
        Process the window into an image and a label
        :param window: The window to process
        :param p_quant: The quantification of the price
        :return: The image, the label
        """
        image = gen_image(window.numpy(), p_quant, mode=mode, space_between=space_between, enlarge_factor=enlarge_factor)
        image = image.permute(2, 0, 1)

        # Interpolate 2x
        if interpolation_factor > 1:
            image = F.interpolate(image.unsqueeze(0), scale_factor=interpolation_factor, mode='nearest').squeeze(0)
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
def make_dataloader(config, pipe: DataPipe, start: datetime, train_end: datetime, val_end: datetime, test_end: datetime,
                    fract: float = 1., annotation_type: str = "default"):
    # Step 1: Fetch the data
    data = pipe.get(start, test_end)

    # Step 2: Split the data
    train_data = split_data(data, start, train_end)
    val_data = split_data(data, train_end + timedelta(days=1), val_end)
    test_data = split_data(data, val_end + timedelta(days=1), test_end)

    # Step 3: Make the datasets
    train_ds = ImageDataset(train_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                            window_len=config["data"]["window_len"], fract=fract,
                            random_seed=config["data"]["random_seed"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type=annotation_type,
                            offset=config["data"]["offset"]
                            )
    val_ds = ImageDataset(val_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                          window_len=config["data"]["window_len"], fract=fract,
                            random_seed=config["data"]["random_seed"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type=annotation_type,
                            offset=config["data"]["offset"])
    test_ds = ImageDataset(test_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                           window_len=config["data"]["window_len"], fract=fract,
                            random_seed=config["data"]["random_seed"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type=annotation_type,
                            offset=config["data"]["offset"])

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
    annot = annotate_tickers(data["AAPL"].values)
    print(annot)
    # plt.plot(data["AAPL"].index, data["AAPL"]["Close"])
    # labels_up = data["AAPL"]["Close"].values.copy()
    # labels_up[annot == 0] = np.nan
    # labels_down = data["AAPL"]["Close"].values.copy()
    # labels_down[annot == 1] = np.nan
    # plt.scatter(data["AAPL"].index, labels_up, color='green')
    # plt.scatter(data["AAPL"].index, labels_down, color='red')
    # plt.show()
    # dataset = ImageDataset(data, mode='subtract', window_len=20, space_between=1, enlarge_factor=1, interpolation_factor=1)
    # print(len(data["AAPL"]), len(data["NVDA"]), len(data["META"]))
    # # print(dataset[11214])
    # print(len(dataset))
    # k = 0
    # for i, (image, label) in enumerate(tqdm(dataset)):
    #     if i % 256 == 0:
    #         plt.imshow(image.permute(1, 2, 0), aspect=1)
    #         plt.tight_layout()
    #         # plt.axis("off")
    #         plt.show()
    #         # plt.savefig("/Users/alavertu/Downloads/viz_repr_no_spectrum.png", dpi=400)
