from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from backtest.data import DataPipe
import random
import numpy.typing as npt
import mplfinance as mpf
import matplotlib as mpl
import io
import cv2
import os
mpl.rcParams['savefig.pad_inches'] = 0
mc = mpf.make_marketcolors(up='#00FF00', down='#FF0000',
                           wick="#999999",
                           edge="inherit")
s = mpf.make_mpf_style(marketcolors=mc, figcolor="black")

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

def gen_plt_image(chart: np.ndarray) -> torch.Tensor:
    chart = pd.DataFrame(chart[:, :5], columns=["Open", "High", "Low", "Close", "Volume"],
                         index=pd.date_range("2020-01-01", periods=len(chart), freq='D'))
    fig, ax = mpf.plot(chart, type='candle', style=s, block=False,
                       volume=False, ylabel='Price', ylabel_lower='Volume', figsize=(1, 1),
                       returnfig=True, axisoff=True, tight_layout=True, scale_padding=0.)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="png", pad_inches=0, bbox_inches='tight')
    io_buf.seek(0)
    arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    io_buf.close()
    plt.close(fig)
    img = cv2.imdecode(arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / img.max()
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    return tensor

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

def annotate_chart_change(chart: np.ndarray, offset = 1, predictive: bool = True):
    today = chart[:-offset]
    next_day = chart[offset:]
    classes = (next_day[:, 3] > today[:, 3]).astype(int)
    if predictive:
        return np.concatenate([classes, [np.nan] * offset])
    else:
        return np.concatenate([[np.nan] * offset, classes])
def sample_tickers(data: Dict[str, pd.DataFrame], fract: float, seed: Optional[int] = None) -> Set[str]:
    """
    Sample a fraction of the given data. Sample across stocks (Ex: keep 42 stocks out of 400)
    :param data: The data to be sampled from
    :param fract: The fraction of the original data to keep
    :param seed: The random seed to use. If None, no random seed is set.  NOTE: Setting the random seed here will
    set the random seed globally
    :return: The sampled data
    """
    if seed is not None:
        random.seed(seed)
    keys = list(data.keys())
    # Shuffle
    random.shuffle(keys)
    threshold = int(fract * len(data))
    return {ticker for ticker in keys[:threshold]}
class ImageDataset(Dataset):
    LABELS = ["DOWN", "UP", "NEUTRAL"]
    def __init__(self, data: Dict[str, pd.DataFrame], p_quant: int = 128, window_len: int = 256, mode: str = 'subtract',
                 space_between: int = 0, enlarge_factor: int = 1,
                 interpolation_factor: int = 2, annotation_type: str = 'default', offset: int = 1, plt_fig: bool = False, name: str = "train",
                 task: str = "predictive", group_size: int = 1, ts: bool = False, image_shape: Optional[Tuple[int, int]] = None):
        """
        :param data: The data fetched from the data pipeline
        :param p_quant: The precision of the quantification (Number of bins or height of the image)
        :param window_len: The length of the window that the image will represent (Width of the image)
        :param mode: The mode: add or subtract
        :param space_between: The number of pixel between each candle
        :param enlarge_factor: Width in pixels of the candles
        :param interpolation_factor: The factor to interpolate the image. If 1, no interpolation is done
        :param annotation_type: The type of annotation to use (default or change)
        :param offset: The number of day in the future to to use in the change annotation (Used only if annotation_type is change)
        :param plt_fig: If True, the image is generated using matplotlib renderer instead of the custom one
        :param task: The task that the model need to perform
        :param group_size: The number of point that are grouped together to make a single new point. (Used for downsampling)
        :param ts: If true, the data won't be an image, but temporal data.
        :param image_shape: If not None, the images will be resized to this shape before being fed to the model
        """
        self.p_quant = p_quant
        self.window_len = window_len
        self.mode = mode
        self.space_between = space_between
        self.enlarge_factor = enlarge_factor
        self.interpolation_factor = interpolation_factor
        self.plt_fig = plt_fig
        self.task = task
        self.group_size = group_size
        self.ts = ts
        self.out_shape = image_shape

        if annotation_type == 'default':
            self.offsets, self.data = self.process_data(data, window_len, self.group_size)
        elif annotation_type == 'change':
            self.offsets, self.data = self.process_data_change_annotation(data, window_len, offset, predictive=self.task == "predictive", group_size=self.group_size)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")

        # Get the true window length
        self.window_len = self.window_len // self.group_size

        if plt_fig:
            if not os.path.exists(f".cache/{name}_plt_cache.npy"):
                self.plt_cache = self.cache_plt()
                np.save(f".cache/{name}_plt_cache.npy", self.plt_cache)
            else:
                print("Loading plt cache")
                self.plt_cache = np.load(f".cache/{name}_plt_cache.npy")

    def cache_plt(self):
        h = self.p_quant * self.interpolation_factor
        w = self.interpolation_factor * (self.window_len * self.enlarge_factor + self.space_between * (self.window_len - 1))
        out = np.empty((len(self), 3, h, w), dtype=np.uint8)
        for idx in tqdm(range(len(self)), desc="Rendering plt images"):
            # Get coordinates
            chart_idx = np.argmin(idx > self.offsets[1:])
            i = idx - self.offsets[chart_idx] - 1
            # Return the processed sample and the label
            chart, labels = self.data[chart_idx]
            window = chart[i:i + self.window_len]
            out[idx] = self.process_plt(window, (h, w))

        return out

    @staticmethod
    def group(chart: np.ndarray, group_size: int):
        """
        Group the chart into groups of size group_size (OHLCV format). The length of the chart must be divisible by the
        group_size.
        :param chart: The chart array
        :param group_size: The number of point that are grouped together to make a single new point.
        :return:
        """
        if chart.shape[0] % group_size != 0:
            raise ValueError(f"Chart length ({chart.shape[0]}) is not divisible by group_size ({group_size})")
        a = chart[:, :5].reshape((-1, group_size, 5))
        ohlc = np.column_stack((
            a[:, 0, 0],  # Open: first element of each group
            a[:, :, 1].max(axis=1),  # High: maximum of each group
            a[:, :, 2].min(axis=1),  # Low: minimum of each group
            a[:, -1, 3],  # Close: last element of each group
            a[:, :, 4].sum(axis=1)  # Volume: last element of each group
        ))
        return ohlc

    @staticmethod
    def sample_data(data: Dict[str, pd.DataFrame], tickers: Set[str]) -> Dict[str, pd.DataFrame]:
        """
        Sample a fraction of the given data. Sample across stocks (Ex: keep 42 stocks out of 400)
        :param data: The data to be sampled from
        :param frac: The fraction of the original data to keep
        :param seed: The random seed to use. If None, no random seed is set.  NOTE: Setting the random seed here will
        set the random seed globally
        :return: The sampled data
        """
        return {ticker: data[ticker] for ticker in tickers}


    def process_data(self, data: Dict[str, pd.DataFrame], window_len, group_size: int = 1) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """
        Process the data into the format we want and generate the index
        :param data: The raw data to process
        :param window_len: The length of the window
        :return: The index, the processed data
        """
        offsets = [-1]
        out_data = []
        effective_window_len = window_len // group_size
        for name, chart in tqdm(data.items()):
            chart = chart[["Open", "High", "Low", "Close", "Volume"]]
            if len(chart) > window_len + 4 * WINDOW_SIZE + 1:
                chart = chart.values
                if group_size > 1:
                    exploitable_len = (len(chart) // group_size) * group_size
                    chart = self.group(chart[-exploitable_len:], group_size)
                offsets.append(len(chart) - effective_window_len - 4 * WINDOW_SIZE + 1)
                out_data.append((torch.from_numpy(chart[2 * WINDOW_SIZE:-2 * WINDOW_SIZE]),
                                 annotate_tickers(chart, WINDOW_SIZE)[2 * WINDOW_SIZE:-2 * WINDOW_SIZE]))
        offsets.append(offsets[-1] + 1)    # To avoid an overflow in the indexing algorithm where all offset are passed
        offsets = np.cumsum(offsets)
        return offsets, out_data

    def process_data_change_annotation(self, data: Dict[str, pd.DataFrame], window_len, offset: int = 1,
                                       predictive: bool = True, group_size: int = 1) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """
        Process the data into the format we want and generate the index
        :param data: The raw data to process
        :param window_len: The length of the window
        :param predictive: If True, the task is to predict the future
        :param group_size: The number of point that are grouped together to make a single new point. (Used for downsampling)
        :return: The index, the processed data
        """
        if not predictive:
            print("PERFORMING AUXILIARY TASK: Not predictive")
        offsets = [-1]
        out_data = []
        effective_window_len = window_len // group_size
        for name, chart in tqdm(data.items()):
            chart = chart[["Open", "High", "Low", "Close", "Volume"]]
            if len(chart) > window_len + offset + 1:
                chart = chart.values
                if group_size > 1:
                    exploitable_len = (len(chart) // group_size) * group_size
                    chart = self.group(chart[-exploitable_len:], group_size)
                offsets.append(len(chart) - effective_window_len - offset + 1)
                out_data.append((torch.from_numpy(chart[:-offset]),
                                 annotate_chart_change(chart, offset, predictive=predictive)[:-offset]))
        offsets.append(offsets[-1] + 1)
        offsets = np.cumsum(offsets)
        return offsets, out_data

    def __getitem__(self, idx):
        # Get coordinates
        chart_idx = np.argmin(idx > self.offsets[1:])
        i = idx - self.offsets[chart_idx] - 1
        # Return the processed sample and the label
        chart, labels = self.data[chart_idx]
        window = chart[i:i + self.window_len] # Shape(T, 5)
        label = torch.tensor(labels[i + self.window_len - 1]).long()
        if self.task == "count":
            label = self.get_count_label(window)
        if self.plt_fig:
            img = self.plt_cache[idx]
            img = torch.from_numpy(img).float() / 255
            return img, label
        elif self.ts:
            first_idx = self.first_non_null(window)
            starts = window[first_idx, np.arange(5)]
            starts[first_idx == -1] = 1.  # Features that are all zeros will be divided by 1 to avoid division by zeros
            norm_window = window / starts[None, :]   # Shape(T, 5) / Shape(1, 5)
            assert (norm_window >= 0).all(), f"Found negative values in the window at index {idx}"
            # To reduce the effect of extremes values, we take to log10
            processed = torch.log10(norm_window + 1e-4)
            return processed.float(), label
        else:
            return (self.process(window, self.p_quant, self.mode, self.space_between, self.enlarge_factor, self.interpolation_factor, out_shape=self.out_shape),
                label)

    @staticmethod
    def first_non_null(x):
        """
        Get the first index of x non-null along columns. If all the elements of a given columns are null, it returns -1.
        :param x: Array of shape(T, F) where F correspond to the number of features and T, the number of timestamps
        :return: Array of shape(F, )
        """
        first_idx = np.argmin((x == 0), axis=0)
        n_features = x.shape[1]
        first_idx[x[first_idx, np.arange(n_features)] == 0] = -1
        return first_idx


    @staticmethod
    def get_count_label(chart):
        """
        Compute the number of green bar on the total number of bars. If there is more green bars than red ones, the
        label is 1. 0 otherwise
        :param chart: The chart shape(N, 5)
        :return: A long 0 or 1
        """
        green = chart[:, 0] <= chart[:, 3] # Candle is green if the open price is smaller or equal to the close price
        ratio = green.int().sum() / len(chart)
        return (ratio > 0.5).long()

    @staticmethod
    def process(window: torch.Tensor, p_quant: int, mode: str, space_between: int, enlarge_factor: int, interpolation_factor: int, out_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
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
        elif out_shape is not None:
            image = F.interpolate(image.unsqueeze(0), size=out_shape, mode='nearest').squeeze(0)
        return image.float()

    @staticmethod
    def process_plt(window: torch.Tensor, out_size: Tuple[int, int]) -> npt.NDArray[np.uint8]:
        """
        Render the window chart as an image using matplotlib
        :param window: The window to process
        :param out_size: The size of the output image
        :return: The image, the label
        """
        image = gen_plt_image(window.numpy())
        image = F.interpolate(image.unsqueeze(0), size=out_size, mode='bilinear').squeeze(0)
        return (image.numpy() * 255).astype(np.uint8)

    def __len__(self):
        return self.offsets[-2] + 1   # Last one is just padding to avoid overflow in indexing algorithm

class PltDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: torch.Tensor):
        super().__init__()
        assert len(images) == len(labels), f"There must be the same number if images than labels. Got {len(images)} images and {len(labels)} labels"
        self.data = images
        self.labels = labels

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = img.float() / 255
        return img, label

    def __len__(self):
        return len(self.data)

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

def make_plt_dataloader(config, pipe: DataPipe, start, end, test_size: float = 0.2, val_size: float = 0.1,
                    fract: float = 1.):
    # Step 1: Fetch the data
    data: Dict[str, pd.DataFrame] = pipe.get(start, end)
    # Sample a fraction of the data if necessary
    if fract < 1.:
        tickers = sample_tickers(data, fract, seed=config["data"]["random_seed"])
        data = {ticker: data[ticker] for ticker in tickers}

    # Step 2: Make a temporary dataset
    # This will generate and cache the plt figures as .cache/whole_plt_cache.npy
    tmp_ds = ImageDataset(data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                            window_len=config["data"]["window_len"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type='change',
                            offset=config["data"]["offset"],
                            plt_fig=True,
                            name="whole",
                            task='predictive',
                            group_size=config["data"]["group_size"],
                            ts=False
                            )
    labels = torch.empty(len(tmp_ds), dtype=torch.long)
    h = tmp_ds.p_quant * tmp_ds.interpolation_factor
    w = tmp_ds.interpolation_factor * (tmp_ds.window_len * tmp_ds.enlarge_factor + tmp_ds.space_between * (tmp_ds.window_len - 1))
    images = torch.empty((len(tmp_ds), 3, h, w), dtype=torch.uint8)
    for i, (img, label) in enumerate(tqdm(tmp_ds, desc="Acquiring labels")):
        labels[i] = label
        images[i] = img * 255

    # Step 3: Split the data
    indices = np.arange(len(images))
    if config["data"]["random_seed"] is not None:
        np.random.seed(config["data"]["random_seed"])
    np.random.shuffle(indices)
    train_end = int((1 - test_size - val_size) * len(images))
    val_end = int((1-test_size) * len(images))

    X_train, y_train = images[indices[:train_end]], labels[indices[:train_end]]
    X_val, y_val = images[indices[train_end:val_end]], labels[indices[train_end:val_end]]
    X_test, y_test = images[indices[val_end:]], labels[indices[val_end:]]

    # Step 4: Make the real datasets
    train_ds = PltDataset(X_train, y_train)
    val_ds = PltDataset(X_val, y_val)
    test_ds = PltDataset(X_test, y_test)

    # Step 5: Initialize the dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"],
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)
    val_dataloader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)
    test_dataloader = DataLoader(test_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)

    return train_dataloader, val_dataloader, test_dataloader

def make_dataloader(config, pipe: DataPipe, start: datetime, train_end: datetime, val_end: datetime, test_end: datetime,
                    fract: float = 1., annotation_type: str = "default", task: str = "predictive", ts: bool = False,
                    image_shape: Optional[Tuple[int, int]] = None, split_method: str = "time"):
    # Step 1: Fetch the data
    data: Dict[str, pd.DataFrame] = pipe.get(start, test_end)

    # Step 2: Split the data
    if split_method == "time":
        train_data = split_data(data, start, train_end)
        val_data = split_data(data, train_end + timedelta(days=1), val_end)
        test_data = split_data(data, val_end + timedelta(days=1), test_end)

        if fract < 1.:
            tickers = sample_tickers(train_data, fract, seed=config["data"]["random_seed"])
            train_data = {ticker: train_data[ticker] for ticker in tickers}
            val_data = {ticker: val_data[ticker] for ticker in tickers}
            test_data = {ticker: test_data[ticker] for ticker in tickers}

    elif split_method == "stocks":
        print("Splitting along stocks")
        if fract < 1.0:
            tickers = sample_tickers(data, fract, seed=config["data"]["random_seed"])
            data = {ticker: data[ticker] for ticker in tickers}
        tickers = list(data.keys())
        random.seed(config["data"]["random_seed"]) if config["data"]["random_seed"] is not None else None
        random.shuffle(tickers)
        train_tickers = tickers[:int(0.8 * len(tickers))]
        val_tickers = tickers[int(0.8 * len(tickers)):]
        test_tickers = tickers[int(0.8 * len(tickers)):]
        train_data = {ticker: data[ticker].copy() for ticker in train_tickers}
        val_data = {ticker: data[ticker].copy() for ticker in val_tickers}
        test_data = {ticker: data[ticker].copy() for ticker in test_tickers}
    else:
        raise ValueError(f"Invalid split method: {split_method}")


    # Step 3: Make the datasets
    train_ds = ImageDataset(train_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                            window_len=config["data"]["window_len"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type=annotation_type,
                            offset=config["data"]["offset"],
                            plt_fig=config["data"]["plt_fig"],
                            name="train",
                            task=task,
                            group_size=config["data"]["group_size"],
                            ts=ts,
                            image_shape=image_shape
                            )
    val_ds = ImageDataset(val_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                          window_len=config["data"]["window_len"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type=annotation_type,
                            offset=config["data"]["offset"],
                            plt_fig=config["data"]["plt_fig"],
                            name="val",
                            task=task,
                            group_size=config["data"]["group_size"],
                            ts=ts,
                            image_shape=image_shape
                          )
    test_ds = ImageDataset(test_data, mode=config["data"]["mode"], p_quant=config["data"]["p_quant"],
                           window_len=config["data"]["window_len"],
                            space_between=config["data"]["space_between"],
                            enlarge_factor=config["data"]["enlarge_factor"],
                            interpolation_factor=config["data"]["interpolation_factor"],
                            annotation_type=annotation_type,
                            offset=config["data"]["offset"],
                            plt_fig=config["data"]["plt_fig"],
                            name="test",
                            task=task,
                            group_size=config["data"]["group_size"],
                            ts=ts,
                            image_shape=image_shape
                           )

    # Step 4: Initialize the dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"],
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)
    val_dataloader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)
    test_dataloader = DataLoader(test_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                                  num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute
    from pipes import Finviz, RmTz, FromFile
    from datetime import datetime

    pipe = FromFile("tw50.json") | JSONCache() | \
           FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
           Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
    pipe.set_id(10)
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    data = split_data(data, datetime(2000, 1, 1), datetime(2016, 12, 31))
    # plt.plot(data["AAPL"].index, data["AAPL"]["Close"])
    # labels_up = data["AAPL"]["Close"].values.copy()
    # labels_up[annot == 0] = np.nan
    # labels_down = data["AAPL"]["Close"].values.copy()
    # labels_down[annot == 1] = np.nan
    # plt.scatter(data["AAPL"].index, labels_up, color='green')
    # plt.scatter(data["AAPL"].index, labels_down, color='red')
    # plt.show()
    dataset = ImageDataset(data, mode='basic', p_quant=57, window_len=20, space_between=1, enlarge_factor=2, interpolation_factor=1, ts=True)
    # print(dataset[11214])
    print(len(dataset))
    print(dataset[122089])
    # for i, (image, label) in enumerate(tqdm(dataset)):
    #     pass
        # if i % 256 == 0:
        #     plt.imshow(image.permute(1, 2, 0), aspect=1)
        #     plt.tight_layout()
        #     plt.axis("off")
        #     plt.show()
        #     # plt.savefig("/Users/alavertu/Downloads/viz_repr_no_spectrum.png", dpi=400)
# python main.py --experiment=experiment2 --config=configs/paper_2.yml --dataset="small" --config.data.plt_fig=True --sample_inputs --split="random" --watch=accuracy --debug
