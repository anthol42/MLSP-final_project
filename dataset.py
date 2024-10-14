from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def gen_image(chart: np.ndarray, p_quant = 128) -> torch.Tensor:
    """
    Generate an image from a chart
    :param chart: A tensor in the format [Open, High, Low, Close, Volume, LABEL]
    :param p_quant: The quantification size of the price
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
    mask_down = np.log(1 + np.exp((quant_low[None, :] - np.arange(p_quant)[:, None]) / p_quant)) - np.log(2)
    mask_down[mask_down < 0] = 0
    mask_up = np.log(1 + np.exp((np.arange(p_quant)[:, None] - quant_high[None, :]) / p_quant)) - np.log(2)
    mask_up[mask_up < 0] = 0
    mask = mask_up + mask_down
    mask = mask / (mask.max(axis=0) + 1e-8)

    # Step 4: Get volume
    volume = np.tile(np.log(1 + chart[:, 4][None, :] / (np.max(chart[:, 4] + 1))), (p_quant, 1))
    volume_image = plt.cm.coolwarm(volume)[:, : :, :3]    # Or plasma
    volume_image = mask[:, :, None] * volume_image

    # Step 5: Combine the images and clamp 0-1
    # image = np.clip(image + volume_image, 0, 1)

    # Flip vertically the image
    image = image[::-1, :, :].copy()
    return torch.from_numpy(image)


class ImageDataset(Dataset):
    def __init__(self, data: Dict[str, pd.DataFrame], p_quant: int = 128, window_len: int = 256):
        self.p_quant = p_quant
        self.window_len = window_len
        self.offsets, self.data = self.process_data(data, window_len)

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
            # TODO: Add the label algorithm and add a column to the chart tensor
            offsets.append(len(chart) - window_len + 1)
            out_data.append(torch.from_numpy(chart.values))
        offsets.append(offsets[-1] + 1)    # To avoid an overflow in the indexing algorithm where all offset are passed
        offsets = np.cumsum(offsets)
        # offsets[1:-1] += 1
        return offsets, out_data

    def __getitem__(self, idx):
        # Get coordinates
        chart_idx = np.argmin(idx > self.offsets[1:])
        i = idx - self.offsets[chart_idx] - 1
        # Return the processed sample and the label
        window = self.data[chart_idx][i:i + self.window_len]
        return self.process(window, self.p_quant)

    @staticmethod
    def process(window: torch.Tensor, p_quant: int):
        """
        Process the window into an image and a label
        :param window: The window to process
        :param p_quant: The quantification of the price
        :return: The image, the label
        """
        # print(window.shape)
        assert len(window) == 256
        return gen_image(window.numpy(), p_quant), 0.    # TODO: Add the computed label


    def __len__(self):
        return self.offsets[-2] + 1   # Last one is just padding to avoid overflow in indexing algorithm


if __name__ == "__main__":
    from backtest.data import FetchCharts, Cache
    from datetime import datetime
    TICKERS = ['AAPL', 'NVDA', 'META', "AMZN"]
    pipe = FetchCharts(TICKERS) | Cache()
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    dataset = ImageDataset(data)
    print(len(data["AAPL"]), len(data["NVDA"]), len(data["META"]))
    # print(dataset[11214])
    print(len(dataset))
    k = 0
    for i, (image, label) in enumerate(tqdm(dataset)):
        # print(i)
        if i % 256 == 0:
            k+=1
            if k == 3:
                plt.imshow(image)
                plt.tight_layout()
                plt.axis("off")
                plt.savefig("/Users/alavertu/Downloads/viz_repr_no_spectrum.png", dpi=400)
