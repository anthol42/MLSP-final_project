# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from backtest.data import FetchCharts, Cache
# from datetime import datetime
# from tqdm import tqdm
#
# TICKERS = ['AAPL', 'NVDA', 'META']
# pipe = FetchCharts(TICKERS) | Cache()
#
# data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
#
#
# chart = data["NVDA"] # .iloc[:256]
#
# def gen_image(chart, p_quant = 128):
#     # Step 1: Quantify prices
#     prices = np.linspace(chart["Low"].min(), chart["High"].max(), p_quant)
#     quant_open = np.argmin((chart["Open"].values - prices[:, None]) ** 2, axis=0)
#     quant_close = np.argmin((chart["Close"].values - prices[:, None]) ** 2, axis=0)
#     # Step 2: Color the image with the 'candlesticks' like representation
#     _, t_idx = np.meshgrid(np.arange(p_quant), np.arange(len(chart)))
#
#     # Positive days
#     length = 1 + quant_close - quant_open
#     length[length <= 0] = 0    # Keep only positives days
#     range_array = np.arange(np.max(length))
#     mask = range_array < length[:, None]
#     q_pos_days = (quant_open[:, None] + range_array)[mask]
#     idx_pos_days = t_idx[:, :len(range_array)][mask]
#     # Negative days
#     length = 1 + quant_open - quant_close
#     length[length < 2] = 0    # Keep only negative days
#     range_array = np.arange(np.max(length))
#     mask = range_array < length[:, None]
#     q_neg_days = (quant_close[:, None] + range_array)[mask]
#     idx_neg_days = t_idx[:, :len(range_array)][mask]
#
#     # Generate the image
#     image = np.zeros((p_quant, len(chart), 3))    # RGB
#     image[q_pos_days, idx_pos_days, 1] = 1    # Green
#     image[q_neg_days, idx_neg_days, 0] = 1    # Red
#
#     # Step 3: Make the spectrum
#     quant_high = np.argmin((chart["High"].values - prices[:, None]) ** 2, axis=0)
#     quant_low = np.argmin((chart["Low"].values - prices[:, None]) ** 2, axis=0)
#     mask_up = np.log(1 + np.exp((quant_high[None, :] - np.arange(p_quant)[:, None]) / p_quant)) - np.log(2)
#     mask_up[mask_up < 0] = 0
#     mask_down = np.log(1 + np.exp((np.arange(p_quant)[:, None] - quant_high[None, :]) / p_quant)) - np.log(2)
#     mask_down[mask_down < 0] = 0
#     mask = mask_up + mask_down
#     mask = mask / mask.max()
#
#     # Step 4: Get volume
#     volume = np.tile(np.log(1 + chart["Volume"].values[None, :] / (np.max(chart["Volume"].values + 1))), (p_quant, 1))
#     volume_image = plt.cm.seismic(volume)[:, : :, :3]
#     volume_image = mask[:, :, None] * volume_image
#
#     # Step 5: Combine the images and clamp 0-1
#     image = np.clip(image + volume_image, 0, 1)
#
#     # Flip vertically the image
#     image = image[::-1, :, :]
#     return image
# def make_dataset(chart: pd.DataFrame, window_size: int = 256, p_quant: int = 128):
#     for i in tqdm(range(len(chart) - window_size)):
#         image = gen_image(chart[i:window_size + i], p_quant)
#         if i % 256 == 0:
#             plt.imshow(image)
#             plt.show()
#         # label = mid[i + window_size + n_notes - 1] > mid[i + window_size + n_notes - 2]# Is today greater than yesterday
#         # dataset.append((music, label))
#
# make_dataset(chart)
print(np.argmin([True, False, False]))