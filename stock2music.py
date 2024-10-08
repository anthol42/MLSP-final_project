import pandas as pd
from backtest.data import FetchCharts, Cache
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import Spectrogram
from scipy.signal import spectrogram
import pickle
# from torch_pitch_shift import pitch_shift
TICKERS = ['AAPL', 'NVDA', 'META']
pipe = FetchCharts(TICKERS) | Cache()

data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))

chart = data["META"]
def window2note1(window, length: float):
    """
    Convert a timeseries to a sound.
    :param window: The time series.  pd.Series or np.array
    :param length: The length in second of the sound (MUST be smaller than one:  < 1 )
    :return: The sound array
    """
    # First part will generate 1 second of audio
    window = 2 * (window - window.min()) / (window.max() - window.min()) - 1
    x = window.repeat(44_100 // len(window[1:])).diff(1)
    return x[:int(length * len(x))]

def melodie(ts, window_size: int):
    notes = []
    ts = torch.from_numpy(ts.values)
    for i in range(1, len(ts) - window_size):
        window = ts[i:window_size + i + 1] / ts[i]
        note = window2note1(window, 0.5)
        # note = torch.ones(len(ts))
        fft = torch.fft.rfft(note).real
        # Low pass filter
        # fft[1200:] = 0
        # fft[1080:1120] = 0
        # fft[800:900] = 0
        fft[fft < 50] = 0
        note = torch.fft.irfft(fft)
        notes.append(note)
    return np.concatenate(notes)
def make_dataset(chart: pd.DataFrame, window_size: int = 100, n_notes: int = 50):
    mid = (chart["High"] + chart["Low"]) / 2
    change = mid.pct_change()
    dataset = []
    for i in tqdm(range(len(change) - window_size - n_notes)):
        music = melodie(mid[i:i + window_size + n_notes], n_notes)
        # label = mid[i + window_size + n_notes - 1] > mid[i + window_size + n_notes - 2]# Is today greater than yesterday
        # dataset.append((music, label))

    return dataset

dataset = make_dataset(chart, n_notes=50)
# mid = (chart["High"] + chart["Low"]) / 2
# music = melodie(mid[300:500], 50)
# plt.imshow(Spectrogram(n_fft=512)(torch.from_numpy(music)).log2(), aspect=60)
# plt.show()
# plt.specgram(music, Fs=44_100, NFFT=512)
# plt.show()
# sd.play(music, 44_100)
# sd.wait()
print("Done")
