from backtest.data import FetchCharts, Cache
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
import math
from scipy.signal import spectrogram
# from torch_pitch_shift import pitch_shift
TICKERS = ['AAPL', 'NVDA', 'META']
pipe = FetchCharts(TICKERS) | Cache()

data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))

chart = data["META"]

# mpf.plot(chart.iloc[-100:], type='candle', volume=True, style='yahoo')
# mpf.show()

def window2note1(window, length: float):
    """
    Convert a timeseries to a sound.
    :param window: The time series.  pd.Series or np.array
    :param length: The length in second of the sound
    :return: The sound array
    """
    # First part will generate 1 second of audio
    x = torch.from_numpy(window.values).repeat((44_100 // len(window[1:]))).diff(1)
    # Norm -1 to 1
    x_fft = torch.fft.rfft(x).real
    x_fft[int(0.5 * len(x_fft)):] = 0
    x_out = torch.fft.irfft(x_fft)
    x_norm = 2 * (x_out - x_out.min()) / (x_out.max() - x_out.min()) - 1
    # return sound / np.abs(sound).max()
    if length < 1:
        return x_norm[:int(length * len(x_norm))].numpy()
    else:
        sound = x_norm.repeat(math.ceil(length))
        return sound[:int(length * len(x_norm))].numpy()

def melodie(ts, window_size: int):
    notes = []
    for i in range(1, len(ts) - window_size):
        window = ts[i:window_size + i + 1] / ts[i]
        note = window2note1(window, 0.5)
        fft = torch.fft.rfft(torch.from_numpy(note)).real
        # Low pass filter
        # fft[1200:] = 0
        # fft[1080:1120] = 0
        # fft[800:900] = 0
        fft[fft < 50] = 0
        note = torch.fft.irfft(fft)
        notes.append(note)
    return np.concatenate(notes)

ereturn = np.log10(chart["Close"] / chart["Close"].shift(1))
mid = (chart["High"] + chart["Low"]) / 2
change = mid.pct_change()
offset = 27
window = mid[offset:100 + offset] / mid[offset]
print("Playing sound")
music = melodie(mid[0:200], 100)
plt.specgram(music, Fs=44_100, NFFT=512)
plt.show()
# sd.play(music, 44_100)
# sd.wait()
print("Done")
