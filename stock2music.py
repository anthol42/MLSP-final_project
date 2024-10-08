from backtest.data import FetchCharts, Cache
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
import math
TICKERS = ['AAPL', 'NVDA', 'META']
pipe = FetchCharts(TICKERS) | Cache()

data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))

chart = data["AAPL"]

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
    # print(x)
    x_fft = torch.fft.rfft(x).real
    x_fft[int(0.5 * len(x_fft)):] = 0
    x_out = torch.fft.irfft(x_fft)
    x_norm = 2 * (x_out - x_out.min()) / (x_out.max() - x_out.min()) - 1
    # plt.plot(torch.fft.irfft(x_fft))
    # plt.show()
    # fft = torch.fft.irfft(x).real
    # x_prime = torch.fft.irfft(fft)
    # plt.plot(x)
    # plt.show()
    # sound = fft[1:].repeat(int(length * (44_100 // len(fft[1:]))))
    # return sound / np.abs(sound).max()
    if length < 1:
        return x_norm[:int(length * len(x_norm))].numpy()
    else:
        sound = x_norm.repeat(math.ceil(length))
        return sound[:int(length * len(x_norm))].numpy()

def window2note2(window, length: float):
    """
    Convert a timeseries to a sound.
    :param window: The time series.  pd.Series or np.array
    :param length: The length in second of the sound
    :return: The sound array
    """
    x = torch.from_numpy(window.values)

    # Interpolate to 44_100 / 2
    # x_int = F.interpolate(x.unsqueeze(0).unsqueeze(0), 44_100 // 2, mode='linear').squeeze(0).squeeze(0)
    x_sound = torch.fft.irfft(x)
    x_sound = F.interpolate(x_sound[1:].unsqueeze(0).unsqueeze(0), int(length * 44_100), mode='linear').squeeze(0).squeeze(0)
    # plt.plot(x_sound[1:])
    # plt.show()
    # x_fft = torch.fft.rfft(x).real
    # x_fft[int(0.5 * len(x_fft)):] = 0
    # x_norm = torch.fft.irfft(x_fft)
    return x_sound.numpy()
def melodie(ts, note_len, window_size: int):
    notes = []
    for i in range(1, len(ts) - window_size):
        window = ts[i:window_size + i + 1] / ts[i]
        note = window2note1(window, note_len[i])
        # plt.plot(note)
        # plt.show()
        fft = torch.fft.rfft(torch.from_numpy(note)).real
        # Low pass filter
        fft[1200:] = 0
        # fft[1080:1120] = 0
        # fft[800:900] = 0
        note = torch.fft.irfft(fft)
        # print(i)
        notes.append(note)
        # sd.play(notes[-1], 44_100)
        # sd.wait()
        # plt.plot(fft)
        # plt.show()
    return np.concatenate(notes)
ereturn = np.log10(chart["Close"] / chart["Close"].shift(1))
mid = (chart["High"] + chart["Low"]) / 2
change = mid.diff()
offset = 27
window = mid[offset:100 + offset] / mid[offset]
print("Playing sound")
# music = melodie(mid[:200], change[:100], 100)
# plt.plot(music)
# plt.show()
# music_fft = torch.fft.rfft(torch.from_numpy(music))
# music_fft[int(0.25 * len(music_fft)):] = 0.
# music_filtered = torch.fft.irfft(music_fft)
# music_fft[:5000] = 0.
# music = torch.fft.irfft(music_fft).numpy()
# plt.plot(music_fft.real)
# plt.show()
sd.play(music, 44_100)
sd.wait()
print("Done")
# rolling_std = ereturn.rolling(100).std()
# plt.plot(ereturn)
# plt.plot(rolling_std)
# plt.show()