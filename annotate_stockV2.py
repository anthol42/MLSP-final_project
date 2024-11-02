from backtest.data import FetchCharts, Cache
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import find_peaks

pd.set_option('display.max_rows', 200)

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
    Annoter les prix de fermeture
    :param df: jeu de données contenant les prix de fermeture et les dates
    :return: Jeu de données annoté et information pour les graphiques
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

if __name__ == '__main__':
    # Fetch de stock data of NVIDIA over 20 years
    TICKERS = ['AAPL']
    pipe = FetchCharts(TICKERS) | Cache()
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    trend, plt_curve = annotate_tickers(df=data[TICKERS[0]])
    # with pd.option_context('display.max_rows', None, 'display.max_columns',
    #                        None):
    #     print(trend)
    print(trend[trend['Anno'] == 'NEUTRAL'])
    trend_up = plt_curve['close_price'].copy()
    trend_down = plt_curve['close_price'].copy()
    trend_neutral = plt_curve['close_price'].copy()

    trend_up[trend['Anno'] != 'UP'] = np.nan
    trend_down[trend['Anno'] != 'DOWN'] = np.nan
    trend_neutral[trend['Anno'] != 'NEUTRAL'] = np.nan

    # peaks, _ = find_peaks(plt_curve['close_price'], distance=100)
    # plot the mean stock price
    plt.figure(figsize=(12, 6))
    plt.plot(plt_curve['moving_average'], label = 'Mean Curve')
    plt.plot(trend_down, color='#a80000')
    plt.plot(trend_up, color='#14a800')
    plt.plot(trend_neutral, color='#808080')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    plt.title('Overlay of stock and mean set window prices')
    plt.show()