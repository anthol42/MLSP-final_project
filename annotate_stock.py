import sys
from backtest.data import FetchCharts, Cache
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

pd.set_option('display.max_rows', 200)

# Fetch de stock data of NVIDIA over 20 years
TICKERS = ['NVDA']
pipe = FetchCharts(TICKERS) | Cache()
data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))

### Generate the mean curve
WINDOW_SIZE = 14

# Get close prices
close_price = data['NVDA']['Close'].iloc[-100:-55]
close_price.index = pd.to_datetime(close_price.index).date

# Generate all windows
moving_average = close_price.rolling(WINDOW_SIZE).mean()
moving_average = moving_average.dropna()
close_price = close_price[moving_average.index]

# Split the data point on which are over or under the mean curve
over = close_price >= moving_average

t = np.array(over, dtype=np.int8)
i = 0
inflection_pts = []
while(i < len(t)):
    val = t[i]

    if(val == 0):
        idx = np.argmax(t[i:]) + i if np.all(~t[i:]) else len(t)
        arg = close_price.iloc[i:idx].argmin() + i + 1
        t[i:arg] = 0
        t[arg:idx] = 1

    else:
        idx = np.argmin(t[i:]) + i if not np.all(t[i:]) else len(t)
        arg = close_price.iloc[i:idx].argmax() + i + 1
        t[i:arg] = 1
        t[arg:idx] = 0

    inflection_pts.append(arg)
    i = idx

trend_up = close_price.copy()
trend_down = close_price.copy()

#inflection_pts = [close_price.index[i] for i in inflection_pts]
trend_up[(t == 0)] = np.nan
trend_down[(t == 1)] = np.nan

peaks, _ = find_peaks(close_price, distance=100)
# plot the mean stock price
plt.figure(figsize=(12, 6))
plt.plot(moving_average, label = 'Mean Curve')
plt.plot(trend_down, color='#a80000')
plt.plot(trend_up, color = '#14a800')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.title('Overlay of stock and mean set window prices')
plt.show()