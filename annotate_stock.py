from backtest.data import FetchCharts, Cache
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

pd.set_option('display.max_rows', 200)


def add_neutral(df, pct : float):
    """
    Ajouter l'annotation "Neutral" au jeu de données.
    :param df: Prix de fermeture et annotation préliminaire des données.
    :param pct: Pourcentage de variation en dessous duquel on considère neutre.
    :return: Jeu de données ayant les annotations mise à jour.
    """
    chunk_size = 4

    for i in range(df.shape[0] - chunk_size):
        # Extract a chunk of 4 rows
        chunk = df.iloc[i:i + chunk_size]

        # Calculate price variations for consecutive rows in the chunk
        price_diffs = chunk['Close'].pct_change().abs().iloc[1:] > pct

        # Apply 'NEUTRAL' if both conditions are met
        if price_diffs.all():
            df.loc[chunk.index[0], 'Anno'] = 'NEUTRAL'
            df.loc[chunk.index[1], 'Anno'] = 'NEUTRAL'

    return df

def annotate_tickers(df):
    """
    Annoter les prix de fermeture
    :param df: jeu de données contenant les prix de fermeture et les dates
    :return: Jeu de données annoté et information pour les graphiques
    """
    WINDOW_SIZE = 14

    # Get close prices
    close_price = df['Close']
    close_price.index = pd.to_datetime(close_price.index).date

    # Generate all windows
    moving_average = close_price.rolling(WINDOW_SIZE).mean()
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
            if(i != idx):
                arg = close_price.iloc[i:idx].argmin() + i + 1
                t[i:arg] = 0
                t[arg:idx] = 1

        else:
            idx = np.argmin(t[i:]) + i if not np.all(t[i:]) else len(t)
            arg = close_price.iloc[i:idx].argmax() + i + 1
            t[i:arg] = 1
            t[arg:idx] = 0

        inflection_pts.append(arg)
        if i != idx: i = idx
        else: break

    annotated_price = pd.DataFrame({
        'Close': close_price,
        'Anno': t.tolist()
    })

    # Indiquer les noms pour faciliter la compréhension
    annotated_price['Anno'] = annotated_price['Anno'].replace({1: 'UP', 0: 'DOWN'})
    # Mise à jour des annotations
    annotated_price = add_neutral(df=annotated_price, pct=0.02)
    # Dictionnaire des informations pour les graphiques
    plt_curve = {'close_price': close_price, 'moving_average': moving_average}

    return annotated_price, plt_curve

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