import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from tqdm import tqdm

def load_annotations(filename: str) -> Dict[str, pd.DataFrame]:
    annotations = {}
    with pd.HDFStore(filename, 'r') as store:
        for key in store.keys():
            annotations[key[1:]] = store[key]
    return annotations

def plot_chart(chart: pd.DataFrame, show: bool = True):
    price = chart["Close"]
    annot = chart["Anno"]
    buy = np.full_like(price, np.nan)
    sell = np.full_like(price, np.nan)
    hold = np.full_like(price, np.nan)

    buy[annot == 1] = price[annot == 1]
    sell[annot == 0] = price[annot == 0]
    hold[annot == 2] = price[annot == 2]

    plt.plot(price)
    plt.scatter(chart.index, buy, color='green')
    plt.scatter(chart.index, sell, color='red')
    plt.scatter(chart.index, hold, color='grey')
    if show:
        plt.show()

if __name__ == "__main__":
    data = load_annotations("evaluation/annotations.anno")
    plot_chart(data["BLDR"], show=False)

    out = {}
    for ticker, chart in tqdm(data.items(), desc="Computing performances"):  # data.items(): #
        o, c, index = chart['Open'].values, chart['Close'].values, chart.index
        label = chart['Anno'].values

        state = label[0]
        buy_price = None
        buy_time = None
        transactions = []
        durations = []
        for i in range(len(label) - 1):
            if label[i] != state:
                if label[i] == 1 and buy_price is None:  # Buy
                    buy_price = o[i + 1]
                    buy_time = index[i + 1]
                    state = 1.
                elif (label[i] == 0 or label[i] == 2) and buy_price is not None:  # Sell
                    sell_price = o[i + 1]
                    sell_time = index[i + 1]
                    # print(f"Buy: {buy_price}, Sell: {sell_price}")
                    if buy_price != 0.:
                        transactions.append((sell_price - buy_price) / buy_price)
                        durations.append((sell_time - buy_time).days)
                    buy_price = None
                    state = label[i]
            else:
                continue
        transactions = np.array(transactions)
        durations = np.array(durations)
        out[ticker] = (transactions, durations)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    transactions = [t for ticker, (transaction_list, duration_list) in out.items() for t in transaction_list]
    quantile = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
    percentile = np.percentile(a=transactions, q=quantile)

    for i in range(len(quantile)):
        print(f'Gains Quantile:{quantile[i]} -> {percentile[i]}')
    plt.hist(transactions, bins=100)
    plt.xlim(-0.25, 0.33)
    plt.title("Relative gains of trades")
    for p in percentile:
        plt.axvline(x=p, color='black', linestyle="--", lw=1)

    plt.subplot(1, 2, 2)
    durations = [t for ticker, (transaction_list, duration_list) in out.items() for t in duration_list]
    quantile = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
    percentile = np.percentile(a=durations, q=quantile)

    for i in range(len(quantile)):
        print(f'Duration Quantile:{quantile[i]} -> {percentile[i]}')
    plt.hist(durations, bins=100)
    plt.xlim(0, 60)
    plt.title("Duration of trades")
    for p in percentile:
        plt.axvline(x=p, color='black', linestyle="--", lw=1)

    # plt.subplot(2, 2, 3)
    # plt.scatter(durations, transactions)
    # plt.tight_layout()
    plt.show()