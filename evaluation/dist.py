import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from tqdm import tqdm
import h5py

def load_annotations(filename: str) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    annotations = {}
    # Open the HDF5 file in read mode
    with h5py.File(filename, 'r') as f:
        # Iterate over each group in the file
        for key in f.keys():
            group = f[key]

            # Load the DataFrame (as a numpy array) and convert it back to DataFrame
            df_data = group['dataframe'][:]
            df_index = group['index'][:]
            df_index = pd.to_datetime(df_index)
            df = pd.DataFrame(df_data, index=df_index,
                              columns=[f"col{i}" for i in range(df_data.shape[1])])  # Recreate DataFrame

            # Load the numpy array
            array = group['array'][:]

            # Store the DataFrame and array as a tuple in the dictionary
            annotations[key] = (df, array)

    return annotations

def plot_chart(chart: pd.DataFrame, annot: np.ndarray, show: bool = True):
    price = chart.iloc[:, 3]
    hard_annot = np.argmax(annot, axis=1).astype(float)
    hard_annot[np.isnan(annot).any(axis=1)] = np.nan
    buy = np.full_like(price, np.nan)
    sell = np.full_like(price, np.nan)
    hold = np.full_like(price, np.nan)

    buy[hard_annot == 1] = price[hard_annot == 1]
    sell[hard_annot == 0] = price[hard_annot == 0]
    hold[hard_annot == 2] = price[hard_annot == 2]

    plt.plot(price)
    plt.scatter(chart.index, buy, color='green')
    plt.scatter(chart.index, sell, color='red')
    plt.scatter(chart.index, hold, color='grey')
    if show:
        plt.show()

if __name__ == "__main__":
    data = load_annotations("annotations/E_s.anno")
    print(data.keys())
    plot_chart(*data["AAAU"], show=False)

    out = {}
    for ticker, (chart, annotations) in tqdm(data.items(), desc="Computing performances"):  # data.items(): #
        o, c, index = chart.values[:, 0], chart.values[:, 3], chart.index
        print(index)
        label = np.argmax(annotations, axis=1).astype(float)
        label[np.isnan(annotations).any(axis=1)] = np.nan

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