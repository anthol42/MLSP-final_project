import numpy as np
import matplotlib.pyplot as plt
from annotate_stockV2 import annotate_tickers
from backtest.data import FetchCharts, Cache, JSONCache, FilterNoneCharts, CausalImpute
from datetime import datetime
from deep.pipes.finviz import Finviz
from tqdm import tqdm
from datetime import datetime, timedelta
def fetch_snp500():
    pipe = Finviz(
        "https://finviz.com/screener.ashx?v=111&f=idx_sp500%2Cipodate_more5",
        True) | JSONCache() | FetchCharts(progress=True, throttle=1.,
                                          auto_adjust=False) | Cache() | FilterNoneCharts() | CausalImpute()
    pipe.set_id(0)
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    return data

data = fetch_snp500()
out = {}
for ticker, chart in tqdm(data.items(), desc="Computing performances"): # data.items(): #
    o, c, index = chart['Open'].values, chart['Close'].values, chart.index
    s = datetime.now()
    label = annotate_tickers(chart.values)

    state = label[0]
    buy_price = None
    buy_time = None
    transactions = []
    durations = []
    for i in range(len(label) - 1):
        if label[i] != state:
            if label[i] == 1 and buy_price is None: # Buy
                buy_price = o[i + 1]
                buy_time = index[i + 1]
                state = 1.
            elif (label[i] == 0 or label[i] == 2) and buy_price is not None: # Sell
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
    # Color the curve according to the label
    # print(np.sum(transactions > 0) / len(transactions))
    # print(transactions)
    # print(durations)
    # red = c.copy()
    # red[label != 0] = np.nan
    # green = c.copy()
    # green[label != 1] = np.nan
    # grey = c.copy()
    # grey[label != 2] = np.nan
    # plt.plot(index, c, label=ticker, color="black")
    # plt.scatter(index, red, color='r')
    # plt.scatter(index, green, color='g')
    # plt.scatter(index, grey, color='grey')
    # plt.show()
    # break
    out[ticker] = (transactions, durations)

# Flatten all transactions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
transactions = [t for ticker, (transaction_list, duration_list) in out.items() for t in transaction_list]
quantile = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
percentile = np.percentile(a=transactions, q=quantile)

for i in range(len(quantile)):
    print(f'Gains Quantile:{quantile[i]} -> {percentile[i]}')
plt.hist(transactions, bins=100)
for p in percentile:
    plt.axvline(x=p, color='black', linestyle="--", lw=1)

plt.subplot(1, 2, 2)
durations = [t for ticker, (transaction_list, duration_list) in out.items() for t in duration_list]
quantile = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
percentile = np.percentile(a=durations, q=quantile)

for i in range(len(quantile)):
    print(f'Duration Quantile:{quantile[i]} -> {percentile[i]}')
plt.hist(durations, bins=100)
for p in percentile:
    plt.axvline(x=p, color='black', linestyle="--", lw=1)
plt.show()