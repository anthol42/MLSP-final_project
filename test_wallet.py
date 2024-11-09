import math
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from annotate_stock import annotate_tickers
from backtest.data import FetchCharts, Cache, JSONCache, FilterNoneCharts, CausalImpute
from datetime import datetime
from deep.pipes.finviz import Finviz
from tqdm import tqdm

# Fetch S&P500 stock data
def fetch_snp500():
    pipe = Finviz(
        "https://finviz.com/screener.ashx?v=111&f=idx_sp500%2Cipodate_more5",
        True) | JSONCache() | FetchCharts(progress=True, throttle=1.,
                                          auto_adjust=False) | Cache() | FilterNoneCharts() | CausalImpute()
    pipe.set_id(0)
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    return data

data = fetch_snp500()
buy_in, sell_out, prices, price_date, buy_date, sell_date = [], [], [], [], [], []
money_tickers, transaction_ticker = [], []

for tick in tqdm(data):
    # Starting wallet size
    money = 1_000
    stock2sell = False

    # Fetch annotations for a givens stock
    price, _ = annotate_tickers(df=data[tick])
    price['Anno_PreviousDay'] = price['Anno'].shift(1)
    # Reduce dataset size for efficiency
    buy_row = (price['Anno_PreviousDay'] == 'DOWN') & (price['Anno'] == 'UP')
    sell_row = (price['Anno_PreviousDay'] == 'UP') & (price['Anno'] == 'DOWN')
    price = price[buy_row | sell_row]

    for idx in range(price.shape[0]):
        today = price.iloc[idx]
        price_date.append(today.name)
        prices.append(money)
        money_tickers.append(tick)

        # Optimize transaction to buy
        if today['Anno_PreviousDay'] == 'UP':
            if stock2sell:
                sell = today['Close']
                money += sell
                # Append buying information
                buy_in.append(buy)
                buy_date.append(tmp_buy_date)
                # Append selling information
                sell_date.append(today.name)
                sell_out.append(sell)
                transaction_ticker.append(today.name)
                # No more stock to sell
                stock2sell = False

        # Optimize transaction to sell
        elif today['Anno_PreviousDay'] == 'DOWN':
            if not stock2sell:
                buy = today['Close']
                money -= buy
                tmp_buy_date = today.name
                stock2sell = True

    # Supprimer les dernières achats si non matérialisées en vente
    if len(buy_date) != len(sell_date):
        buy_date.pop(len(buy_date) - 1)
        buy_in.pop(len(buy_in) - 1)

holding_time = np.array(sell_date) - np.array(buy_date)
money_pourcentage = np.array(sell_out) / np.array(buy_in)

transaction = pd.DataFrame({
    'Tickers': transaction_ticker,
    'Buy': buy_in,
    'Sell': sell_out,
    'Money_pourcentage': money_pourcentage,
    'buy_date': buy_date,
    'sell_date': sell_date,
    'Holding_time': holding_time
})

money_ticker = pd.DataFrame({
    'Tickers': money_tickers,
    'Date': price_date,
    'Price': prices
})

# Plot the bins of Holding_time distribution
bins = int(math.sqrt(len(transaction['Holding_time'])))
hist_data = transaction['Holding_time'].dt.days
counts, bin_edges, _ = plt.hist(hist_data, bins=bins)
plt.xlim([0, np.max(transaction['Holding_time'].dt.days) + 1])
plt.title('Histogramme des durées de maintien d\'une action ')

# Add  quantile axis & annotation
quantile = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
percentile = np.percentile(a=holding_time, q=quantile)

# for i in range(len(quantile)):
#     print(f'quantile:{quantile[i]} -> {percentile[i]}')

p = [10, 50, 75, 90] # Percentile values
q = [percentile[1], percentile[3], percentile[4], percentile[5]] # Quantile values
y = [np.max(counts) - i * 15_000 for i in range(len(p))] # Y values for plotting

bbox = dict(boxstyle="round", edgecolor="black", facecolor="none")
arrowprops = dict(arrowstyle="->", color="black")

for i, j, k in zip(q, y, p):
    plt.axvline(x=i, color='gray', linestyle="dotted", lw=1, zorder=0)
    plt.annotate(
        text=('Q'+str(k)+': '+str(i)+' jour'), xy=(i, j), xytext=(1 * 72, 0),
        textcoords='offset points', bbox=bbox, arrowprops=arrowprops
    )

# Show plot
plt.show()

# Group the data by 'Tickers'
groups = money_ticker.groupby('Tickers')

# Plot each ticker's data as a separate line
plt.figure(figsize=(12, 6))
for ticker, group in groups:
    plt.plot(group['Date'], group['Price'], label=ticker)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price over Time by Ticker")
plt.legend(loc='upper left', ncol=3, fontsize='small')  # Adjust legend for readability
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()