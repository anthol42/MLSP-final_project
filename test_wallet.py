import pandas as pd
from matplotlib import ticker
from annotate_stock import annotate_tickers
from backtest.data import FetchCharts, Cache, JSONCache, FilterNoneCharts, CausalImpute
from datetime import datetime
from pipes.finviz import Finviz
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
stock2sell = False
buy_in, sell_out, prices, price_date, buy_date, sell_date = [], [], [], [], [], []
money_tickers, transaction_ticker = [], []
for tick in tqdm(data):
    # Starting wallet size
    money = 1_000

    # Fetch annotations for a givens stock
    price, _ = annotate_tickers(df=data[tick])
    price['Anno_NextDay'] = price['Anno'].shift(-1)


    for idx in range(len(price)):
        today = price.iloc[idx]
        price_date.append(today)
        prices.append(money)
        money_tickers.append(tick)

        # Optimize transaction to buy
        if today['Anno'] == 'UP' and today['Anno_NextDay'] == 'DOWN':
            if stock2sell:
                sell = today['Close']
                money += sell
                sell_date.append(today.name)
                sell_out.append(sell)
                transaction_ticker.append(today)
                stock2sell = False

        # Optimize transaction to sell
        elif today['Anno'] == 'DOWN' and today['Anno_NextDay'] == 'UP':
            outcome = money - today['Close']
            if outcome > 0:
                buy = today['Close']
                money -= buy
                buy_date.append(today.name)
                buy_in.append(buy)
                stock2sell = True


print(len(transaction_ticker), len(buy_date), len(sell_date), len(buy_in), len(sell_out))
print(len(money_tickers), len(price_date), len(prices))
transaction = pd.DataFrame({'Tickers': transaction_ticker, 'Buy': buy_in, 'Sell': sell_out, 'buy_date': buy_date, 'sell_date': sell_date})
money_ticker = pd.DataFrame({'Tickers': money_tickers, 'Date': price_date, 'Price': prices})
print(transaction)
print(money_ticker)