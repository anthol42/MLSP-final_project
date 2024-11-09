from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute, ToTSData, PadNan, Process, PipeOutput
from deep.pipes import Finviz, RmTz
from datetime import datetime
from backtest import Strategy, Backtest, RecordsBucket, TSData, Record
import pandas as pd
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm

N_STOCKS = 500
@Process
def Sample(frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Dict[str, pd.DataFrame]]], **kwargs) -> Dict[str, pd.DataFrame]:
    keys = np.array(list(po.value.keys()))
    np.random.shuffle(keys)
    return {k: po.value[k] for k in keys[:N_STOCKS]}

def annotate(data: Dict[str, TSData]):
    out = {}
    for ticker, ts in tqdm(data[0].items(), desc="Annotating"):
        chart = ts.data
        ma = chart["Close"].rolling(14).mean()
        over = chart["Close"] >= ma
        chart["Anno"] = over.astype(int) # 0: under = sell, 1: over = buy
        ts.data = chart

    return data

pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=cap_largeover%2Cexch_nyse%2Cidx_sp500%2Cipodate_more5"
              ,True) | JSONCache() | \
       FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
       Cache() | FilterNoneCharts() | Sample() | CausalImpute() | PadNan() | ToTSData()
print(pipe)
data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
annotated_data = annotate(data)
class NaiveStrategy(Strategy):
    def run(self, data: RecordsBucket, timestep: datetime):
        data = data.main
        name: str; ts: Record
        for name, record in data:
            if record.chart["Anno"].iloc[-1] == 1 and name not in self.portfolio.long:
                # Buy 1% of available cash
                price = record.chart.iloc[-1, 3]
                n_stocks = int(self.account.available_cash * 0.01 / price)
                if n_stocks > 0:
                    self.broker.buy_long(name, n_stocks)
            elif record.chart["Anno"].iloc[-1] == 0 and name in self.portfolio.long:
                # Sell all
                self.broker.sell_long(name, self.portfolio.long[name].amount)

# The magnificent 7 tickers
bt = Backtest(annotated_data,
              strategy=NaiveStrategy(),
              window=1)
results = bt.run()
print(results)