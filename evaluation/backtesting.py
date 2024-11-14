from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute, ToTSData, PadNan, Process, PipeOutput, DataPipe, DataPipeType
from deep.pipes import Finviz, RmTz
from datetime import datetime
from backtest import Strategy, Backtest, RecordsBucket, TSData, Record
import pandas as pd
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm
from evaluation.dist import load_annotations

N_STOCKS = 500
@Process
def Sample(frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Dict[str, pd.DataFrame]]], **kwargs) -> Dict[str, pd.DataFrame]:
    keys = np.array(list(po.value.keys()))
    np.random.shuffle(keys)
    return {k: po.value[k] for k in keys[:N_STOCKS]}

class LoadAnnotation(DataPipe):
    def __init__(self, annotations):
        super().__init__(DataPipeType.FETCH)
        self.annotation = annotations
    def fetch(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Dict[str, pd.DataFrame]]], **kwargs):
        data = load_annotations(self.annotation)
        out = {}
        for ticker, (chart, annotations) in data.items():
            print(ticker, chart.shape, annotations.shape)
            chart["Down"] = annotations[:, 0]
            chart["Up"] = annotations[:, 1]
            chart["Neutral"] = annotations[:, 2]
            out[ticker] = chart
        return PipeOutput(out, self)

pipe = LoadAnnotation(annotations="annotations/E_s.anno") | Sample() | ToTSData()
print(pipe)
data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))

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
bt = Backtest(data,
              strategy=NaiveStrategy(),
              window=1)
results = bt.run()
print(results)