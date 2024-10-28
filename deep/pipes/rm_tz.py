from backtest.data import Process, PipeOutput
from datetime import datetime
from typing import Optional, Dict
import pandas as pd

@Process
def RmTz(frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Dict[str, pd.DataFrame]]], **kwargs) -> Dict[str, pd.DataFrame]:
    out = {}
    for ticker, chart in po.value.items():
        out[ticker] = chart
        out[ticker].index = out[ticker].index.date

    return out

if __name__ == "__main__":
    from backtest.data import FetchCharts, Cache, JSONCache, FilterNoneCharts, CausalImpute
    from deep.pipes.finviz import Finviz
    pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=idx_ndx%2Csh_price_o100",
                  True) | JSONCache() | FetchCharts(progress=True, throttle=1.,
                                                    auto_adjust=False) | Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
    pipe.set_id(0)
    print(pipe)
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
    print(data["AAPL"])