from backtest.data import DataPipe, PipeOutput, DataPipeType
from datetime import datetime, timedelta
from typing import List, Tuple, Union, Optional, Any, Callable
import pandas as pd
import requests
import time
import re
from io import StringIO
from tqdm import tqdm
import numpy as np

class Finviz(DataPipe):
    def __init__(self, url: str, tickers_only: bool = False):
        super().__init__(DataPipeType.FETCH, "Finviz")
        self.url = url
        self.tickers_only = tickers_only

    def fetch(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Any]], **kwargs) -> PipeOutput:
        url = self.url
        headers = {
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
        }
        res_ok = False
        while not res_ok:
            response = requests.get(url=url, headers=headers)
            if response.status_code != 200:
                time.sleep(5)
            else:
                res_ok = True
        data_raw = response.content.decode()
        f_index = data_raw.find('<table class="styled-table-new is-rounded is-tabular-nums w-full screener_table">')
        data = data_raw[f_index:].replace('<table class="table-light is-new">', '')
        e_index = data.find('</table>')
        data = data[:e_index]
        data = data.replace("\n", '')
        data = data.replace("\r", '')
        data = re.sub('data-boxover=".*?"', '', data)
        data = re.sub('</tr>', '\n', data)
        data = data.replace('<tr valign="top">', '\n')
        data = re.sub('<tr.*?>', '\n', data)
        data = data.replace('\n\n', '\n')
        data = re.sub('</th>', ';', data)
        data = re.sub('</td>', ';', data)
        data = re.sub('<.*?>', '', data)
        df_csv2 = data.replace("    ", '').strip("\n")
        # get the total number of stocks
        f_index = data_raw.find('<div id="screener-total" class="count-text whitespace-nowrap">')
        data_total = data_raw[f_index:]
        data_total = data_total[:data_total.find(r' Total')]
        totalStocks = int(data_total[data_total.find("/"):].replace("/ ", ''))

        # make data frame
        df = pd.read_csv(StringIO(df_csv2), sep=";", index_col=0)
        stocks = df.drop(columns=[df.columns[-1]])
        actualStock = len(stocks)
        last_page = int(totalStocks/20)+1
        for _ in tqdm(range(0,last_page)):
            response = requests.get(url=url+'&r='+str(actualStock+1), headers=headers)
            if response.status_code != 200:
                time.sleep(1)
                response = requests.get(url=url+'&r='+str(actualStock+1), headers=headers)
            if response.status_code != 200:
                print(f'HTTP Error -- Status-code: {response.status_code}, message: {response.content}')
            data_raw = response.content.decode()
            f_index = data_raw.find('<table class="styled-table-new is-rounded is-tabular-nums w-full screener_table">')
            data = data_raw[f_index:].replace('<table class="table-light is-new">', '')
            e_index = data.find('</table>')
            data = data[:e_index]
            data = data.replace("\n", '')
            data = data.replace("\r", '')
            data = re.sub('data-boxover=".*?"', '', data)
            data = re.sub('</tr>', '\n', data)
            data = data.replace('<tr valign="top">', '\n')
            data = re.sub('<tr.*?>', '\n', data)
            data = data.replace('\n\n', '\n')
            data = re.sub('</th>', ';', data)
            data = re.sub('</td>', ';', data)
            data = re.sub('<.*?>', '', data)
            df_csv2 = data.replace("    ", '').strip("\n")
            df = pd.read_csv(StringIO(df_csv2), sep=";", index_col=0)
            df = df.drop(columns=[df.columns[-1]])
            stocks = pd.concat([stocks, df])
            actualStock = len(stocks)


        # Post process to convert to float columns that can be converted to float
        def convert_to_float(row):
            for col in row.index:
                if row[col] == "-":
                    row[col] = np.nan
            return row

        screen_results = stocks.apply(convert_to_float, axis=1)
        for col in screen_results.columns:
            try:
                screen_results[col] = screen_results[col].astype(float)
            except:
                pass
        screen_results = screen_results.drop_duplicates(subset=["Ticker"])
        if self.tickers_only:
            # Remove nans from the data
            tickers = screen_results["Ticker"].dropna().to_list()
            return PipeOutput(tickers, self)
        else:
            return PipeOutput(screen_results, self)

if __name__ == "__main__":
    from backtest.data import FetchCharts, Cache, JSONCache, FilterNoneCharts, CausalImpute

    pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=idx_sp500%2Cipodate_more5",
                  True) | JSONCache() | FetchCharts(progress=True, throttle=1.,
                                                    auto_adjust=False) | Cache() | FilterNoneCharts() | CausalImpute()
    pipe.set_id(0)
    print(pipe)
    data = pipe.get(datetime(2000, 1, 1), datetime(2020, 1, 1))
