from backtest.data import DataPipe, PipeOutput, DataPipeType
from datetime import datetime, timedelta
from typing import List, Tuple, Union, Optional, Any, Callable
import json

class FromFile(DataPipe):
    def __init__(self, path: str):
        super().__init__(DataPipeType.FETCH, "FromFile")
        self.path = path

    def fetch(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Any]], **kwargs) -> PipeOutput:
        with open(self.path, "r") as f:
            data = json.load(f)
        return PipeOutput(value=data, output_from=self)
