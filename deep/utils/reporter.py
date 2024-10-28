from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


class Reporter(SummaryWriter):
    def __init__(self,
                 log_dir=None,
                 comment="",
                 purge_step=None,
                 max_queue=10,
                 flush_secs=120,
                 filename_suffix=""):
        super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                         flush_secs=flush_secs, filename_suffix=filename_suffix)

    @property
    def keys(self):
        return self._load_events().Tags()["scalars"]

    def __getitem__(self, item):
        return pd.DataFrame(self._load_events().Scalars(item))

    def _load_events(self):
        return event_accumulator.EventAccumulator(
            self.log_dir,
            size_guidance={event_accumulator.SCALARS: 0},
        )