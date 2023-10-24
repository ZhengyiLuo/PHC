import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
import logging

class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if (pl_module.global_step + 1) % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"model_{pl_module.global_step}.ckpt"
            trainer.save_checkpoint(current)

class TextLogger(LightningLoggerBase):
    def __init__(self, cfg, filename, file_handle = True):
        super().__init__()
        self.cfg = cfg
        self.logger = logger = logging.getLogger(filename)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(stream_formatter)
        logger.addHandler(ch)

        if file_handle:
            # create file handler which logs even debug messages
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fh = logging.FileHandler(filename, mode='a')
            fh.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

    @property
    def name(self):
        
        return 'TextLogger'

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        log_str = "".join([f"{k} : {v:.3f} \t" for k, v in metrics.items()])
        self.logger.info(log_str)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        # super().save()
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
