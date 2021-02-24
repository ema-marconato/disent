#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import warnings
from typing import Iterable

from pytorch_lightning.loggers import WandbLogger, LoggerCollection


def yield_wandb_loggers(logger) -> Iterable[WandbLogger]:
    if logger:
        if isinstance(logger, WandbLogger):
            yield logger
        elif isinstance(logger, LoggerCollection):
            for l in logger:
                yield from yield_wandb_loggers(l)


def wb_log_metrics(logger, metrics_dct: dict):
    wb_logger = None
    # iterate over loggers & update metrics
    for wb_logger in yield_wandb_loggers(logger):
        wb_logger.log_metrics(metrics_dct)
    # warn if nothing logged
    if wb_logger is None:
        warnings.warn('no wandb logger found to log metrics to!')


_SUMMARY_REDICTIONS = {
    'min': min,
    'max': max,
    'recent': lambda prev, current: current,
}


def wb_log_reduced_summaries(logger, summary_dct: dict, reduction='max'):
    reduce_fn = _SUMMARY_REDICTIONS[reduction]
    wb_logger = None
    # iterate over loggers & update summaries
    for wb_logger in yield_wandb_loggers(logger):
        for key, current in summary_dct.items():
            key = f'{key}.{reduction}'
            wb_logger.experiment.summary[key] = reduce_fn(wb_logger.experiment.summary.get(key, current), current)
    # warn if nothing logged!
    if wb_logger is None:
        warnings.warn('no wandb logger found to log metrics to!')


def log_metrics(logger, metrics_dct: dict):
    if logger:
        logger.log_metrics(metrics_dct)
    else:
        warnings.warn('no trainer.logger found!')
