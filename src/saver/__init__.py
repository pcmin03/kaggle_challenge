import torch
import logging

from .checkpoint import CheckpointManager
from . import metric_grap

LOGGER = logging.getLogger(__name__)

def create(conf):
    # saver = metric_grap.metric_gatter(conf,conf.saver.top_k,conf.saver.mode)
    save = CheckpointManager()
    return saver