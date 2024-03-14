"""
# File       : Visualization.py
# Time       : 2024/3/14 22:04
# Author     : fei jie
# version    : 1.0
# Description: 
"""
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter


def writer(logs_dir):
    return SummaryWriter(log_dir=logs_dir, max_queue=5, flush_secs=30)