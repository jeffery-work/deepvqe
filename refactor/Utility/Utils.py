"""
# File       : Utils.py
# Time       : 2024/3/14 21:45
# Author     : fei jie
# version    : 1.0
# Description: 
"""
import importlib
import time
import torch
import os

def InitializeConfig(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    eg2, specific function
         module_cfg = {
            "module": "Dataset.WavDataset", # WavDataset instance from Dataset/ path
            "main": "WavDataset", # function name, it's WavDataset constructor here
            "args": {...} # args in WavDataset constructor function
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"]) # module is a specific instance

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])

class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)

def prepare_device(n_gpu: int, cudnn_deterministic=False):
    """Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0")

    return device

def LoadCheckpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(os.path.abspath(os.path.expanduser(checkpoint_path)), map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]

def MakeEmptyDir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
