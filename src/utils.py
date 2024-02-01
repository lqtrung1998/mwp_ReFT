# Copyright 2023 Bytedance Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import random
import numpy as np
import torch
import signal
import json

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def is_numeric(value):
    try:
        value = float(value)
        return True
    except Exception as e:
        return False

def floatify(s):
    try:
        return float(s)
    except:
        return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


from torch.distributed import all_reduce, ReduceOp
def do_gather(var):
    var = torch.FloatTensor(var).cuda()
    all_reduce(var, op=ReduceOp.SUM)
    var = var.cpu().numpy().tolist()
    return var

def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """
    if group is None:
        group = torch.distributed.group.WORLD
    allgather_tensor = [torch.zeros_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)
    return allgather_tensor

from trl.core import masked_mean, masked_var
def allgather_masked_whiten(values, mask, shift_mean=False):
    """Whiten values with all-gathered masked values.

    Args:
        values: (bs, ...)
        mask: (bs, ...)
        shift_mean: bool

    Returns:
        whitened values, (bs, ...)
    """
    allgather_values = allgather(values)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_values {allgather_values.shape}, {allgather_values[0, 0:3]}')

    allgather_mask = allgather(mask)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_mask {allgather_mask.shape}, {allgather_mask[0, 0:3]}')

    global_mean = masked_mean(allgather_values, allgather_mask)
    global_var = masked_var(allgather_values, allgather_mask)
    whitened = (values - global_mean) * torch.rsqrt(global_var + 1e-8)
    if shift_mean:
        whitened += global_mean
    return whitened


import scipy.signal as scipy_signal
def discount_cumsum(rewards, discount):
    return scipy_signal.lfilter([1], [1, -discount], x=rewards[::-1])[::-1]

from datetime import timedelta
def compute_ETA(tqdm_t, num_period=1):
    # elapsed = tqdm_t.format_dict["elapsed"]
    rate = tqdm_t.format_dict["rate"]
    time_per_period = tqdm_t.total / rate if rate and tqdm_t.total else 0  # Seconds*
    period_remaining = (tqdm_t.total - tqdm_t.n) / rate if rate and tqdm_t.total else 0  # Seconds*
    remaining = time_per_period*(num_period-1) + period_remaining
    return timedelta(seconds=remaining)