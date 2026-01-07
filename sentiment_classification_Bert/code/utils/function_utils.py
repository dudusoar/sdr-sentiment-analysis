import time
import torch
import random
import numpy as np
from datetime import timedelta

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # Set CPU random seed
    torch.cuda.manual_seed(seed) # Set GPU random seed
    torch.cuda.manual_seed_all(seed) # Set random seed for all GPUs

def get_time_dif(start_time):
    """
    Get elapsed time
    :param start_time: Start time
    :return: Elapsed time as timedelta
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# Given labels are text, create label_map to build mapping relationships
def get_label_map(label_list):
    id2label = dict([(idx, label) for idx, label in enumerate(label_list)])
    label2id = dict([(label, idx) for idx, label in enumerate(label_list)])
    return id2label, label2id

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

