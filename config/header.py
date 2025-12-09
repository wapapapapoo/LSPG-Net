# from numba import jit, config
# config.DISABLE_JIT = True

import torch
import os
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_save = torch.device("cpu")

# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.5, 0)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

RANDSEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDSEED)
random.seed(RANDSEED)
np.random.seed(RANDSEED)
torch.manual_seed(RANDSEED)
torch.cuda.manual_seed(RANDSEED)
torch.cuda.manual_seed_all(RANDSEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
