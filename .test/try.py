# %%
import math

import torch
import torch.nn as nn
# %%
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
from copy import deepcopy
import string
from time import perf_counter
from typing import Callable

from lightning.fabric.wrappers import _FabricModule
import numpy as np
import torch
import torch.nn as nn
# %%
from typing import List

import torch
from torch_geometric.loader import DataLoader
# %%
from typing import Any, List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# %%
from typing import Any, List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# %%
from copy import deepcopy
from os import makedirs, path as osp
from time import perf_counter
from typing import Dict, Tuple, Any
import warnings

from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

# %%
import seaborn as sns
# %%
import torch
import torch.nn as nn
import torchlens as tl

print("torch:", torch.__version__)
print("torchlens:", getattr(tl, "__version__", "unknown"))

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

x = torch.randn(2, 10)
# %%
# model_history = tl.log_forward_pass(
#     model,
#     x,
#     layers_to_save="all",
#     vis_opt="unrolled",
# )

# print(model_history)

# %%
import graphviz

def _save_source_only(self, filename=None, directory=None, view=False,
                      cleanup=False, format=None, outfile=None,
                      engine=None, quiet=False):
    return self.save(filename, directory)

graphviz.Digraph.render = _save_source_only

model_history = tl.log_forward_pass(
    model,
    x,
    layers_to_save="all",
    vis_opt="unrolled",
    vis_outpath="graph.gv",      # 默认就是 graph.gv，可省略
    vis_save_only=True,          # 避免在 notebook 里 display
)

print(model_history)
print(open("graph.gv").read()[:500])
# %%
