"""
mcts??
treenode for representing board position. what it tracks? initial guess, win loss score,
track of how many times it has visited state and avg score may be to tell how good is the state
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import Dict, Optional, List
from model import PolicyValueNet

#class TreeNode:

#class Mcts:

#if __name__ == "__main__":
#  run_unit()
