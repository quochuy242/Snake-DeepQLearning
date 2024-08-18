import random
from collections import deque

import numpy as np
import torch

from game import Direction, Point, SnakeGame

# Constant
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        pass
