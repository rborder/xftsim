import warnings
import numpy as np
import pandas as pd
import math
import functools

from typing import Iterable, Union
from nptyping import NDArray, Shape, Float, Int, Object 
from collections.abc import Sequence

import xftsim as xft


import traceback
import warnings
import sys

import pgenlib as pg



with pg.PgenReader("/media/rsb/Extreme\ SSD/1KG/all_phase3_ns.pgen".encode("utf-8"),
                   sample_subset = ['HG00097']) as reader: