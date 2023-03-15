import xftsim as xft
import warnings
import functools
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
import pandas as pd


class PostProcessor:
    def __init__(self,
                 processor: Callable,
                 name: str,
                 ):
        self.name = name
        self.processor = processor

    """docstring for PostProcessor"""

    def process(self,
                sim: xft.sim.Simulation) -> None:
        self.processor(sim)


class LimitMemory(PostProcessor):
    def __init__(self,
                 n_haplotype_generations: int = -1,
                 n_phenotype_generations: int = -1,
                 ):
        assert (n_haplotype_generations == -1) | (n_haplotype_generations >=
                                                  1), "valid inputs are -1 (keep all generations) OR n>=1 (keep n most recent generations)"
        assert (n_phenotype_generations == -1) | (n_phenotype_generations >=
                                                  1), "valid inputs are -1 (keep all generations) OR n>=1 (keep n most recent generations)"
        self.n_haplotype_generations = n_haplotype_generations
        self.n_phenotype_generations = n_phenotype_generations

    def processor(self,
                  sim: xft.sim.Simulation) -> None:
        # delete all but last
        if self.n_haplotype_generations >= 1:
            for k in range(0, sim.generation - (self.n_haplotype_generations - 1)):
                if k >= 0:
                    sim.haplotype_store[k] = None
        if self.n_phenotype_generations >= 1:
            for k in range(0, sim.generation - (self.n_phenotype_generations - 1)):
                if k >= 0:
                    sim.phenotype_store[k] = None
        return None


class WriteToDisk(PostProcessor):
    """docstring for PostProcess"""

    def __init__(self, arg):
        self.arg = arg
