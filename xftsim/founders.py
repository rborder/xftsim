import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property

import xftsim as xft


def founder_haplotypes_from_AFs(n: int,
                                afs: Iterable,
                                diploid: bool = True):
    if diploid:
        afs = np.repeat(afs, 2)
    afs = xft.utils.ensure2D(afs)
    haplotypes = np.apply_along_axis(lambda q: np.random.binomial(1, q, n).astype(np.int8),
                                     0, afs.T)
    m = afs.shape[0] // 2
    variant_indexer = xft.index.DiploidVariantIndex(m=m,
                                                    n_chrom=np.min([22, m]),
                                                    af=afs[::2, :].ravel()).to_haploid()

    return xft.struct.HaplotypeArray(haplotypes, variant_indexer=variant_indexer)


def founder_haplotypes_uniform_AFs(n: int,
                                   m: int,
                                   minMAF: float = .1):
    afs = np.random.uniform(minMAF, 1 - minMAF, m)
    return founder_haplotypes_from_AFs(n=n,
                                       afs=afs,
                                       diploid=True)
