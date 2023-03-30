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
                                diploid: bool = True) -> xft.struct.HaplotypeArray:
    """
    Generate founder haplotypes from specified allele frequencies.
    
    Parameters
    ----------
    n : int
        Number of haplotypes to simulate.
    afs : Iterable
        Allele frequencies as an iterable of floats.
    diploid : bool, optional
        Flag indicating if the generated haplotypes should be diploid or haploid.
        
    Returns
    -------
    xft.struct.HaplotypeArray
        An object representing a set of haplotypes generated from the given allele frequencies.
    """
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
                                   minMAF: float = .1) -> xft.struct.HaplotypeArray:
    """
    Generate founder haplotypes from uniform-distributed allele frequencies.
    
    Parameters
    ----------
    n : int
        Number of haplotypes to simulate.
    m : int
        Number of variants.
    minMAF : float, optional
        Minimum minor allele frequency for generated haplotypes.
        
    Returns
    -------
    xft.struct.HaplotypeArray
        An object representing a set of haplotypes generated with uniform allele frequencies.
    """
    afs = np.random.uniform(minMAF, 1 - minMAF, m)
    return founder_haplotypes_from_AFs(n=n,
                                       afs=afs,
                                       diploid=True)
