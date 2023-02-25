import sys
import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
from dataclasses import dataclass, field

import xftsim as xft

## simple additive effects that are standardized and/or scaled
class AdditiveEffects:
    def __init__(self,
        beta: NDArray[Any, Any] = None, ## effects matrix or vector
        vid: NDArray[Shape["*"], Any] = None, ## variant id
        AF: NDArray[Any, Any] = None,
        phenotype_name: NDArray[Shape["*"], Any] = None,
        standardized: bool = False,
        scaled: bool = False,
        m_causal: NDArray[Shape["*"], Int64] = None, 
        ):
        
        self.phenotype_name = phenotype_name
        self.AF = xft.utils.ensure2D(AF)
        self._tiledAF = np.tile(self.AF, 2)
        self.vid = vid

        ## ensure shape
        if beta is None:
            self._beta = None
            self.m = None
            self.c = None
            self._standardized = False
            self._scaled = False
            self._m_causal = None
            warnings.warn("Effects not provided")
        else:
            self._beta = xft.utils.ensure2D(beta)
            self.m, self.c = beta.shape
            self._standardized = standardized
            self._scaled = scaled
            self._m_causal = m_causal

        if phenotype_name is None:
            self.phenotype_name = np.arange(self.c).astype(str)

    @cached_property
    def m_causal(self): ## if not provided set to number of non-zero betas per phenotype
        if self._m_causal is None:
            return(np.sum(self.beta == 0, axis = 0))
        else:
            return self._m_causal

    @cached_property
    def beta_standardized_scaled_diploid(self): ## multiply against standardized (0,1,2)/sqrt(m) genotypes
        if self._standardized:
            if self._scaled:
                return self._beta
            else:
                return self._beta * np.sqrt(self.m_causal)
        elif self._scaled:
            return self._beta * np.sqrt(2*self.AF*(1-self.AF))
        else:
            return self._beta * np.sqrt(2*self.AF*(1-self.AF)*self.m_causal)

    @cached_property
    def beta_standardized_scaled_haploid(self): #TODO#
        return self.beta_standardized_scaled_diploid.repeat(2,axis=0)

    @cached_property
    def beta_raw_diploid(self): ## multiple against raw (0,1,2) genotypes and add offset
        if self._standardized:
            if self._scaled:
                return self._beta / np.sqrt(2*self.AF*(1-self.AF)*self.m_causal)
            else:
                return self._beta / np.sqrt(2*self.AF*(1-self.AF)) 
        elif self._scaled:
            return self._beta / np.sqrt(self.m_causal)
        else:
            return self._beta 
    
    @cached_property
    def beta_raw_haploid(self):
        return self.beta_raw_diploid.repeat(2,axis=0)

    @cached_property
    def offset(self): ## add to raw multiple against raw (0,1,2) 
        return np.sum(-2*self.AF * self.beta_raw_diploid, axis = 0)
