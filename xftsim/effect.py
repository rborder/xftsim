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
        beta: NDArray[Any, Any], ## effects matrix or vector
        variant_indexer: xft.index.HaploidVariantIndex = None,
        component_indexer: xft.index.ComponentIndex = None,
        standardized: bool = False, ## have the effects been divided by variant SD under HWE?
        scaled: bool = False, ## have the effects been divided by the number of causal variants per column?

        ):
        
        self.component_indexer = component_indexer
        self.variant_indexer = variant_indexer

        self._tiledAF = xft.utils.ensure2D(self.variant_indexer.af)
        self.AF = xft.utils.ensure2D(self.variant_indexer.af)[::2,:]
        if np.any(np.isnan(self.AF)):
            raise RuntimeError('Must provided allele frequencies')
        # self._tiledAF = np.tile(self.AF, 2)

        ## ensure shape
        self._beta = xft.utils.ensure2D(beta)   
        self.m, self.c = beta.shape
        self._standardized = standardized
        self._scaled = scaled

        if component_indexer is None:
            self.component_indexer = xft.index.ComponentIndex.range_index(c)

    @property
    def m_causal(self): ## if not provided set to number of non-zero betas per phenotype
        # if self._m_causal is None:
        return(np.sum(self._beta != 0, axis = 0))
        # else:
        # return self._m_causal

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
    def beta_standardized_scaled_haploid(self): 
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



class GCTAEffects(AdditiveEffects:
    def __init__(self,
        vg: NDArray, ## either genetic variances or genetic variance/covariance matrix
        variant_indexer: xft.index.HaploidVariantIndex,
        component_indexer: xft.index.ComponentIndex,
        ):
        vg = np.array(vg)
        if len(vg.shape) == 1:
            vg = np.diag(vg)
        m = variant_indexer.m
        k = component_indexer.k_phenotypes
        beta = np.random.multivariate_normal(np.zeros(k), vg, m)



