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
        beta: NDArray[Any, Any], ## diploid effects matrix or vector
        variant_indexer: Union[xft.index.HaploidVariantIndex, xft.index.DiploidVariantIndex] = None,
        component_indexer: xft.index.ComponentIndex = None,
        standardized: bool = True, ## True implies these are effects of standardized variants, False = Raw
        scaled: bool = False, ## True implies these are effects of variants / sqrt(m_causal)
        ):
        
        unscaled = not scaled
        unstandardized = not standardized
        if isinstance(variant_indexer, xft.index.HaploidVariantIndex):
            self.component_indexer = component_indexer

        self.variant_indexer = variant_indexer

        self._tiledAF = xft.utils.ensure2D(self.variant_indexer.af)
        self.AF = xft.utils.ensure2D(self.variant_indexer.af)[::2,:]
        if np.any(np.isnan(self.AF)):
            raise RuntimeError('Must provided allele frequencies')

        ## ensure shape
        beta = xft.utils.ensure2D(beta)   
        m, c = beta.shape
        self.m, self.c = variant_indexer.m, sample_indexer.k_phenotypes
        if (self.m != m) or (self.c != c):
            raise RuntimeError('beta and indexers nonconformable')  
        self._beta = beta   

        self._beta_scaled_standardized_diploid = None
        self._beta_unscaled_standardized_diploid = None
        self._beta_unscaled_unstandardized_diploid = None
        self._beta_scaled_unstandardized_diploid = None

        if scaled and standardized:
            self._beta_scaled_standardized_diploid = self._beta
        elif scaled and unstandardized:
            self._beta_scaled_unstandardized_diploid = self._beta
        elif unscaled and standardized:
            self._beta_unscaled_standardized_diploid = self._beta
        elif unscaled and unstandardized:
            self._beta_unscaled_unstandardized_diploid = self._beta
        else:
            raise TypeError("standardized, scaled must be bool")

        if component_indexer is None:
            self.component_indexer = xft.index.ComponentIndex.range_index(c)

    @property
    def m_causal(self): ## number of non-zero betas per phenotype
        return(np.sum(self._beta != 0, axis = 0))

    @cached_property
    def beta_scaled_standardized_diploid(self): ## multiply against standardized (0,1,2)/sqrt(m) genotypes
        if self._beta_scaled_standardized_diploid is not None:
            return self._beta_scaled_standardized_diploid
        else: 
            return beta_scaled_standardized_diploid


        self._standardized:
            if self._scaled:
                return self._beta
            else:
                return self._beta * np.sqrt(self.m_causal)
        elif self._scaled:
            return self._beta * np.sqrt(2*self.AF*(1-self.AF))
        else:
            return self._beta * np.sqrt(2*self.AF*(1-self.AF)*self.m_causal)

    @property
    def beta_standardized_scaled_haploid(self): 
        return self.beta_standardized_scaled_diploid.repeat(2,axis=0)

    @property
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
    
    @property
    def beta_raw_haploid(self):
        return self.beta_raw_diploid.repeat(2,axis=0)

    @property
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



