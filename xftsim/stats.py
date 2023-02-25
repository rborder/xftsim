import xftsim as xft
import warnings
import functools
import numpy as np
import numba as nb
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property


class Statistic:
    def __init__(self, 
                 estimator: Callable,
                 name: str,
                 ):
        self.name = name
        self.estimator = estimator

    def estimate(self,
                 sim: xft.sim.Simulation) -> None:
        output = self.estimator(sim)
        self.update_results(sim, output)

    def update_results(self, sim: xft.sim.Simulation, results: object) -> None:
        ## initialize empty dict if result store for generation is empty
        if sim.generation not in sim.results_store.keys():
            sim.results_store[sim.generation] = {}
        ## initialize empty dict if estimator hasn't been used before in current generation
        if self.name not in sim.results_store[sim.generation].keys():
            sim.results_store[sim.generation][self.name] = {}
        ## append results to simulation results_store
        sim.results_store[sim.generation][self.name] = results

class SampleStatistics(Statistic):
    def __init__(self,
                 means: bool = True,
                 variance_components: bool = True,
                 variances: bool = True,
                 vcov: bool = True,
                 corr: bool = True,
                 prettify: bool = True,
                 ):
        self.name = 'sample_statistics'
        self.means = means
        self.variances = variances
        self.variance_components = variance_components
        self.vcov = vcov
        self.corr = corr
        self.prettify = prettify

    def estimator(self, sim: xft.sim.Simulation) -> Dict:
        output = dict()
        pheno_df = sim.phenotypes.xft.as_pd(prettify=self.prettify)
        h = [g.var(axis=1) for _, g in pheno_df.T.groupby(['phenotype_name','vorigin_relative'])]
        mm = [g.mean(axis=1) for _, g in pheno_df.T.groupby(['phenotype_name','vorigin_relative'])]
        if self.means:
            output['means'] = pd.concat([g for g in mm])
        if self.variances:
            output['variances'] = pd.concat([g for g in h])
        if self.variance_components:
            output['variance_components'] = pd.concat([x.iloc[:-1] for x in [g/g.iloc[g.index.get_locs([slice(None),'phenotype'])].values for g in h] if x.size>1])
        if self.vcov:
            output['vcov'] = pheno_df.cov()
        if self.corr:
            output['corr'] = pheno_df.corr()
        return output


class MatingStatistics(Statistic):
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 ):
        self.name = 'mating_statistics'
        self.component_index = component_index

    def estimator(self, sim: xft.sim.Simulation) -> Dict:
        output = dict(
                      n_reproducing_pairs = sim.mating.n_reproducing_pairs,
                      n_total_offspring = sim.mating.n_total_offspring,
                      mean_n_offspring_per_pair =  np.mean(sim.mating.n_offspring_per_pair),
                      mean_n_female_offspring_per_pair =  np.mean(sim.mating.n_females_per_pair),
                      mate_correlations = sim.mating.get_mate_phenotypes(phenotypes=sim.phenotypes,
                                                                         component_index=self.component_index,
                                                                         ).corr(),
                      )
        return output

    

# ## estimates tr (X %*% t(X) %*% X %*% t(X)) using l probing vectors
@nb.jit(parallel=True)
def _rtrace_K2(X, l): ## (matrix, int) -> float
    probes = np.random.randn(X.shape[0], l)
    W = probes.T @ (X @ (X.T @ (X @ (X.T @ probes))))
    return np.trace(W)/l

@nb.jit(parallel=True)
def _haseman_elston_randomized(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    l, # (int) number of random probes for trace estimation
                    ):
    n,m = G.shape
    Ky = G @ (G.T @ Y)/m
    trK2 = _rtrace_K2(G/np.sqrt(m), l)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = (Y.T @ (Ky - Y)) / denom
    return cov_g_HE

@nb.jit(parallel=True)
def _haseman_elston_deterministic(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    ):
    n,m = G.shape
    Ky = G @ (G.T @ Y)/m
    trK2 = np.trace(G @ (G.T @ G @ (G.T/m))/m)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = (Y.T @ (Ky - Y)) / denom
    return cov_g_HE

def haseman_elston(
                   G, # (2D array) standardized (but not scaled) diploid genotypes
                   Y, # (2D array) standardized phenotypes
                   n_probe = 500, # (int) number of random probes for trace estimator, inf -> deterministic
                   dtype=np.float32,
                   ):
    Y = xft.utils.ensure2D(Y).astype(dtype)
    G = np.array(G, dtype=dtype)
    if np.isinf(n_probe):
        return _haseman_elston_deterministic(G, Y)
    else:
        return _haseman_elston_randomized(G, Y, n_probe)

  
class HasemanElstonEstimator(Statistic):
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 genetic_correlation: bool = True,
                 randomized: bool = True,
                 prettify: bool = True,
                 n_probe: int = 100,
                 ):
        self.name = 'HE_regression'
        self.component_index = component_index
        self.genetic_correlation = genetic_correlation
        self.randomized = randomized
        self.prettify = prettify
        self.n_probe = n_probe

    def estimator(self, sim: xft.sim.Simulation) -> Dict:
        ## look for "phenotype" components if component_index not provided
        if self.component_index is None:
            # pheno_cols= sim.phenotypes.component_name.values[sim.phenotypes.component_name.str.contains('phenotype')]
            # component_index = sim.phenotypes.xft.get_component_indexer()[dict(component_name=pheno_cols)]
            component_index = sim.phenotypes.xft.grep_component_index('phenotype')
        else:
            component_index = self.component_index
        Y = sim.current_std_phenotypes.xft[None, component_index].xft.as_pd()
        he_out = haseman_elston(G = sim.current_std_genotypes,
                                Y = Y,
                                n_probe = self.n_probe,
                                )
        output = dict()
        output['cov_HE'] = pd.DataFrame(he_out, index = Y.columns, columns = Y.columns)
        if self.genetic_correlation:
            output['corr_HE'] = xft.utils.cov2cor(output['cov_HE'])
        return output




        # h = [g.var(axis=1) for _, g in pheno_df.T.groupby(['phenotype


# def _average_sib_corr(a: NDArray):
#     ac = np.corrcoef(a)
#     acshape = ac.shape
#     if acshape[0] == 1:
#         return np.NaN
#     else:
#         return np.mean(ac[np.tril_indices(ac.shape[0])])


# class SiblingSampleStatistics(Statistic):
#     def __init__(self,
#                  variance_components: bool = True,
#                  variances: bool = True,
#                  vcov: bool = True,
#                  corr: bool = True,
#                  prettify: bool = True,
#                  ):
#         self.name = 'sample_statistics'
#         self.variances = variances
#         self.variance_components = variance_components
#         self.vcov = vcov
#         self.corr = corr
#         self.prettify = prettify

#     def processor(self, sim: xft.sim.Simulation) -> Dict:
#         output = dict()
#         pheno_df = sim.phenotypes.xft.as_pd(prettify=self.prettify)
#         h = [g.var(axis=1) for _, g in pheno_df.T.groupby(['phenotype_name','vorigin_relative'])]
#         if self.variances:
#             output['variances'] = pd.concat([g for g in h])
#         if self.variance_components:
#             output['variance_components'] = pd.concat([x.iloc[:-1] for x in [g/g.iloc[g.index.get_locs([slice(None),'phenotype'])].values for g in h] if x.size>1])
#         if self.vcov:
#             output['vcov'] = pheno_df.cov()
#         if self.corr:
#             output['corr'] = pheno_df.corr()
#         return output
