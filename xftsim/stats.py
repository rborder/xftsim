import xftsim as xft
import warnings
import functools
import numpy as np
import numba as nb
import pandas as pd
import nptyping as npt
import scipy as sp
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property
import dask.array as da


class Statistic:
    """
    Base class for defining statistic estimators.

    Attributes
    ----------
    name : str
        The name of the statistic.
    estimator : Callable
        The function that estimates the statistic.

    Methods
    -------
    estimate(sim: xft.sim.Simulation) -> None:
        Estimate the statistic and update the results.

    update_results(sim: xft.sim.Simulation, results: object) -> None:
        Update the simulation's results_store with the estimated results.
    """
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
    """
    Calculate and return various sample statistics for the given simulation.

    Attributes
    ----------
    means : bool
        If True, calculate and return the mean of each phenotype.
    variance_components : bool
        If True, calculate and return the variance components of each phenotype.
    variances : bool
        If True, calculate and return the variances of each phenotype.
    vcov : bool
        If True, calculate and return the variance-covariance matrix.
    corr : bool
        If True, calculate and return the correlation matrix.
    prettify : bool
        If True, prettify the output by converting it to a pandas DataFrame.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Calculate and return the requested sample statistics for the given simulation.
    """
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

    @xft.utils.profiled(level=2, message = "sample statistics")
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
    """
    Calculate and return various mating statistics for the given simulation.

    Parameters
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
    full: bool
        Ignore component_index and compute statistics for all components
        If component_index is not provided, and full = False, calculate statistics for phenotype components only.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Calculate and return the requested mating statistics for the given simulation.
    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 full: bool = False,
                 ):
        self.name = 'mating_statistics'
        self._full = full
        self.component_index = component_index

    @xft.utils.profiled(level=2, message = "mating statistics")
    def estimator(self, sim: xft.sim.Simulation) -> Dict:
        output = dict(
                      n_reproducing_pairs = sim.mating.n_reproducing_pairs,
                      n_total_offspring = sim.mating.n_total_offspring,
                      mean_n_offspring_per_pair =  np.mean(sim.mating.n_offspring_per_pair),
                      mean_n_female_offspring_per_pair =  np.mean(sim.mating.n_females_per_pair),
                      mate_correlations = sim.mating.get_mate_phenotypes(phenotypes=sim.phenotypes,
                                                                         component_index=self.component_index,
                                                                         full = self._full).corr(),
                      )
        return output

    

# ## estimates tr (X %*% t(X) %*% X %*% t(X)) using l probing vectors
@nb.jit(parallel=True)
def _rtrace_K2(X, l): ## (matrix, int) -> float
    """
    Estimate the trace of the matrix product (X %*% t(X) %*% X %*% t(X)) using l probing vectors.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    float
        The estimated trace of the matrix product.
    """
    probes = np.random.randn(X.shape[0], l)
    W = probes.T @ (X @ (X.T @ (X @ (X.T @ probes))))
    return np.trace(W)/l

def _rtrace_K2_dask(X, l): ## (matrix, int) -> float
    """
    Estimate the trace of the matrix product (X %*% t(X) %*% X %*% t(X)) using l probing vectors with Dask.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    float
        The estimated trace of the matrix product.
    """
    probes = np.random.randn(X.shape[0], l)
    W = da.dot(probes.T, da.dot(X, da.dot(X.T, da.dot(X, da.dot(X.T, probes)))))
    return da.trace(W)/l

@nb.jit(parallel=True)
def _haseman_elston_randomized(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    l, # (int) number of random probes for trace estimation
                    ):
    """
    Perform randomized Haseman-Elston regression.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    n,m = G.shape
    Ky = G @ (G.T @ Y)/m
    trK2 = _rtrace_K2(G/np.sqrt(m), l)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = (Y.T @ (Ky - Y)) / denom
    return cov_g_HE

def _haseman_elston_randomized_dask(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    l, # (int) number of random probes for trace estimation
                    ):
    """
    Perform randomized Haseman-Elston regression with Dask.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    n,m = G.shape
    Ky = da.dot(G, da.dot(G.T, Y))/m
    trK2 = _rtrace_K2_dask(G/np.sqrt(m), l)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = da.dot(Y.T, (Ky - Y)) / denom
    return cov_g_HE.compute()


@nb.jit(parallel=True)
def _haseman_elston_deterministic(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    ):
    """
    Perform deterministic Haseman-Elston regression.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
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
                   dask: bool = False,
                   ):
    """
    Perform Haseman-Elston regression, with the option to choose randomized, deterministic, or randomized dask-based methods.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.
    n_probe : int, optional, default=500
        The number of random probes for trace estimation. If n_probe is set to inf, use deterministic method.
    dtype : numpy data type, optional, default=np.float32
        The data type for the input arrays.
    dask : bool, optional, default=False
        If True, use dask for calculations.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    Y = xft.utils.ensure2D(Y).astype(dtype)
    G = np.array(G)#, dtype=dtype)
    if np.isinf(n_probe):
        return _haseman_elston_deterministic(G, Y)
    elif dask:
        return _haseman_elston_randomized_dask(G, Y, n_probe)
    else:
        return _haseman_elston_randomized(G, Y, n_probe)

  
class HasemanElstonEstimator(Statistic):
    """
    Estimate Haseman-Elston regression for the given simulation.

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all components.
    genetic_correlation : bool
        If True, calculate and return the genetic correlation matrix.
    randomized : bool
        If True, use a randomized trace estimator.
    prettify : bool
        If True, prettify the output by converting it to a pandas DataFrame.
    n_probe : int
        The number of random probes for trace estimation.
    dask : bool
        If True, use dask for calculations.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Estimate and return the Haseman-Elston regression for the given simulation.
    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 genetic_correlation: bool = True,
                 randomized: bool = True,
                 prettify: bool = True,
                 n_probe: int = 100,
                 dask: bool = True,
                 ):
        self.name = 'HE_regression'
        self.component_index = component_index
        self.genetic_correlation = genetic_correlation
        self.randomized = randomized
        self.prettify = prettify
        self.n_probe = n_probe
        self.dask = dask

    @xft.utils.profiled(level=2, message = "haseman elston estimator")
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
                                dask = self.dask,
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


@nb.njit
def _linear_regression_with_intercept_nb(X: NDArray,
                                         y: NDArray) -> NDArray:
    """
    Numba implementation of simple linear regression of y ~ 1 + X. 
    Intercepts term is included but corresponing coefficients are not reported
    
    Parameters
    ----------
    X : NDArray
        n-by-1 array of predictor
    y : NDArray
        n-by-1 standardized outcome array
    
    Returns
    -------
    NDArray
        2 array consisting of standardized slope corresponding standard errors.
    """
    n = X.shape[0]
    y = y.reshape((n,1))
    X = X/np.std(X)
    X = X.reshape((n,1))
    pred = np.hstack((np.ones_like(X), X))
    XtX = pred.T @ pred
    XtXinv = np.linalg.pinv(pred.T @ pred)
    coef = XtXinv @ (pred.T @ y)
    yhat = pred @ coef
    res = y - yhat
    s2 = (res.T @ res) / (n - 2)
    std_err = np.sqrt(s2 * np.diag(XtXinv))
    return [coef.ravel()[1], std_err.ravel()[1]]

@nb.njit
def _vec_linear_regression_with_intercept_nb(X: NDArray,
                                             y: NDArray) -> NDArray:
    """
    Numba implementation of simple linear regression vectorized over predictors. 
    Intercepts term is included but corresponing coefficients are not reported
    
    Parameters
    ----------
    X : NDArray
        n-by-m array of predictors
    y : NDArray
        n-by-1 standardized outcome array
    
    Returns
    -------
    NDArray
        m-by-2 array of standardized slopes (first column) and corresponding standard errors (second column)
        for each of the m predictors. 
    """
    output = np.empty((X.shape[1], 2), dtype=y.dtype)
    for j in np.arange(X.shape[1]):
        output[j,:] = _linear_regression_with_intercept_nb(X[:,j],y)
    return output



@nb.njit
def _mv_vec_linear_regression_with_intercept_nb(X: NDArray,
                                                Y: NDArray) -> NDArray:
    """
    Numba implementation of simple linear regression vectorized over predictors
    and outcomes. 
    Intercept terms are included but corresponing coefficients are not reported
    
    Parameters
    ----------
    X : NDArray
        n-by-m array of predictors
    Y : NDArray
        n-by-k standardized outcome arrays
    
    Returns
    -------
    NDArray
        m-by-2-by-k array of standardized slopes (first column) and corresponding standard errors (second column)
        for each of the m predictors. 
    """
    output = np.empty((X.shape[1], 2, Y.shape[1]), dtype=Y.dtype)
    for k in nb.prange(Y.shape[1]):
        y = Y[:,k].ravel()
        y = y / np.std(y)
        for j in np.arange(X.shape[1]):
            output[j,:, k] = _linear_regression_with_intercept_nb(X[:,j],y)
    return output


def _p_val_t(t: float, df: Union[int, float]) -> float:
    """
    p-values for student's t distribution
    
    Parameters
    ----------
    t : float
        Test statistic
    df : int or float
        degrees of freedom
    
    Returns
    -------
    float
        p-value
    """
    return (1-sp.stats.t.cdf(t,df))*2


def _mv_gwas_nb(X: NDArray, Y: NDArray) -> NDArray:
    """
    Numba implementation of simple linear regression vectorized over predictors and outcomes 
    Intercepts term is included but corresponing coefficients are not reported.
    Both X and Y columns are automatically scaled to have unit variance.
    
    Parameters
    ----------
    X : NDArray
        n-by-m array of predictors
    Y : NDArray
        n-by-k outcome array
    
    Returns
    -------
    NDArray
        m-by-4-by-k array with columns respectively corresponding to
        slopes, standard errors, test statistics, and p-values 
        for each of the m predictors. 
    """
    tmp = _mv_vec_linear_regression_with_intercept_nb(X,Y)
    output = np.tile(tmp,(1,2,1))
    output[:,2,:] = output[:,0,:] /output[:,1,:] 
    output[:,3,:] = _p_val_t(output[:,2,:], X.shape[0]-2) 
    return(output)




class GWAS_Estimator(Statistic):
    """
    Perform linear assocation studies for the given simulation.

    When called within a Simulation, will add to Simulation.results['GWAS']
    a 3-D array indexed as follows:
        - the first dimension indexes variants via xft.index.DiploidVariantIndex
        - the second dimension indexes four association statistics: slope, se, test-statistic, and p-value
        - the third dimension indexes phenotypic components via xft.index.ComponentIndex

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all phenotype components.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Estimate and return the Haseman-Elston regression for the given simulation.
    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 # numba: bool = True,
                 ):
        self.name = 'GWAS'
        self.component_index = component_index
        # self.numba = numba

    @xft.utils.profiled(level=2, message = "GWAS estimator")
    def estimator(self, sim: xft.sim.Simulation) -> Dict:
        ## look for "phenotype" components if component_index not provided
        if self.component_index is None:
            # pheno_cols= sim.phenotypes.component_name.values[sim.phenotypes.component_name.str.contains('phenotype')]
            # component_index = sim.phenotypes.xft.get_component_indexer()[dict(component_name=pheno_cols)]
            component_index = sim.phenotypes.xft.grep_component_index('phenotype')
        else:
            component_index = self.component_index
        Y = sim.current_std_phenotypes.xft[None, component_index].data.astype(np.float32)
        # if Y.shape[1] <2:
        #     raise NotImplementedError()
        G = sim.current_std_genotypes
        sum_stats = _mv_gwas_nb(G,Y)
        coord_dict = component_index.coord_dict.copy()
        coord_dict.update(sim.haplotypes.xft.get_variant_indexer().to_diploid().coord_dict)
        coord_dict.update({'statistic':('statistic', ['beta', 'se', 't', 'p'])})
        output = dict(estimates=xr.DataArray(sum_stats,dims=('variant', 'statistic', 'component'), 
                                        coords=coord_dict))
        return output

