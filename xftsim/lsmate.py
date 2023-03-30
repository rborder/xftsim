"""
This module contains functions and classes for implementing different mating regimes in the context of population genetics simulations.

Functions:

    _solve_qap_ls: Private function that solves the Quadratic Assignment Problem using LocalSolver.

Classes:

    MatingRegime: Base class for defining mating regimes.
    RandomMatingRegime: A class for implementing random mating.
    BalancedRandomMatingRegime: A class for implementing balanced random mating.
    LinearAssortativeMatingRegime: A class for implementing linear assortative mating.
    KAssortativeMatingRegime: A class for implementing k-assortative mating.
    BatchedMatingRegime: A class for batching individuals to improve mating regime performance.
"""


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

import localsolver


def _solve_qap_ls(Y, Z, R, nb_threads=6, time_limit=30, tolerance=1e-5):
    """
    Solves the Quadratic Assignment Problem (QAP) using the LocalSolver optimization solver.
    
    Parameters
    ----------
    Y : numpy.ndarray
        The first set of mate phenotypes.
    Z : numpy.ndarray
        The second set of mate phenotypes.
    R : numpy.ndarray
        The target cross-correlation matrix.
    nb_threads : int, optional
        The number of threads to be used for the optimization.
    time_limit : int, optional
        The amount of time (in seconds) to run the optimization.
    tolerance : float, optional
        The minimum threshold for the objective function to terminate optimization.

    Returns
    -------
    P : numpy.ndarray
        A permutation matrix.
    """
    n = Y.shape[0]
    # for later use as initial value
    tmp = np.argsort(np.apply_along_axis(np.mean, 1, Y))[
        np.argsort(np.argsort(np.apply_along_axis(np.mean, 1, Z)))]
    with localsolver.LocalSolver() as ls:
        ls.param.set_time_limit(int(time_limit))
        ls.param.set_nb_threads(nb_threads)

        model = ls.model
        # flows
        YY = Y @ Y.T / (n - 1)
        array_YY = model.array(model.array(YY[i, :]) for i in range(n))
        # distance
        ZZ = Z @ Z.T / (n - 1)
        # cost
        W = Y @ R @ Z.T / (n - 1)
        array_W = model.array(model.array(W[i, :]) for i in range(n))
        # permutation
        p = model.list(n)
        model.constraint(model.eq(model.count(p), n))
        # objective
        const = np.trace(W @ W.T)
        qobj = model.sum(
            model.at(array_YY, p[i], p[j]) * ZZ[i, j] for j in range(n) for i in range(n))
        lobj = model.sum(model.at(array_W, p[i], i) for i in range(n))
        obj = qobj - 2 * lobj + const
        model.minimize(obj)
        model.close()
        # set initial value of permutation
        p.value.clear()
        for pp in tmp:
            p.value.add(pp)
        # solve
        # ls.param.set_objective_threshold(0,tolerance)
        ls.param.set_verbosity(1)
        ls.solve()

        # solution
        P = np.array([p.value.get(i) for i in range(n)])

        return P


class KAssortativeMatingRegime(xft.mate.MatingRegime):
    """
    A class that implements the K-assortative mating regime. I.e., matches two sets of individuals with
    K phenotypes to achieve an arbitrary K x K cross-mate cross-correlation structure.

    Parameters
    ----------
    component_index : xft.index.ComponentIndex
        An object containing information about the components.
    cross_corr : ndarray
        The cross-correlation matrix of size K x K.
    offspring_per_pair : Union[int, xft.utils.VariableCount], optional
        The number of offspring per mating pair. Default is 1.
    mates_per_female : Union[int, xft.utils.VariableCount], optional
        The number of mates for each female. Default is 1.
    female_offspring_per_pair : Union[str, int, xft.utils.VariableCount], optional
        The number of offspring per mating pair for females. Default is 'balanced'.
    sex_aware : bool, optional
        Whether to consider sex in mating pairs. Default is False.
    exhaustive : bool, optional
        Whether to enumerate all possible pairs. Default is True.

    Attributes
    ----------
    cross_corr : ndarray
        The cross-correlation matrix of size K x K.
    component_index : xft.index.ComponentIndex
        An object containing information about the components.
    K : int
        The total number of components.

    Methods
    -------
    mate(haplotypes: xr.DataArray = None, phenotypes: xr.DataArray = None, control: dict = None) -> xft.mate.MateAssignment:
        Mate haplotypes and phenotypes based on the K-assortative mating regime.
    """

    def __init__(self,
                 component_index: xft.index.ComponentIndex,
                 cross_corr: NDArray,
                 offspring_per_pair: Union[int, xft.utils.VariableCount] = xft.utils.ConstantCount(
                     1),
                 mates_per_female: Union[int, xft.utils.VariableCount] = xft.utils.ConstantCount(
                     1),
                 # doesn't make total sense
                 female_offspring_per_pair: Union[str, int,
                                                  xft.utils.VariableCount] = 'balanced',
                 sex_aware: bool = False,
                 exhaustive: bool = True,
                 ):
        super().__init__(self,
                         offspring_per_pair=offspring_per_pair,
                         mates_per_female=mates_per_female,
                         female_offspring_per_pair=female_offspring_per_pair,
                         sex_aware=sex_aware,
                         exhaustive=exhaustive,
                         )
        # Linear regime attributes
        self.cross_corr = cross_corr
        self.component_index = component_index
        self.K = component_index.k_total

    def mate(self,
             haplotypes: xr.DataArray = None,
             phenotypes: xr.DataArray = None,
             control: dict = None,
             ):
        """
        Mate haplotypes and phenotypes based on the K-assortative mating regime.

        Parameters
        ----------
        haplotypes : xr.DataArray, optional
            The haplotype data to be mated. Default is None.
        phenotypes : xr.DataArray, optional
            The phenotype data to be mated. Default is None.
        control : dict, optional
            A dictionary of control parameters for mating. Default is None.

        Returns
        -------
        assignment : xft.mate.MateAssignment
            The assignment of haplotypes to parents.
        """
        female_indices, male_indices = self.get_potential_mates(
            haplotypes, phenotypes)

        sample_indexer = phenotypes.xft.get_sample_indexer()
        female_indexer = sample_indexer[female_indices]
        male_indexer = sample_indexer[male_indices]

        # downsample for balance
        n_new = np.min([female_indexer.n, male_indexer.n])
        female_indexer = female_indexer.at_most(n_new)
        male_indexer = male_indexer.at_most(n_new)

        female_components = xft.utils.standardize_array(phenotypes.xft[female_indexer,
                                                                       self.component_index].data)
        male_components = xft.utils.standardize_array(phenotypes.xft[male_indexer,
                                                                     self.component_index].data)

        perm = _solve_qap_ls(
            female_components, male_components, self.cross_corr)

        return self.enumerate_assignment(female_indices=female_indices[perm],
                                         male_indices=male_indices,
                                         haplotypes=haplotypes,
                                         phenotypes=phenotypes,
                                         )


# kamr = KAssortativeMatingRegime(cross_corr=np.ones((3, 3)) * .1,
#                                 component_index=xft.index.ComponentIndex.from_product(['height', 'weight', 'bmi'],
#                                                                                       ['phenotype'],
#                                                                                       [-1]),
#                                 offspring_per_pair=xft.utils.ZeroTruncatedPoissonCount(1.6))


# bkamr=
