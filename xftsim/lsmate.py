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


kamr = KAssortativeMatingRegime(cross_corr=np.ones((3, 3)) * .1,
                                component_index=xft.index.ComponentIndex.from_product(['height', 'weight', 'bmi'],
                                                                                      ['phenotype'],
                                                                                      [-1]),
                                offspring_per_pair=xft.utils.ZeroTruncatedPoissonCount(1.6))


# bkamr=
