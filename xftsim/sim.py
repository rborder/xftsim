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
import funcy



class Simulation():
    """docstring for Simulation"""

    def __init__(self,
                 founder_haplotypes: xr.DataArray,
                 mating_regime: xft.mate.MatingRegime,
                 recombination_map: xft.reproduce.RecombinationMap,
                 architecture: xft.arch.Architecture,
                 statistics: Iterable = [],
                 post_processors: Iterable = [],
                 generation: int = -1,
                 control={},
                 reproduction_method=xft.reproduce.Meiosis,
                 ):
        # attributes
        self.mating_regime = mating_regime
        self.recombination_map = recombination_map
        self.architecture = architecture
        self.statistics = statistics
        self.post_processors = post_processors
        self.reproduction_method = reproduction_method
        self.reproduction_regime = self.reproduction_method(
            self.recombination_map)
        # default control parameters:
        ctrl = Simulation._default_control()
        ctrl.update(control)
        self.control = ctrl
        # properties
        self._generation = generation
        # initialize data stores
        # for proper initialization
        self.haplotype_store = {np.max([generation, 0]): founder_haplotypes}
        self.phenotype_store = {}
        self.mating_store = {}
        self.results_store = {}
        self.pedigree = None  # TODO

        #### generation specific cached properties ####
        # computed once per generation, deleted next generation
        # by call to increment_generation
        self._current_afs_empirical = None
        self._current_std_genotypes = None
        self._current_std_phenotypes = None

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, value):
        self._control = value
        self._validate_control()

    @staticmethod
    def _default_control():
        return dict(
            standardization='hardy_weinberg_empirical',
        )

    def _validate_control(self):
        if self._control['standardization'] not in [
            'hardy_weinberg_empirical',
            # 'hardy_weinberg_reference',
        ]:
            raise RuntimeError()

    @xft.utils.profiled(level=0)
    def run(self, n_generations: int):
        for generation in range(n_generations):
            self.run_generation()

    @xft.utils.profiled(level=0)
    def run_generation(self):
        self.increment_generation()
        self.reproduce()
        self.compute_phenotypes()
        self.mate()
        self.update_pedigree()
        self.estimate_statistics()
        self.process()

    @xft.utils.profiled()
    def compute_phenotypes(self):
        # generation zero
        if self.generation == 0:
            self.phenotypes = self.architecture.initialize_founder_phenotype_array(
                self.haplotypes, self.control)
            self.architecture.compute_phenotypes(
                self.haplotypes, self.phenotypes, self.control)
        elif self.generation >= 1:
            self.phenotypes = self.architecture.initialize_phenotype_array(
                self.haplotypes, self.control)
            xft.reproduce.transmit_parental_phenotypes(self.parent_mating,
                                                       self.parent_phenotypes,
                                                       self.phenotypes,
                                                       self.control)
            self.architecture.compute_phenotypes(
                self.haplotypes, self.phenotypes, self.control)

    @xft.utils.profiled()
    def mate(self):
        self.mating = self.mating_regime.mate(
            self.haplotypes, self.phenotypes, self.control)

    @xft.utils.profiled()
    def reproduce(self):
        if self.generation == 0:
            pass
        elif self.generation >= 1:
            self.haplotypes = self.reproduction_regime.reproduce(self.parent_haplotypes,
                                                                 self.parent_mating,
                                                                 self.control)

    @xft.utils.profiled()
    def estimate_statistics(self):
        for stat in self.statistics:
            stat.estimate(self)

    @xft.utils.profiled()
    def process(self):
        for ind, proc in enumerate(self.post_processors):
            # allow callables that map xft.Simulation -> None
            if isinstance(proc, Callable):
                proc = xft.proc.PostProcessor(
                    proc, ' '.join(['process', str(ind)]))
            proc.process(self)

    @xft.utils.profiled()
    def update_pedigree(self):
        pass  # TODO

    # generation is immutable except through Simulation().increment_generation()
    @property
    def generation(self):
        return self._generation

    # erase current-generation specific cached properties
    @xft.utils.profiled()
    def increment_generation(self):
        self._current_afs_empirical = None
        self._current_std_genotypes = None
        self._current_std_haplotypes = None
        self._generation += 1

    @property
    def haplotypes(self):
        if self.generation in self.haplotype_store.keys():
            return self.haplotype_store[self.generation]
        else:
            return None

    @haplotypes.setter
    def haplotypes(self, value):
        self.haplotype_store[self.generation] = value

    @property
    def phenotypes(self):
        if self.generation in self.phenotype_store.keys():
            return self.phenotype_store[self.generation]
        else:
            return None

    @phenotypes.setter
    def phenotypes(self, value):
        self.phenotype_store[self.generation] = value

    @property
    def mating(self):
        if self.generation in self.mating_store.keys():
            return self.mating_store[self.generation]
        else:
            return None

    @mating.setter
    def mating(self, value):
        self.mating_store[self.generation] = value

    @property
    def parent_mating(self):
        if (self.generation - 1) in self.mating_store.keys():
            return self.mating_store[(self.generation - 1)]
        else:
            return None

    @property
    def parent_haplotypes(self):
        if (self.generation - 1) in self.haplotype_store.keys():
            return self.haplotype_store[(self.generation - 1)]
        else:
            return None

    @property
    def parent_phenotypes(self):
        if (self.generation - 1) in self.phenotype_store.keys():
            return self.phenotype_store[(self.generation - 1)]
        else:
            return None

    def move_forward(self, n_generations):
        pass

    @property
    def results(self):
        if self.generation in self.results_store.keys():
            return self.results_store[self.generation]
        else:
            return None

    # generation specific properties that are overwritten
    # by increment_generation()
    @property
    def current_afs_empirical(self):
        if self._current_afs_empirical is None:
            return self.haplotypes.xft.af_empirical
        else:
            return self._current_afs_empirical

    @property
    def current_std_genotypes(self):
        if self._current_std_genotypes is None:
            if self.control['standardization'] == 'hardy_weinberg_empirical':
                return self.haplotypes.xft.to_diploid_standardized(af=self.current_afs_empirical,
                                                                   scale=False)
            else:
                raise NotImplementedError()
        else:
            return self._current_std_genotypes

    @property
    def current_std_phenotypes(self):
        if self._current_std_phenotypes is None:
            return self.phenotypes.xft.standardize()
        else:
            return self._current_std_phenotypes

    # def __repr__(self):
        # pass
