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
    """
    A class for running an xft simulation.

    Attributes
    ----------
    mating_regime : xft.mate.MatingRegime
        Mating regime.
    recombination_map : xft.reproduce.RecombinationMap
        Recombination map.
    architecture : xft.arch.Architecture
        Phenogenetic architecture.
    statistics : Iterable, optional
        Iterable of statistics to compute each generation, by default empty list.
    post_processors : Iterable, optional
        Iterable of post processors to apply each generation, by default empty list.
    generation : int, optional
        Initial generation, by default -1, corresponding to an uninitialized simulation
    control : Dict, optional
        Control parameters for the simulation, by default an empty dictionary.
    reproduction_method : xft.reproduce.ReproductionMethod, optional
        Reproduction method for the simulation, by default xft.reproduce.Meiosis.
    control : dict
        Control parameters for the simulation
    haplotypes : xr.DataArray
        Haplotypes for the current generation.
    phenotypes : xr.DataArray
        Phenotypes for the current generation.
    mating : xr.DataArray
        Mating information for the current generation.
    parent_mating : xr.DataArray
        Mating information for the previous generation.
    parent_haplotypes : xr.DataArray
        Haplotypes for the previous generation.
    parent_phenotypes : xr.DataArray
        Phenotypes for the previous generation.
    results : xr.DataArray
        Results for the current generation.
    current_afs_empirical : xr.DataArray
        Current empirical allele frequencies.
    current_std_genotypes : xr.DataArray
        Current standardized genotypes.
    current_std_phenotypes : xr.DataArray
        Current standardized phenotypes.
    phenotype_store : Dict[int, xr.DataArray]
        Dictionary storing phenotypes for each generation.
    haplotype_store : Dict[int, xr.DataArray]
        Dictionary storing haplotypes for each generation.
    mating_store : Dict[int, xr.DataArray]
        Dictionary storing mating information for each generation.
    results_store : Dict[int, xr.DataArray]
        Dictionary storing results for each generation.
    pedigree : Any
        Pedigree information for the simulation (currently not implemented).


    Methods
    -------
    run(n_generations: int):
        Run the simulation for a specified number of generations.
    run_generation():
        Run a single generation of the simulation.
    compute_phenotypes():
        Compute phenotypes for the current generation.
    mate():
        Perform mating for the current generation.
    reproduce():
        Perform reproduction for the current generation.
    estimate_statistics():
        Estimate statistics for the current generation.
    process():
        Process the current generation using post-processors.
    update_pedigree():
        Update pedigree information for the current generation.
    increment_generation():
        Increment the current generation.
    move_forward(n_generations: int):
        Move the simulation forward by a specified number of generations.
    """
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
        """
        Initialize a Simulation instance.

        Parameters
        ----------
        founder_haplotypes : xr.DataArray
            Haplotypes for the founder generation.
        mating_regime : xft.mate.MatingRegime
            Mating regime for the simulation.
        recombination_map : xft.reproduce.RecombinationMap
            Recombination map for the simulation.
        architecture : xft.arch.Architecture
            Architecture for the simulation.
        statistics : Iterable, optional
            Iterable of statistics to compute, by default an empty list.
        post_processors : Iterable, optional
            Iterable of post processors to apply, by default an empty list.
        generation : int, optional
            Initial generation, by default -1.
        control : Dict, optional
            Control parameters for the simulation, by default an empty dictionary.
        reproduction_method : xft.reproduce.ReproductionMethod, optional
            Reproduction method for the simulation, by default xft.reproduce.Meiosis.
        """
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
        """
        Run the simulation for a specified number of generations.

        Parameters
        ----------
        n_generations : int
            Number of generations to run the simulation.
        """
        for generation in range(n_generations):
            self.run_generation()

    @xft.utils.profiled(level=0)
    def run_generation(self):
        """
        Run a single generation of the simulation.
        """
        self.increment_generation()
        self.reproduce()
        self.compute_phenotypes()
        self.mate()
        self.update_pedigree()
        self.estimate_statistics()
        self.process()

    @xft.utils.profiled()
    def compute_phenotypes(self):
        """
        Compute phenotypes for the current generation.
        """
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
        """
        Perform mating for the current generation.
        """
        self.mating = self.mating_regime.mate(
            self.haplotypes, self.phenotypes, self.control)

    @xft.utils.profiled()
    def reproduce(self):
        """
        Perform reproduction for the current generation.
        """
        if self.generation == 0:
            pass
        elif self.generation >= 1:
            self.haplotypes = self.reproduction_regime.reproduce(self.parent_haplotypes,
                                                                 self.parent_mating,
                                                                 self.control)

    @xft.utils.profiled()
    def estimate_statistics(self):
        """
        Estimate statistics for the current generation.
        """
        for stat in self.statistics:
            stat.estimate(self)

    @xft.utils.profiled()
    def process(self):
        """
        Apply post-processors to the current generation.
        """
        for ind, proc in enumerate(self.post_processors):
            # allow callables that map xft.Simulation -> None
            if isinstance(proc, Callable):
                proc = xft.proc.PostProcessor(
                    proc, ' '.join(['process', str(ind)]))
            proc.process(self)

    @xft.utils.profiled()
    def update_pedigree(self):
        """
        Update pedigree information (NOT IMPLEMENTED).
        """
        # warnings.warn('update_pedigree() not implemented')
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
