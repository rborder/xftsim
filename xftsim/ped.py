import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
from dataclasses import dataclass, field
import networkx as nx

import xftsim as xft


class Pedigree:
    def __init__(self, 
        founder_sample_index: xft.index.SampleIndex,
        founder_generation = 0,
        ):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(founder_iids)
        self._generation = {node:founder_generation for node in founder_sample_index.unique_identifier}
        # self._parents = }{
        if founder_fids is not None:
            self._fid = {node:fid for (node,fid) in zip(founder_iids,founder_fids)}
        else:
            self._fid = {}
        self._generational_depth = founder_generation

    def generation(self, K: int):
        return nx.subgraph_view(ped.G, filter_node= lambda x: ped._generation[x] == K)

    def generations(self, gens):
        return nx.subgraph_view(ped.G, filter_node= lambda x: ped._generation[x] in gens)

    def current_generation(self, K):
        return self.generation(self._generational_depth)

    def most_recent_K_generations(self):
        return self.generations(range(self._generational_depth, self._generational_depth - K, -1))

    def _add_edges_from_arrays(self, x, y):
        self.G.add_edges_from([(xx, yy) for (xx, yy) in zip(x, y)])

    def add_offspring(self, 
        mating: xft.mate.MateAssignment, 
        ):

        self.G.add_nodes_from(mating.offspring_iids)
        self._generation.update({node:mating.generation for node in mating.offspring_iids})
        self._fid.update({node:fid for (node,fid) in zip(mating.offspring_iids, mating.offspring_fids)})
        self._add_edges_from_arrays(mating.maternal_iids, mating.offspring_iids)
        self._add_edges_from_arrays(mating.paternal_iids, mating.offspring_iids)
        if mating.generation + 1 > self._generational_depth:
            self._generational_depth = mating.generation + 1

    def _get_trios(self):
        pass ## TODO


# NN=100
# test_iids = np.array(['0_' + str(i) for i in range(NN)])
# ped = Pedigree(test_iids)

# mothers = np.random.permutation(np.repeat(test_iids[0::2],2))
# fathers = np.random.permutation(np.repeat(test_iids[1::2],2))
# offspring = np.array(['1_' + str(i) for i in range(NN)])
# mating = MateAssignment(0, mothers, fathers, offspring)
# ped.add_offspring(mating)


# mothers = np.random.permutation(np.repeat(offspring[0::2],2))
# fathers = np.random.permutation(np.repeat(offspring[1::2],2))
# offspring = np.array(['2_' + str(i) for i in range(NN)])
# mating = MateAssignment(0, mothers, fathers, offspring)
# ped.add_offspring(mating)
