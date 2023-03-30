import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
# from dataclasses import dataclass, field
import networkx as nx

import xftsim as xft


class Pedigree:
    """
    A class representing a pedigree as a graph.

    Attributes
    ----------
    G : nx.DiGraph
        The directed graph representing the pedigree.
    _generation : dict
        A dictionary containing node generations.
    _fid : dict
        A dictionary containing node family IDs.
    _generational_depth : int
        The generational depth of the tree.

    Methods
    -------
    generation(K: int):
        Returns the subgraph of nodes with generation K.
    generations(gens):
        Returns the subgraph of nodes with generations in the given iterable.
    current_generation(K):
        Returns the subgraph of nodes in the current generation.
    most_recent_K_generations():
        Returns the subgraph of nodes in the most recent K generations.
    _add_edges_from_arrays(x, y):
        Adds edges from arrays x and y.
    add_offspring(mating: xft.mate.MateAssignment):
        Adds offspring nodes and edges to the pedigree based on a MateAssignment object.
    _get_trios():
        TODO
    """
    def __init__(self,
                 founder_sample_index: xft.index.SampleIndex,
                 founder_generation=0,
                 ):
        """
        Initialize the Pedigree object with founder samples and their generation.

        Parameters
        ----------
        founder_sample_index : xft.index.SampleIndex
            The founder samples index.
        founder_generation : int, optional
            The generation of the founder samples (default is 0).
        """
        self.G = nx.DiGraph()
        self.G.add_nodes_from(founder_iids)
        self._generation = {
            node: founder_generation for node in founder_sample_index.unique_identifier}
        # self._parents = }{
        if founder_fids is not None:
            self._fid = {node: fid for (node, fid) in zip(
                founder_iids, founder_fids)}
        else:
            self._fid = {}
        self._generational_depth = founder_generation

    def generation(self, K: int):
        """
        Returns the subgraph of nodes with generation K.

        Parameters
        ----------
        K : int
            The generation number.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes with generation K.
        """
        return nx.subgraph_view(ped.G, filter_node=lambda x: ped._generation[x] == K)

    def generations(self, gens):
        """
        Returns the subgraph of nodes with generations in the given iterable.

        Parameters
        ----------
        gens : iterable
            An iterable containing generations.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes with generations in the given iterable.
        """
        return nx.subgraph_view(ped.G, filter_node=lambda x: ped._generation[x] in gens)

    def current_generation(self, K):
        """
        Returns the subgraph of nodes in the current generation.

        Parameters
        ----------
        K : int
            The generation number.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes in the current generation.
        """
        return self.generation(self._generational_depth)

    def most_recent_K_generations(self):
        """
        Returns the subgraph of nodes in the most recent K generations.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes in the most recent K generations.
        """
        return self.generations(range(self._generational_depth, self._generational_depth - K, -1))

    def _add_edges_from_arrays(self, x, y):
        """
        Adds edges from arrays x and y.

        Parameters
        ----------
        x : array-like
            The source nodes for the edges.
        y : array-like
            The target nodes for the edges.
        """
        self.G.add_edges_from([(xx, yy) for (xx, yy) in zip(x, y)])

    def add_offspring(self,
                      mating: xft.mate.MateAssignment,
                      ):
        """
        Adds offspring nodes and edges to the pedigree based on a MateAssignment object.

        Parameters
        ----------
        mating : xft.mate.MateAssignment
            The MateAssignment object containing mating information.
        """
        self.G.add_nodes_from(mating.offspring_iids)
        self._generation.update(
            {node: mating.generation for node in mating.offspring_iids})
        self._fid.update({node: fid for (node, fid) in zip(
            mating.offspring_iids, mating.offspring_fids)})
        self._add_edges_from_arrays(
            mating.maternal_iids, mating.offspring_iids)
        self._add_edges_from_arrays(
            mating.paternal_iids, mating.offspring_iids)
        if mating.generation + 1 > self._generational_depth:
            self._generational_depth = mating.generation + 1

    def _get_trios(self):
        """
        TODO: Implement this method.
        """
        raise NotImplementedError


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
