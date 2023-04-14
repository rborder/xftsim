import warnings
import numpy as np
import pandas as pd
import dask.array as da
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Union
from functools import cached_property
import numba as nb
import math
import nptyping as npt

import xftsim as xft

from xftsim.mate import MateAssignment

# todo -- base on DiploidVariantIndex


class RecombinationMap:  # diploid recombination map
    """
    A class to represent a diploid recombination map.
    In the future, will require XftIndex object instead of vid and chrom.


    Parameters
    ----------
    p : float or numpy.ndarray, optional
        Probabilities, either a float or a numpy.ndarray, default is None. A single value
        results in an exchangle map, an array corresponds to probabilities of recombination
        between specified loci
    vindex: xft.index.HaploidVariantIndex | xft.index.DiploidVariantIndex
        Variant index. Only provide if not providing vid / chrom
    vid : NDArray[Shape["*"], Any], optional
        Variant IDs, default is None.
    chrom : NDArray[Shape["*"], Int64], optional
        Chromosomes, default is None.
    """
    def __init__(self,
                 p=None,  # probabilities numpy.ndarray or float
                 vindex: Union[xft.index.HaploidVariantIndex, xft.index.DiploidVariantIndex] = None,
                 vid: NDArray[Shape["*"], Any] = None,  # variant id,
                 chrom: NDArray[Shape["*"], Int64] = None,  # chromosome
                 ):
        if vindex is not None:
            vid = vindex.vid
            chrom = vindex.chrom
        # enforce haploid
        if len(vid) == 2 * len(np.unique(vid)):
            vid = vid[::2]
            chrom = chrom[::2]
        self.vid = vid
        self.chrom = chrom
        self.m = vid.shape[0]
        self._chrom_boundary = np.concatenate(
            [[0], 1+np.where(chrom[1:] != chrom[:-1])[0]])
        if type(p) is float:
            assert p <= 1 and p >= 0, "Provide a valid probability"
            self._probabilities = np.ones(self.m) * p
        if type(p) is np.ndarray:
            assert p.shape[0] == self.m, "p and chrom must agree in length"
            self._probabilities = p

        self._probabilities[self._chrom_boundary] = .5

        self.probabilities = xr.DataArray(
            self._probabilities,
            dims=["variant"],
            coords=[vid],
            name='probabilities',
        )

    @staticmethod
    def constant_map_from_haplotypes(haplotypes=xr.DataArray,
                                     p: np.float64 = .5,
                                     ) -> "RecombinationMap":
        """
        Create a constant recombination map from haplotypes.
        
        Parameters
        ----------
        haplotypes : xr.DataArray
            Haplotypes data array.
        p : np.float64, optional
            Probability, default is 0.5.
        
        Returns
        -------
        RecombinationMap
            A constant recombination map.
        """
        vi = haplotypes.xft.get_variant_indexer()
        return RecombinationMap(p, vid=vi.vid, chrom=vi.chrom)

    @staticmethod
    def variable_map_from_haplotypes_with_cM(haplotypes=xr.DataArray,
                                             ) -> "RecombinationMap":
        """
        Create a variable recombination map from haplotypes with centimorgan distances.
        
        Parameters
        ----------
        haplotypes : xr.DataArray
            Haplotypes data array.
        
        Returns
        -------
        RecombinationMap
            A variable recombination map.
        
        Raises
        ------
        ValueError
            If distance in centimorgans is required and not present in the input.
        """
        vi_dip = haplotypes.xft.get_variant_indexer().to_diploid()
        if np.any(np.isnan(vi_dip.pos_cM)):
            raise ValueError("Distance in centimorgans required"
                             "Try interpolating with `xr.DataArray.xft.interpolate_cM`")
        d = np.diff(vi_dip.pos_cM)
        pp = (1 - np.exp(-2*d/100)) / 2
        pp[vi_dip.chrom[1:]!=vi_dip.chrom[:-1]] = .5 
        pp = np.concatenate([[.5], pp])
        return RecombinationMap(pp, vid=vi_dip.vid, chrom=vi_dip.chrom)

    def __repr__(self):
        df = pd.DataFrame.from_dict(dict(vid=self.vid,
                                         chrom=self.chrom,
                                         p=self._probabilities))
        return df.__repr__()

def transmit_parental_phenotypes(
    mating: MateAssignment,
    parental_phenotypes: xr.DataArray,
    offspring_phenotypes: xr.DataArray,
    control: dict = None,
    ) -> None:
    """
    Transmits parental phenotypes to offspring.
    
    Parameters
    ----------
    mating : MateAssignment
        An object representing mating assignments.
    parental_phenotypes : xr.DataArray
        A data array containing parental phenotypes.
    offspring_phenotypes : xr.DataArray
        A data array containing offspring phenotypes.
    control : dict, optional
        A dictionary containing additional control parameters, default is None.

    Returns
    -------
    None
    """
    # sample indexes (wrt to previous generation) for parents
    # of current generation
    parent_gen_mat_sample_ind = mating.reproducing_maternal_index
    parent_gen_pat_sample_ind = mating.reproducing_paternal_index
    # sample indexes in current generation:
    offspring_gen_mat_sample_ind = None
    offspring_gen_pat_sample_ind = None

    # component index of current generation
    offspring_component_index = parental_phenotypes.xft.get_component_indexer()
    # component indexes (in current generation) for inherited phenotypes
    offspring_gen_maternal_component_ind = offspring_component_index[dict(
        vorigin_relative=0)]
    offspring_gen_paternal_component_ind = offspring_component_index[dict(
        vorigin_relative=1)]
    # component indexes (in previous generation) for inherited phenotypes
    parent_gen_maternal_component_ind = offspring_gen_maternal_component_ind.to_proband()
    parent_gen_paternal_component_ind = offspring_gen_paternal_component_ind.to_proband()

    # transmit maternal components
    maternal_data = parental_phenotypes.xft[parent_gen_mat_sample_ind,
                                            parent_gen_maternal_component_ind].data
    offspring_phenotypes.xft[offspring_gen_mat_sample_ind,
                             offspring_gen_maternal_component_ind] = maternal_data

    # transmit paternal components
    paternal_data = parental_phenotypes.xft[parent_gen_pat_sample_ind,
                                            parent_gen_paternal_component_ind].data
    offspring_phenotypes.xft[offspring_gen_pat_sample_ind,
                             offspring_gen_paternal_component_ind] = paternal_data

    # paternal_data = parental_phenotypes.xft.access_phenotype_components(component_indexer = parent_gen_ppaternal_component_ind,
    #                                                                     sample_indexer = parent_gen_pat_sample_ind)
    # offspring_phenotypes.xft.assign_phenotype_components(data = paternal_data.data,
    #                                                      component_indexer = offspring_gen_paternal_component_ind,
    #                                                      sample_indexer = offspring_gen_pat_sample_ind)

    # offspring_gen_maternal_component_in[d
    # ## no-op if no inheritance
    # if np.all(offspring_phenotypes.vorigin_relative.values==-1):
    #     pass
    # ## else transmit phenotypes
    # else:
    #     ## columns of parent components of current generation in offspring phenotype array
    #     offspring_gen_mat_component = offspring_phenotypes[:,
    #         offspring_phenotypes.vorigin_relative==0].xft.get_component_indexer()
    #     offspring_gen_mat_uid = offspring_gen_mat_component.unique_identifier
    #     offspring_gen_pat_component = offspring_phenotypes[:,
    #         offspring_phenotypes.vorigin_relative==1].xft.get_component_indexer()
    #     offspring_gen_pat_uid = offspring_gen_pat_component.unique_identifier

    #     ## columns of parent components of current generation in parent phenotype array
    #     parent_gen_mat_component_frame = offspring_gen_mat_component.frame
    #     parent_gen_mat_component_frame.vorigin_relative = 0
    #     parent_gen_mat_component = xft.index.ComponentIndex.from_frame(parent_gen_mat_component_frame)
    #     parent_gen_mat_uid = parent_gen_mat_component.unique_identifier

    #     parent_gen_pat_component_frame = offspring_gen_pat_component.frame
    #     parent_gen_pat_component_frame.vorigin_relative = 1
    #     parent_gen_pat_component = xft.index.ComponentIndex.from_frame(parent_gen_pat_component_frame)
    #     parent_gen_pat_uid = parent_gen_pat_component.unique_identifier

    #     ## transmit maternal phenotypes if necessary
    #     if offspring_gen_mat_component.k_total > 0: ## TODO refactor to use assign by index
    #         # offspring_phenotypes.xft.assign_phenotype_components()
    #         #     parental_phenotypes.xft.access_phenotype_components(
    #         #                                                                     component_indexer=parent_gen_mat_component)
    #         # parental_phenotypes.xft.access_phenotype_components(sample_indexer=mating.reproducing_paternal_index,
    #         #                                                     component_indexer=parent_gen_mat_component)

    #         offspring_phenotypes.loc[:,
    #             offspring_gen_mat_uid] =  parental_phenotypes.loc[parent_gen_mat_sample_inds,
    #             parent_gen_mat_uid].data
    #     ## transmit paternal phenotypes if necessary
    #     if offspring_gen_pat_component.k_total > 0: ## TODO refactor to use assign by index
    #         offspring_phenotypes.loc[:,
    #             offspring_gen_pat_uid] =  parental_phenotypes.loc[parent_gen_pat_sample_inds,
    #             parent_gen_pat_uid].data


# maps recombination probabilities to haploid indicies
@nb.njit("int64[:](float64[:])")
def _meiosis_i(p):
    """
    Maps recombination probabilities to haploid indices.

    Parameters
    ----------
    p : numpy.ndarray[float64]
        An array of recombination probabilities.

    Returns
    -------
    numpy.ndarray[int64]
        An array of haploid indices.
    """
    m = p.shape[0]
    output = np.empty(m, dtype=np.int64)
    for j in range(m):
        output[j] = np.random.binomial(1, p[j])
    output = np.cumsum(output) % 2
    output += np.arange(0, 2 * m, 2)
    return output

# meiosis i and ii


@nb.njit("int8[:,:](int8[:,:], int8[:,:], int64, int64, float64[:], int64[:], int64[:], int64[:], int64[:])", parallel=True)
def _meiosis(parental_haplotypes,
             offspring_haplotypes,
             n,
             m_hap,
             recombination_p,
             maternal_inds,
             paternal_inds,
             m_meiosis_ii_inds,
             p_meiosis_ii_inds,
             ):
    """
    Performs meiosis I and II.

    Parameters
    ----------
    parental_haplotypes : numpy.ndarray[int8]
        An array of parental haplotypes.
    offspring_haplotypes : numpy.ndarray[int8]
        An array of offspring haplotypes.
    n : int64
        The number of offspring.
    m_hap : int64
        The number of haploid chromosomes.
    recombination_p : numpy.ndarray[float64]
        An array of recombination probabilities.
    maternal_inds : numpy.ndarray[int64]
        An array of maternal indices.
    paternal_inds : numpy.ndarray[int64]
        An array of paternal indices.
    m_meiosis_ii_inds : numpy.ndarray[int64]
        An array of maternal meiosis II indices.
    p_meiosis_ii_inds : numpy.ndarray[int64]
        An array of paternal meiosis II indices.

    Returns
    -------
    numpy.ndarray[int8]
        An array of offspring haplotypes.
    """
    for i in nb.prange(n):
        # maternal copies:
        m_meiosis_i_inds = _meiosis_i(recombination_p)
        p_meiosis_i_inds = _meiosis_i(recombination_p)
        for j in range(m_hap // 2):
            offspring_haplotypes[i, m_meiosis_ii_inds[j]
                                 ] = parental_haplotypes[maternal_inds[i], m_meiosis_i_inds[j]]
            offspring_haplotypes[i, p_meiosis_ii_inds[j]
                                 ] = parental_haplotypes[paternal_inds[i], p_meiosis_i_inds[j]]
    return offspring_haplotypes


def meiosis(parental_haplotypes: npt.NDArray[npt.Shape["*,*"], npt.Int8],
            recombination_p: npt.NDArray[npt.Shape["*,*"], npt.Float64],
            maternal_inds: npt.NDArray[npt.Shape["*,*"], npt.Int64],
            paternal_inds: npt.NDArray[npt.Shape["*,*"], npt.Int64],
            ) -> npt.NDArray[npt.Shape["*,*"], npt.Int8]:
    """
    Performs meiosis on parental haplotypes.

    Parameters
    ----------
    parental_haplotypes : numpy.ndarray[int8]
        An array of parental haplotypes.
    recombination_p : numpy.ndarray[float64]
        An array of recombination probabilities.
    maternal_inds : numpy.ndarray[int64]
        An array of maternal indices.
    paternal_inds : numpy.ndarray[int64]
        An array of paternal indices.

    Returns
    -------
    numpy.ndarray[int8]
        An array of offspring haplotypes.
    """
    assert (parental_haplotypes.shape[1] //
            2) == recombination_p.shape[0], "incompatable arg dimension"
    assert paternal_inds.shape[0] == maternal_inds.shape[0], "incompatable arg dimension"
    assert np.max(
        maternal_inds) <= parental_haplotypes.shape[0], "incompatable arg dimension"
    assert np.max(
        paternal_inds) <= parental_haplotypes.shape[0], "incompatable arg dimension"
    assert (parental_haplotypes.dtype == np.int8) & (
        recombination_p.dtype == np.float64), "type error"
    assert (maternal_inds.dtype == paternal_inds.dtype == np.int64), "type error"

    n = maternal_inds.shape[0]
    m_hap = parental_haplotypes.shape[1]
    offspring_haplotypes = np.empty((n, m_hap), dtype=np.int8)
    m_meiosis_ii_inds = np.arange(0, m_hap, 2, dtype=np.int64)
    p_meiosis_ii_inds = np.arange(1, m_hap, 2, dtype=np.int64)
    if isinstance(parental_haplotypes, da.Array):
        parental_haplotypes = parental_haplotypes.compute()
    return _meiosis(parental_haplotypes,
                    offspring_haplotypes,
                    n,
                    m_hap,
                    recombination_p,
                    maternal_inds,
                    paternal_inds,
                    m_meiosis_ii_inds,
                    p_meiosis_ii_inds,
                    )


class Meiosis:
    """
    A class representing the process of meiosis.

    Attributes
    ----------
    recombinationMap : RecombinationMap, optional
        A pre-defined recombination map.
    p : float, optional
        A probability used when generating an exchangable recombination map on the fly.

    Methods
    -------
    get_recombination_map(haplotypes):
        Returns the recombination map, either pre-defined or generated on the fly.
    reproduce(parental_haplotypes=None, mating=None, control=None):
        Returns a HaplotypeArray representing the offspring after meiosis.
    """
    def __init__(self,
                 rmap: RecombinationMap = None,
                 p: float = None):

        assert (p is not None) ^ (rmap is not None)
        self.recombinationMap = rmap
        self._p = p

    def get_recombination_map(self, haplotypes):
        """
        Get the recombination map, either pre-defined or generated on the fly.

        Parameters
        ----------
        haplotypes : xr.DataArray
            The haplotype data.

        Returns
        -------
        RecombinationMap
            The recombination map.
        """
        if self.recombinationMap is not None:
            return self.recombinationMap
        else:  # generate recombinationmap on the fly if necessary
            return RecombinationMap.constant_map_from_haplotypes(haplotypes, p=self._p)

    # TODO make sure recombination probability indices agree
    def reproduce(self,
                  parental_haplotypes: xr.DataArray = None,
                  mating: MateAssignment = None,
                  control: dict = None,
                  ):
        """
        Return a HaplotypeArray representing the offspring after meiosis.

        Parameters
        ----------
        parental_haplotypes : xr.DataArray, optional
            The parental haplotype data.
        mating : MateAssignment, optional
            The mate assignment object.
        control : dict, optional
            A dictionary containing control parameters.

        Returns
        -------
        HaplotypeArray
            The HaplotypeArray representing the offspring after meiosis.
        """
        rmap = self.get_recombination_map(parental_haplotypes)
        recombination_p = rmap.probabilities.values
        maternal_inds = xft.utils.match(mating.reproducing_maternal_index.unique_identifier,
                                        parental_haplotypes.sample.values)
        paternal_inds = xft.utils.match(mating.reproducing_paternal_index.unique_identifier,
                                        parental_haplotypes.sample.values)
        return xft.struct.HaplotypeArray(meiosis(parental_haplotypes.data,
                                                 recombination_p,
                                                 maternal_inds,
                                                 paternal_inds,
                                                 ),
                                         variant_indexer=parental_haplotypes.xft.get_variant_indexer(),
                                         sample_indexer=mating.offspring_sample_index,
                                         generation=parental_haplotypes.attrs['generation'] + 1)


# parental_haplotypes
# recombination_p
# maternal_inds
# paternal_inds
