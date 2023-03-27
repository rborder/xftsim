import warnings
import numpy as np
import numba as nb
import pandas as pd
import dask.array as da
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
import pandas_plink as pp
from dask.diagnostics import ProgressBar

import xftsim as xft


@nb.njit
def _genotypes_to_pseudo_haplotypes(genotypes):
    haplotypes = np.zeros((genotypes.shape[0],genotypes.shape[1]*2), dtype=np.int8)
    zero_one = np.array([0,1], dtype=np.int8)
    one_one = np.array([1,1], dtype=np.int8)
    zero_zero = np.array([1,1], dtype=np.int8)
    for j in nb.prange(genotypes.shape[1]):
        for i in range(genotypes.shape[0]):
            if genotypes[i,j]==1:
                haplotypes[i,(2*j):(2*j+2)] = np.random.permutation(zero_one)
            elif genotypes[i,j]==2:
                haplotypes[i,(2*j):(2*j+2)] = one_one
    return haplotypes


def genotypes_to_pseudo_haplotypes(x):
    out = x[:,np.repeat(np.arange(x.shape[1]),2)]
    out.values = _genotypes_to_pseudo_haplotypes(x.values)
    return out



def plink1_variant_index(ppxr: xr.DataArray) -> xft.index.DiploidVariantIndex:
    if np.all(ppxr.cm==0):
        cm = np.full_like(ppxr.cm, fill_value=np.NaN)
    return xft.index.DiploidVariantIndex(
                                         vid = ppxr.snp.values,
                                         chrom = ppxr.chrom.values,
                                         zero_allele = ppxr.a0.values,
                                         one_allele = ppxr.a1.values,
                                         pos_bp = ppxr.pos,
                                         pos_cM = cm,
                                         )

def plink1_sample_index(ppxr: xr.DataArray, 
                        generation: int = 0) -> xft.index.SampleIndex:
    return xft.index.SampleIndex(
                                 iid = ppxr.iid.values.astype(str),
                                 fid = ppxr.fid.values.astype(str),
                                 sex = 2 - ppxr.gender.values.astype(int),
                                 generation = generation,
                                 )


def read_plink1_as_pseudohaplotypes(path: str):
    delayed_plink = pp.read_plink1_bin(path).astype(np.int8)
    delayed_plink_dask = pp.read_plink(path)
    dip_variant_index = plink1_variant_index(delayed_plink)
    sample_index = plink1_sample_index(delayed_plink)

    hap_array_template = da.empty_like(delayed_plink_dask[2].T.astype(np.int8))
    hap_array_template = hap_array_template[:,np.repeat(np.arange(hap_array_template.shape[1]),2)]
    haploid = da.map_blocks(_genotypes_to_pseudo_haplotypes, 
                            delayed_plink_dask[2].T,
                            chunks = hap_array_template.chunks,
                            dtype = np.int8)
    haplotypes = xft.struct.HaplotypeArray(haplotypes= haploid, 
                                       variant_indexer = dip_variant_index.to_haploid(),
                                       sample_indexer = sample_index,
                                       generation = 0)
    return haplotypes



    vindex = hh.xft.get_variant_indexer().to_diploid()
    sindex = hh.xft.get_sample_indexer()
    data = da.asarray(hh.xft.to_diploid()).astype(np.float32)
    data_array = xr.DataArray(data,
                              dims=['sample','variant'],
                              coords = dict(
                                            sample  = ("sample", sindex.iid.astype(str)),
                                            fid     = ("sample", sindex.fid.astype(str)),
                                            gender  = ("sample", 2 - sindex.sex.astype(int)),
                                            variant = ("variant", vindex.vid.astype(str)),
                                            snp     = ("variant", vindex.vid.astype(str)),
                                            chrom   = ("variant", vindex.chrom.astype(str)),
                                            a0      = ("variant", vindex.zero_allele.astype(str)),
                                            a1      = ("variant", vindex.one_allele.astype(str)),
                                            )
                              )
    pp.write_plink1_bin(data_array, path, verbose=verbose)


def write_to_plink1(hh: xr.DataArray, path: str, verbose: bool = True):
    vindex = hh.xft.get_variant_indexer().to_diploid()
    sindex = hh.xft.get_sample_indexer()
    data = da.asarray(hh.xft.to_diploid()).astype(np.float32)
    data_array = xr.DataArray(data,
                              dims=['sample','variant'],
                              coords = dict(
                                            sample  = ("sample", sindex.iid.astype(str)),
                                            fid     = ("sample", sindex.fid.astype(str)),
                                            gender  = ("sample", 2 - sindex.sex.astype(int)),
                                            variant = ("variant", vindex.vid.astype(str)),
                                            snp     = ("variant", vindex.vid.astype(str)),
                                            chrom   = ("variant", vindex.chrom.astype(str)),
                                            pos     = ("variant", vindex.pos_bp.astype(np.int32)),
                                            cm     = ("variant", vindex.pos_cM.astype(np.float64)),
                                            a0      = ("variant", vindex.zero_allele.astype(str)),
                                            a1      = ("variant", vindex.one_allele.astype(str)),
                                            )
                              )
    pp.write_plink1_bin(data_array, path+'.bed', verbose=verbose)



def load_haplotype_zarr(path: str, 
                        compute: bool = True) -> xr.DataArray:
    if compute:
        with ProgressBar():
            return xr.open_dataset(path,engine='zarr').HaplotypeArray.compute()
    else:
        return xr.open_dataset(path,engine='zarr').HaplotypeArray

def save_haplotype_zarr(haplotypes: xr.DataArray,
                        path: str,
                        ) -> None:
    haplotypes.to_dataset().to_zarr(path)
