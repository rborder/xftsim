import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
from scipy import interpolate

import xftsim as xft


@xr.register_dataarray_accessor("xft")
class XftAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        # haplotype array
        if self._obj.dims == ('sample', 'variant'):
            self._array_type = 'HaplotypeArray'
            self._non_annotation_vars = [
                'variant', 'vid', 'chrom', 'zero_allele', 'one_allele', 'af', 'hcopy', 'pos_bp', 'pos_cM']
            self._variant_vars = ['vid', 'chrom',
                                  'zero_allele', 'one_allele', 'af']
            self._sample_vars = ['iid', 'fid', 'sex']
        elif self._obj.dims == ('sample', 'component'):
            self._array_type = 'componentArray'
            self._component_vars = ['phenotype_name',
                                    'component_name', 'vorigin_relative']
            self._sample_vars = ['iid', 'fid', 'sex']
        else:
            raise NotImplementedError

    @property
    def _row_dim(self):
        return self._obj.dims[0]

    @property
    def _col_dim(self):
        return self._obj.dims[1]

    @property
    def shape(self):
        return self._obj.shape

    @property
    def n(self):
        return self._obj.shape[0]

    @property
    def data(self):
        return self._obj.data

    #### functions for constructing XftIndex objects ####
    def get_sample_indexer(self):
        if self._row_dim != 'sample':
            raise NotImplementedError
        return xft.index.SampleIndex(
            iid=self._obj.coords['sample'].iid.data,
            fid=self._obj.coords['sample'].fid.data,
            sex=self._obj.coords['sample'].sex.data,
        )

    def set_sample_indexer(self, value):
        raise NotImplementedError

    def get_variant_indexer(self):
        if self._col_dim != 'variant':
            raise NotImplementedError
        annotations = self.get_annotation_dict()
        if len(annotations) == 0:
            annotation_array = None
            annotation_names = None
        else:
            annotation_array, annotation_names = annotations.items()
        return xft.index.HaploidVariantIndex(
            vid=self._obj.coords['variant'].vid.data,
            chrom=self._obj.coords['variant'].chrom.data,
            zero_allele=self._obj.coords['variant'].zero_allele.data,
            one_allele=self._obj.coords['variant'].one_allele.data,
            af=self._obj.coords['variant'].af.data,
            pos_bp=self._obj.coords['variant'].pos_bp.data,
            pos_cM=self._obj.coords['variant'].pos_cM.data,
            annotation_array=annotation_array,
            annotation_names=annotation_names)

    def set_variant_indexer(self, value):
        raise NotImplementedError

    def get_component_indexer(self):
        if self._col_dim != 'component':
            raise NotImplementedError
        return xft.index.ComponentIndex(
            phenotype_name=self._obj.coords['component'].phenotype_name.data,
            component_name=self._obj.coords['component'].component_name.data,
            vorigin_relative=self._obj.coords['component'].vorigin_relative.data,
        )

    def reindex_components(self, value):
        # ugly as hell, works for now
        return PhenotypeArray(self._obj.data,
                              component_indexer=value,
                              sample_indexer=self.get_sample_indexer(),
                              )
        # self._obj['phenotype_name'] = value.phenotype_name
        # self._obj['component_name'] = value.component_name
        # self._obj['vorigin_relative'] = value.vorigin_relative

    def get_row_indexer(self):
        if self._row_dim == 'sample':
            return self.get_sample_indexer()
        else:
            raise TypeError

    def set_row_indexer(self):
        raise NotImplementedError

    def get_column_indexer(self):
        if self._col_dim == 'variant':
            return self.get_variant_indexer()
        elif self._col_dim == 'component':
            return self.get_component_indexer()
        else:
            raise TypeError

    def set_column_indexer(self, value):
        if self._col_dim == 'variant':
            return self.set_variant_indexer(value)
        elif self._col_dim == 'component':
            return self.set_component_indexer(value)
        else:
            raise TypeError

    @property
    def row_vars(self):
        return self.get_row_indexer()._coord_variables

    @property
    def column_vars(self):
        return self.get_column_indexer()._coord_variables

    # accessors for pd.MultiIndex objects
    @property
    def sample_mindex(self):
        if self._row_dim != 'sample':
            raise NotImplementedError
        df = pd.DataFrame.from_dict(dict(
            iid=self._obj.coords['sample'].iid.data,
            fid=self._obj.coords['sample'].fid.data,
            sex=self._obj.coords['sample'].sex.data,
        ))
        return pd.MultiIndex.from_frame(df)

    @property
    def component_mindex(self):
        if self._col_dim != 'component':
            raise NotImplementedError
        df = pd.DataFrame.from_dict(dict(
            phenotype_name=self._obj.coords['component'].phenotype_name.data,
            component_name=self._obj.coords['component'].component_name.data,
            vorigin_relative=self._obj.coords['component'].vorigin_relative.data
        ))
        return pd.MultiIndex.from_frame(df)

    def standardize(self):
        out = self._obj.copy()
        out.data = xft.utils.standardize_array(self._obj.data)
        return out

    ############ HaplotypeArray properties ############

    def interpolate_cM(self,
                       rmap_df: pd.DataFrame = xft.data.get_ceu_map(),
                       **kwargs):
        if self._col_dim != 'variant':
            raise TypeError
        chroms = np.unique(self._obj.chrom.values.astype(int)).astype(str)
        for chrom in chroms:  
            rmap_chrom = rmap_df[rmap_df['Chromosome']=='chr'+chrom]
            interpolator = interpolate.interp1d(x = rmap_chrom['Position(bp)'].values,
                                                y = rmap_chrom['Map(cM)'].values,
                                                **kwargs)
            self._obj.pos_cM[self._obj.chrom==chrom] = interpolator(self._obj.pos_bp[self._obj.chrom==chrom])

    def use_empirical_afs(self):
        if self._col_dim != 'variant':
            raise TypeError
        self._obj['af'] = self.af_empirical

    @property
    def diploid_vid(self):
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.vid[::2]

    @property
    def diploid_chrom(self):
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.chrom[::2]

    @property
    def generation(self):
        if self._col_dim != 'variant':
            raise TypeError
        return self.attrs['generation']

    @property
    def af_empirical(self):
        if self._col_dim != 'variant':
            raise TypeError
        hap_AF = self._obj.mean(axis=0)
        AF = np.asarray(hap_AF.data[0::2] + hap_AF.data[1::2]) / 2.
        return AF

    @property
    def maf_empirical(self):
        if self._col_dim != 'variant':
            raise TypeError
        tmp = self.af_empirical
        tmp2 = 1 - tmp
        return np.where(tmp < tmp2, tmp, tmp2)

    @property
    def m(self):
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.shape[1] // 2

    def to_diploid(self):
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj[:, 0::2] + self._obj[:, 1::2]

    def to_diploid_standardized(self, af: NDArray, scale: bool):
        if self._col_dim != 'variant':
            raise TypeError
        if scale:
            return xft.utils.standardize_array_hw(self._obj.data, af) / np.sqrt(self.m)
        else:
            return xft.utils.standardize_array_hw(self._obj.data, af)

    def get_annotation_dict(self):
        if self._col_dim != 'variant':
            raise TypeError
        return {x[0]: x[1].values for x in self._obj.coords.variables.items() if 'variant' in x[1].dims and x[0] not in self._non_annotation_vars}

    def get_non_annotation_dict(self):
        if self._col_dim != 'variant':
            raise TypeError
        return {x[0]: x[1].values for x in self._obj.coords.variables.items() if 'variant' in x[1].dims and x[0] in self._variant_vars}

    # component index properties / methods
    def grep_component_index(self, keyword: str = 'phenotype'):
        if self._col_dim != 'component':
            raise TypeError
        pheno_cols = self._obj.component_name.values[self._obj.component_name.str.contains(
            keyword).values]
        component_index = self._obj.xft.get_component_indexer()[
            dict(component_name=pheno_cols)]
        return component_index

    @property
    def k_total(self):
        if self._col_dim != 'component':
            raise TypeError
        return self.shape[1]  # number of all phenotype components

    @property
    def k_phenotypes(self):
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.phenotype_name).shape[0]

    @property
    def all_phenotypes(self):
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.phenotype_name)

    @property
    def k_components(self):
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.component_name).shape[0]

    @property
    def all_components(self):
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.component_name)

    @property
    def k_relative(self):
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.vorigin_relative).shape[0]

    @property
    def all_relatives(self):
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.vorigin_relative)

    @property
    def k_current(self):  # number of all current-gen specific components
        if self._col_dim != 'component':
            raise TypeError
        return np.sum(self._obj.vorigin_relative == -1)

    def get_k_rel(self, rel):
        if self._col_dim != 'component':
            raise TypeError
        return np.sum(self._obj.vorigin_relative == rel)

    @property
    def depth(self):  # generational depth from binary relative encoding
        if self._col_dim != 'component':
            raise TypeError
        if len(self.vorigin_relative) != 0:
            return math.floor(math.log2(np.max(self._obj.vorigin_relative) + 2)) + 1
        else:
            return np.NaN

    def split_by_phenotype(self):
        if self._col_dim != 'component':
            raise TypeError
        return {phenotype: pheno.loc[:, pheno.phenotype_name == phenotype] for phenotype in self.all_phenotypes}

    def split_by_component(self):
        if self._col_dim != 'component':
            raise TypeError
        return {component: pheno.loc[:, pheno.component_name == component] for component in self.all_components}

    def split_by_vorigin(self):
        if self._col_dim != 'component':
            raise TypeError
        return {vorigin: pheno.loc[:, pheno.vorigin_relative == vorigin] for vorigin in self.all_relatives}

    def split_by_phenotype_vorigin(self):
        if self._col_dim != 'component':
            raise TypeError
        return {(phenotype, vorigin): pheno.loc[:, (pheno.phenotype_name == phenotype) ^ (pheno.vorigin_relative == vorigin)] for phenotype in self.all_phenotypes for vorigin in self.all_relatives}

    def as_pd(self, prettify: bool = True):
        if self._col_dim != 'component':
            raise TypeError
        component_mind = self.component_mindex
        if prettify:
            fr = component_mind.to_frame()
            fr['vorigin_relative'].replace(
                [-1, 0, 1], ['proband', 'mother', 'father'], inplace=True)
            component_mind = pd.MultiIndex.from_frame(fr)
        return xr.DataArray(self._obj.data,
                            dims=('sample', 'component'),
                            coords={'sample': self.sample_mindex,
                                    'component': component_mind},
                            ).to_pandas()

    def __getitem__(self, *args):
        # indexing with a dict
        argv = args[0]
        # argv = args[0]
        if isinstance(argv, dict):
            row_dict = {key: value for (
                key, value) in argv.items() if key in self.row_vars}
            col_dict = {key: value for (
                key, value) in argv.items() if key in self.column_vars}
            # TODO possible gotcha with extra args
            if len(row_dict) == 0:
                row_dict = slice(None)
            if len(col_dict) == 0:
                col_dict = slice(None)

            row_indices = self.get_row_indexer()[row_dict].unique_identifier
            column_indices = self.get_column_indexer()[
                col_dict].unique_identifier
        # indexing with two postional args
        else:
            # argv = tuple(*(args[0]))
            assert len(argv) == 2, "provide 2 positional arguments"
            # row index
            if argv[0] is None or argv[0] is slice(None):
                row_indices = slice(None)
            elif not isinstance(argv[0], xft.index.SampleIndex):
                raise KeyError
            else:
                row_indices = argv[0].unique_identifier
            # col index
            isvarindex_1 = isinstance(argv[1], xft.index.HaploidVariantIndex)
            isvarindex_2 = isinstance(argv[1], xft.index.DiploidVariantIndex)
            isvarindex = (isvarindex_1 or isvarindex_2) and (
                self._col_dim == 'variant')
            iscomindex = (isinstance(
                argv[1], xft.index.ComponentIndex) and self._col_dim == 'component')
            # print(argv)
            # print(not iscomindex)
            # print(not isinstance(argv[1], xft.index.ComponentIndex))
            if argv[1] is None or argv[1] is slice(None) or argv[1] is slice(None, None, None):
                column_indices = slice(None)
            elif (not iscomindex) and (not isvarindex):
                raise KeyError
            else:
                column_indices = argv[1].unique_identifier
            # indexing by XftIndex
        return self._obj.loc[row_indices, column_indices]

    def __setitem__(self, args, data):
        # indexing with a dict
        argv = args  # [0]
        # argv = args[0]
        if isinstance(argv, dict):
            row_dict = {key: value for (
                key, value) in argv.items() if key in self.row_vars}
            col_dict = {key: value for (
                key, value) in argv.items() if key in self.column_vars}
            # TODO possible gotcha with extra args
            if len(row_dict) == 0:
                row_dict = slice(None)
            if len(col_dict) == 0:
                col_dict = slice(None)

            row_indices = self.get_row_indexer()[row_dict].unique_identifier
            column_indices = self.get_column_indexer()[
                col_dict].unique_identifier
        # indexing with two postional args
        else:
            # argv = tuple(*(args[0]))
            assert len(argv) == 2, "provide 2 positional arguments"
            # row index
            if argv[0] is None or argv[0] is slice(None):
                row_indices = slice(None)
            elif not isinstance(argv[0], xft.index.SampleIndex):
                raise KeyError
            else:
                row_indices = argv[0].unique_identifier
            # col index
            isvarindex_1 = isinstance(argv[1], xft.index.HaploidVariantIndex)
            isvarindex_2 = isinstance(argv[1], xft.index.DiploidVariantIndex)
            isvarindex = (isvarindex_1 or isvarindex_2) and (
                self._col_dim == 'variant')
            iscomindex = (isinstance(
                argv[1], xft.index.ComponentIndex) and self._col_dim == 'component')

            if argv[1] is None or argv[1] is slice(None) or argv[1] is slice(None, None, None):
                column_indices = slice(None)
            elif (not iscomindex) and (not isvarindex):
                raise KeyError
            else:
                column_indices = argv[1].unique_identifier
            # indexing by XftIndex
        self._obj.loc[row_indices, column_indices] = data


class HaplotypeArray:
    def __new__(cls,
                # n x 2m array of binary haplotypes
                haplotypes: NDArray[Shape["*, *"], Int8] = None,
                variant_indexer: xft.index.HaploidVariantIndex = None,
                sample_indexer: xft.index.SampleIndex = None,
                generation: int = 0,
                n: int = None,
                m: int = None,
                ):
        # obtain n,m if missing
        # assert (variant_indexer is not None) ^ (m is not None) ^ (h)), "provide variant_indexer OR m"
        # assert (sample_indexer is not None) ^ (n is not None), "provide sample_indexer OR n"
        if haplotypes is not None:
            assert haplotypes.shape[1] % 2 == 0
            n, m = haplotypes.shape
            m = m // 2
        # populate variant_indexer if missing
        if variant_indexer is None:
            variant_indexer = xft.index.DiploidVariantIndex(
                m=m, n_chrom=np.min([22, m])).to_haploid()
            if haplotypes is not None:
                warnings.warn(
                    'Using empirical allele frequencies as variant_indexer not provided')
                tmp = np.mean(haplotypes, 0)
                variant_indexer.af = np.repeat((tmp[0::2] + tmp[1::2]) * .5, 2)

        # populate sample_indexer if missing
        if sample_indexer is None:
            sample_indexer = xft.index.SampleIndex(n=n, generation=generation)
        # populate haplotypes with NaN if not provided
        if haplotypes is None:
            warnings.warn('Defaulting allele counts to -1', stacklevel=2)
            data = np.full((sample_indexer.n, variant_indexer.m * 2),
                           fill_value=-1, dtype=np.int8)
        else:
            # assert type(haplotypes) is np.ndarray, "haplotypes must be ndarray"
            data = haplotypes.astype(np.int8)

        coord_dict = sample_indexer.coord_dict.copy()
        coord_dict.update(variant_indexer.coord_dict)
        return xr.DataArray(data=data,
                            dims=['sample', 'variant'],
                            coords=coord_dict,
                            name='HaplotypeArray',
                            attrs={
                                'generation': generation,
                            })


class PhenotypeArray:
    def __new__(cls,
                # n x 2m array of binary haplotypes
                components: NDArray[Shape["*, *"], Float] = None,
                component_indexer: xft.index.ComponentIndex = None,
                sample_indexer: xft.index.SampleIndex = None,
                generation: int = 0,
                n: int = None,
                k_total: int = None,
                ):
        # ensure components is conformable with indexers
        if components is not None:
            assert n is None, "Provide n OR components"
            assert k_total is None, "Provide k_total OR components"
            # todo verify this doesn't induce copy
            components = np.array(components)
            if sample_indexer is not None:
                assert components.shape[0] == sample_indexer.n, "Noncomformable sample_indexer"
            if component_indexer is not None:
                assert components.shape[1] == component_indexer.k_total, "Noncomformable component_indexer"
        # obtain dimensions if necessary
        if k_total is not None:
            assert component_indexer is None, "Provide k_total OR component_indexer"
            component_indexer = xft.index.ComponentIndex(k_total=k_total)
        if n is not None:
            assert sample_indexer is None, "Provide n OR sample_indexer"
            sample_indexer = xft.index.SampleIndex(n=n)
        k_total, n = component_indexer.k_total, sample_indexer.n
        # initialize component array if necessary
        if components is None:
            components = np.full((n, k_total), fill_value=np.NaN)

        coord_dict = sample_indexer.coord_dict.copy()
        coord_dict.update(component_indexer.coord_dict)
        return xr.DataArray(data=components,
                            dims=['sample', 'component'],
                            coords=coord_dict,
                            name='PhenotypeArray',
                            attrs={
                                'generation': generation,
                            })

    @staticmethod
    def from_product(
        phenotype_name: Iterable,
        component_name: Iterable,
        vorigin_relative: Iterable,
        components: xr.DataArray = None,
        sample_indexer: xft.index.SampleIndex = None,
        generation: int = None,
        haplotypes: xr.DataArray = None,
        n: int = None,
    ) -> xr.DataArray:
        # use either haplotypes xOR sample_indexer/generation xOR n/generation
        bool_gsi = bool(generation is not None and sample_indexer is not None)
        bool_h = bool(haplotypes is not None)
        bool_n = bool(n is not None and generation is not None)
        assert bool_gsi ^ bool_h ^ bool_n
        if bool_n:
            sample_indexer = xft.index.SampleIndex(n=n, generation=generation)
        elif bool_h:
            generation = haplotypes.xft.generation
            sample_indexer = haplotypes.xft.get_sample_indexer()
        component_indexer = xft.index.ComponentIndex.from_product(
            phenotype_name, component_name, vorigin_relative)
        return PhenotypeArray(
            components=components,
            component_indexer=component_indexer,
            sample_indexer=sample_indexer,
            generation=generation,
        )

    @staticmethod
    def _test():
        generation = 0
        n = 3
        m = 10
        n_chrom = 10
        haplotypes = np.full((n, m * 2), fill_value=-1, dtype=np.int8)
        variant_indexer = xft.index.DiploidVariantIndex(
            m=m, n_chrom=n_chrom).to_haploid()
        sample_indexer = xft.index.SampleIndex(n=n, generation=generation)
        HaplotypeArray(haplotypes, generation=generation)
