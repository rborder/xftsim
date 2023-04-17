import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
import dask.array as da
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, Tuple
from functools import cached_property
from scipy import interpolate
import xftsim as xft


class GeneticMap:
    """
    Map between physical and genetic distances.

    Parameters
    ----------
    chrom : Iterable
        Chromsomes variants are located on
    pos_bp : Iterable
        Physical positions of variants
    pos_cM : Iterable
        Map distances in cM

    Attributes
    __________
    frame : pd.DataFrame
        Pandas DataFrame with the above columns
    chroms : np.ndarray
        Unique chromosomes present in map
    """
    def __init__(self,
                 chrom: Iterable,
                 pos_bp: Iterable,
                 pos_cM: Iterable):
        self.frame = pd.DataFrame.from_dict(dict(chrom = chrom,
                                                 pos_bp = pos_bp,
                                                 pos_cM = pos_cM))
        self.chroms = np.unique(self.frame.chrom.values.astype(int)).astype(str)

    @classmethod
    def from_pyrho_maps(cls, paths: Iterable, sep='\t', **kwargs) -> "GeneticMap":
        """Construct genetic map objects from maps provided at https://github.com/popgenmethods/pyrho
        Please cite their work if you use their maps.
        
        Parameters
        ----------
        paths : Iterable
            Paths for each chromosome
        sep : str, optional
            Passed to pd.read_csv()
        **kwargs
            Additional arguments to pd.read_csv()
        
        Returns
        -------
        GeneticMap
        """
        gmap = pd.concat([pd.read_csv(path, sep='\t', **kwargs) for path in paths])
        chrom = np.char.lstrip(gmap.Chromosome.values.astype(str),'chr')
        pos_bp = gmap['Position(bp)'].values
        pos_cM = gmap['Map(cM)'].values
        return cls(chrom, pos_bp, pos_cM)


    def interpolate_cM_chrom(self, pos_bp: Iterable, chrom: str, **kwargs):
        """
        Interpolate cM values in a specified chromosome based on genetic map information.

        Parameters
        ----------
        pos_bp : Iterable
            Physical positions for which to interpolate cM values
        chrom : str
            Chromosome on which to interpolate
        **kwargs
            Additional keyword arguments to be passed to scipy.interpolate.interp1d.
        """ 
        subset = self.frame[self.frame.chrom==chrom]
        interpolator = interpolate.interp1d(x = subset.pos_bp.values,
                                            y = subset.pos_cM.values,
                                            **kwargs)
        return interpolator(pos_bp)


        if self._col_dim != 'variant':
            raise TypeError
        chroms = np.unique(self._obj.chrom.values.astype(int)).astype(str)
        for chrom in chroms:  
            rmap_chrom = rmap_df[rmap_df['Chromosome']=='chr'+chrom]
            interpolator = interpolate.interp1d(x = rmap_chrom['Position(bp)'].values,
                                                y = rmap_chrom['Map(cM)'].values,
                                                **kwargs)
            self._obj.pos_cM[self._obj.chrom==chrom] = interpolator(self._obj.pos_bp[self._obj.chrom==chrom])

 
                 

@xr.register_dataarray_accessor("xft")
class XftAccessor:
    """
    Accessor for Xarray DataArrays with specialized functionality for HaplotypeArray
    and PhenotypeArray objects.
    
    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to be accessed.
    
    Attributes
    ----------
    _obj : xarray.DataArray
        The DataArray to be accessed.
    _array_type : str
        The type of the DataArray, either 'HaplotypeArray' or 'componentArray'.
    _non_annotation_vars : list of str
        The non-annotation variables in the DataArray.
    _variant_vars : list of str
        The variant annotation variables in the DataArray.
    _sample_vars : list of str
        The sample annotation variables in the DataArray.
    _component_vars : list of str
        The component annotation variables in the DataArray.
    _row_dim : str
        The label of the row dimension.
    _col_dim : str
        The label of the column dimension.
    shape : tuple
        The shape of the DataArray.
    n : int
        The number of rows in the DataArray.
    data : numpy.ndarray
        The data in the DataArray.
    row_vars: list
        List of coordinate variable names for the row dimension.
    column_vars: list
        List of coordinate variable names for the column dimension.
    sample_mindex: pd.MultiIndex
        MultiIndex object for the 'sample' dimension, containing iid, fid, and sex columns.
    component_mindex: pd.MultiIndex
        MultiIndex object for the 'component' dimension, containing phenotype_name, component_name, and vorigin_relative columns.


    Raises
    ------
    NotImplementedError
        If the DataArray dimensions are not ('sample', 'variant') or ('sample', 'component').
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if self._obj.dims == ('sample', 'variant'):
            self._array_type = 'HaplotypeArray'
            self._non_annotation_vars = [
                'variant', 'vid', 'chrom', 'zero_allele', 'one_allele', 'af', 'hcopy', 'pos_bp', 'pos_cM']
            self._variant_vars = ['vid', 'chrom', 'zero_allele', 'one_allele', 'af']
            self._sample_vars = ['iid', 'fid', 'sex']
        elif self._obj.dims == ('sample', 'component'):
            self._array_type = 'componentArray'
            self._component_vars = ['phenotype_name', 'component_name', 'vorigin_relative']
            self._sample_vars = ['iid', 'fid', 'sex']
        else:
            raise NotImplementedError("Unsupported dimensions for DataArray.")
    
    @property
    def _row_dim(self):
        """
        The label of the row dimension.
        
        Returns
        -------
        str
            The label of the row dimension.
        """
        return self._obj.dims[0]

    @property
    def _col_dim(self):
        """
        The label of the column dimension.
        
        Returns
        -------
        str
            The label of the column dimension.
        """
        return self._obj.dims[1]

    @property
    def shape(self):
        """
        The shape of the DataArray.
        
        Returns
        -------
        tuple
            The shape of the DataArray.
        """
        return self._obj.shape

    @property
    def n(self):
        """
        The number of rows in the DataArray.
        
        Returns
        -------
        int
            The number of rows in the DataArray.
        """
        return self._obj.shape[0]

    @property
    def data(self):
        """
        The data in the DataArray.
        
        Returns
        -------
        numpy.ndarray
            The data in the DataArray.
        """
        return self._obj.data

    #### functions for constructing XftIndex objects ####
    def get_sample_indexer(self):
        """
        Returns an instance of `xft.index.SampleIndex` representing the sample indexer
        constructed from the input data.

        Raises
        ------
        NotImplementedError
            If `_row_dim` is not `'sample'`.

        Returns
        -------
        SampleIndex
            An instance of `xft.index.SampleIndex` constructed from the sample data in the
            input object.
        """
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
        """
        Get the variant indexer of a HaplotypeArray.

        Returns
        -------
        xft.index.HaploidVariantIndex
            A HaploidVariantIndex object.
        """
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
            h_copy=self._obj.coords['variant'].hcopy.data,
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
        """
        Get the component indexer of a PhenotypeArray.

        Returns
        -------
        xft.index.ComponentIndex
            A ComponentIndex object.
        """
        if self._col_dim != 'component':
            raise NotImplementedError
        return xft.index.ComponentIndex(
            phenotype_name=self._obj.coords['component'].phenotype_name.data,
            component_name=self._obj.coords['component'].component_name.data,
            vorigin_relative=self._obj.coords['component'].vorigin_relative.data,
            comp_type=self._obj.coords['component'].comp_type.data,
        )

    def reindex_components(self, value):
        """
        Reindex the components.

        Parameters
        ----------
        value : xft.index.ComponentIndex
            A ComponentIndex object.

        Returns
        -------
        PhenotypeArray
            A new PhenotypeArray object.

        """
        # ugly as hell, works for now
        return PhenotypeArray(self._obj.data,
                              component_indexer=value,
                              sample_indexer=self.get_sample_indexer(),
                              )
        # self._obj['phenotype_name'] = value.phenotype_name
        # self._obj['component_name'] = value.component_name
        # self._obj['vorigin_relative'] = value.vorigin_relative

    def get_row_indexer(self):
        """
        Get the row indexer.

        Returns
        -------
        xft.index.SampleIndex
            A SampleIndex object.

        Raises
        ------
        TypeError
            If the row dimension is not 'sample'.

        """
        if self._row_dim == 'sample':
            return self.get_sample_indexer()
        else:
            raise TypeError

    def set_row_indexer(self):
        raise NotImplementedError

    def get_column_indexer(self):
        """
        Get the column indexer object for the PhenotypeArray object.

        Returns
        -------
        xft.index.Indexer
            The indexer object based on the current column dimension.

        Raises
        ------
        TypeError
            If the current column dimension is not recognized.
        """
        if self._col_dim == 'variant':
            return self.get_variant_indexer()
        elif self._col_dim == 'component':
            return self.get_component_indexer()
        else:
            raise TypeError

    def set_column_indexer(self, value):
        """
        Set the column indexer object for the PhenotypeArray object.

        Parameters
        ----------
        value : xft.index.Indexer
            The new indexer object for the PhenotypeArray object.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the current column dimension is not recognized.
        """
        if self._col_dim == 'variant':
            return self.set_variant_indexer(value)
        elif self._col_dim == 'component':
            return self.set_component_indexer(value)
        else:
            raise TypeError

    @property
    def row_vars(self):
        """
        Get the row coordinate variables for the PhenotypeArray object.

        Returns
        -------
        XftIndex
            The row coordinate variables of the row dimension.
        """
        return self.get_row_indexer()._coord_variables

    @property
    def column_vars(self):
        """
        Get the column coordinate variables for the DataArray object.

        Returns
        -------
        XftIndex
            The column coordinate variables of the current column dimension.
        """
        return self.get_column_indexer()._coord_variables

    # accessors for pd.MultiIndex objects
    @property
    def sample_mindex(self):
        """
        Get the sample multi-index for the PhenotypeArray object.

        Returns
        -------
        pd.MultiIndex
            A multi-index object containing sample IDs, family IDs, and sex information.
        
        Raises
        ------
        NotImplementedError
            If the current row dimension is not 'sample'.
        """
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
        """
        Get a Pandas MultiIndex object for the component dimension.

        Returns
        -------
        pandas.MultiIndex
            MultiIndex object with phenotype_name, component_name, and vorigin_relative
            as index levels.

        Raises
        ------
        NotImplementedError
            If the column dimension is not 'component'.
        """ 
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
                       gmap: GeneticMap,
                       # rmap_df: pd.DataFrame = xft.data.get_ceu_map(),
                       **kwargs):
        """
        Interpolate cM values based on genetic map information.
        Specific to HaplotypeArray objects.

        Parameters
        ----------
        gmap : GeneticMap
            Genetic map data
        **kwargs
            Additional keyword arguments to be passed to scipy.interpolate.interp1d.

        Raises
        ------
        TypeError
            If the column dimension is not 'variant'.
        ValueError
            If not all chromosomes required are present in the genetic map
        """
        if self._col_dim != 'variant':
            raise TypeError
        chroms = np.unique(self._obj.chrom.values.astype(int)).astype(str)
        if not (set(chroms) <= set(gmap.chroms)):
            raise ValueError('Not all chromosomes are present on specified genetic Map')
        for chrom in chroms:  
            self._obj.pos_cM[self._obj.chrom==chrom] = gmap.interpolate_cM_chrom(self._obj.pos_bp[self._obj.chrom==chrom], 
                                      chrom=chrom,
                                      **kwargs)
            # self._obj.pos_cM[self._obj.chrom==chrom] = interpolator(self._obj.pos_bp[self._obj.chrom==chrom])

    def use_empirical_afs(self):
        """
        Sets allele frequencies to the empirical frequencies.
        Specific to HaplotypeArray objects.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        self._obj.af.values = np.repeat(self.af_empirical,2)

    @property
    def diploid_vid(self):
        """
        Diploid variant ID.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Diploid variant IDs.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.vid[::2]

    @property
    def diploid_chrom(self):
        """
        Diploid chromosome numbers.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Diploid chromosome numbers.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.chrom[::2]

    @property
    def generation(self):
        """
        Generation of the data.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        int
            Generation attribute.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self.attrs['generation']

    @property
    def af_empirical(self):
        """
        Empirical allele frequencies.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Empirical allele frequencies.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        hap_AF = self._obj.mean(axis=0)
        AF = np.asarray(hap_AF.data[0::2] + hap_AF.data[1::2]) / 2.
        return AF

    @property
    def maf_empirical(self):
        """
        Empirical minor allele frequencies.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Empirical minor allele frequencies.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        tmp = self.af_empirical
        tmp2 = 1 - tmp
        return np.where(tmp < tmp2, tmp, tmp2)

    @property
    def m(self):
        """
        Return the number of distinct diploid variants.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        int:
            The number of distinct diploid variants in the array.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.shape[1] // 2

    def to_diploid(self):
        """
        Convert the object to a diploid representation by adding the two haplotypes for each variant.
        Specific to HaplotypeArray objects.
        
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        data = self._obj[:, 0::2].data + self._obj[:, 1::2].data
        vind = self.get_variant_indexer().to_diploid()
        sind = self.get_sample_indexer()

        coord_dict = sind.coord_dict.copy()
        coord_dict.update(vind.coord_dict)
        ## convert to dask array if necessary
        return xr.DataArray(data=data,
                            dims=['sample', 'variant'],
                            coords=coord_dict,
                            name='GenotypeArray',
                            attrs={
                                'generation': self._obj.generation,
                            })

        # tmp = haplo[:,0::2].data + haplo[:,1::2].data
        # ind = haplo.xft.get_variant_indexer().to_diploid()
        # xft.struct.HaplotypeArray(tmp,sample_indexer=haplo.xft.get_sample_indexer(),
        #                           variant_indexer=ind)
        raise NotImplementedError()


    def to_diploid_standardized(self, af: NDArray, scale: bool):
        """
        Standardize the HaplotypeArray object and convert it to a diploid representation.
        Specific to HaplotypeArray objects.
        
        Parameters
        ----------
        af: NDArray
            An array containing the allele frequencies of each variant.
        scale: bool
            Whether or not to scale the standardized array by the square root of the number of variants.
            
        Returns
        -------
        ndarray:
            A standardized diploid array where each variant is represented as the sum of two haplotypes.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        if scale:
            return xft.utils.standardize_array_hw(self._obj.data, af) / np.sqrt(self.m)
        else:
            return xft.utils.standardize_array_hw(self._obj.data, af)

    def get_annotation_dict(self):
        """
        Return a dictionary of all annotation variables associated with the variants in the object.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        dict:
            A dictionary where the keys are the annotation variable names and the values are the corresponding arrays.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return {x[0]: x[1].values for x in self._obj.coords.variables.items() if 'variant' in x[1].dims and x[0] not in self._non_annotation_vars}

    def get_non_annotation_dict(self):
        """
        Return a dictionary of all non-annotation variables associated with the variants in the object.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        dict:
            A dictionary where the keys are the non-annotation variable names and the values are the corresponding arrays.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return {x[0]: x[1].values for x in self._obj.coords.variables.items() if 'variant' in x[1].dims and x[0] in self._variant_vars}

    # component index properties / methods
    def grep_component_index(self, keyword: str = 'phenotype'):
        """
        Returns the index array of components whose names contain the given keyword.
        Specific to PhenotypeArray objects.

        Parameters
        ----------
        keyword : str, optional
            The keyword to search for in component names, by default 'phenotype'.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        pheno_cols = self._obj.component_name.values[self._obj.component_name.str.contains(
            keyword).values]
        component_index = self._obj.xft.get_component_indexer()[
            dict(component_name=pheno_cols)]
        return component_index


    def get_comp_type(self, ctype='intermediate'):
        """
        Returns the index array of components with comp_type==ctype
        Specific to PhenotypeArray objects.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        pheno_cols = self._obj.component_name.values[self._obj.comp_type.str.contains(
            ctype).values]
        component_index = self._obj.xft.get_component_indexer()[
            dict(component_name=pheno_cols)]
        return component_index

    def get_intermediate_components(self):
        """
        Returns the index array of components with comp_type=='intermediate'
        Specific to PhenotypeArray objects.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        return self.get_comp_type('intermediate')

    def get_outcome_components(self):
        """
        Returns the index array of components with comp_type=='outcome'
        Specific to PhenotypeArray objects.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        return self.get_comp_type('outcome')


    @property
    def k_total(self):
        """
        Returns the total number of components.
        Specific to PhenotypeArray objects.

        Returns
        -------
        int
            The total number of components.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return self.shape[1]  # number of all phenotype components

    @property
    def k_phenotypes(self):
        """
        Returns the number of unique phenotype components.
        Specific to PhenotypeArray objects.

        Returns
        -------
        int
            The number of unique phenotype components.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.phenotype_name).shape[0]

    @property
    def all_phenotypes(self) -> np.ndarray:
        """
        Returns an array of all the unique phenotype component names.
        Specific to PhenotypeArray objects.

        Returns
        -------
        numpy.ndarray
            An array of all the unique phenotype component names.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """

    @property
    def k_components(self) -> int:
        """
        Returns the number of unique component names.
        Specific to PhenotypeArray objects.

        Returns
        -------
        int
            The number of unique component names.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.component_name).shape[0]

    @property
    def all_components(self) -> np.ndarray:
        """
        Returns an array of all the unique component names.
        Specific to PhenotypeArray objects.

        Returns
        -------
        numpy.ndarray
            An array of all the unique component names.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.component_name)

    @property
    def k_relative(self) -> int:
        """
        Returns the number of unique origin relative values.
        Specific to PhenotypeArray objects.
        
        Returns
        -------
        int
            The number of unique origin relative values.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.vorigin_relative).shape[0]

    @property
    def all_relatives(self) -> np.ndarray:
        """
        Returns an array of all the unique origin relative values.
        Specific to PhenotypeArray objects.

        Returns
        -------
        numpy.ndarray
            An array of all the unique origin relative values.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.vorigin_relative)

    @property
    def k_current(self) -> int:
        """Returns the number of all current-gen specific components.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        int
            The number of all current-gen specific components.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.sum(self._obj.vorigin_relative == -1)

    def get_k_rel(self, rel) -> int:
        """Returns the number of components with the given relative origin.
        Specific to PhenotypeArray objects.
        
        Args:
            rel (int): The relative origin of the components.
        
        Raises:
            TypeError: If the column dimension is not 'component'.
            
        Returns:
            int: The number of components with the given relative origin.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.sum(self._obj.vorigin_relative == rel)

    @property
    def depth(self):
        """Returns the generational depth from binary relative encoding.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Union[float, np.nan]
            The generational depth from binary relative encoding, or NaN if the relative origin is empty.
        """
        if self._col_dim != 'component':
            raise TypeError
        if len(self.vorigin_relative) != 0:
            return math.floor(math.log2(np.max(self._obj.vorigin_relative) + 2)) + 1
        else:
            return np.NaN

    def split_by_phenotype(self) -> Dict[str, pd.DataFrame]:
        """Splits the data by phenotype name.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary of dataframes, where the keys are the unique phenotype names and the values are dataframes containing the data for each phenotype.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {phenotype: pheno.loc[:, pheno.phenotype_name == phenotype] for phenotype in self.all_phenotypes}

    def split_by_component(self):
        """Splits the data by component name.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary of dataframes, where the keys are the unique component names and the values are dataframes containing the data for each component.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {component: pheno.loc[:, pheno.component_name == component] for component in self.all_components}

    def split_by_vorigin(self) -> Dict[int, pd.DataFrame]:
        """Splits the data by relative origin.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[int, pd.DataFrame]
            A dictionary of dataframes, where the keys are the unique relative origins and the values are dataframes containing the data for each relative origin.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {vorigin: pheno.loc[:, pheno.vorigin_relative == vorigin] for vorigin in self.all_relatives}

    def split_by_phenotype_vorigin(self) -> Dict[Tuple[str, int], pd.DataFrame]:
        """Splits the data by phenotype name and relative origin.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
        If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[Tuple[str, int], pd.DataFrame]
            A dictionary of dataframes, where the keys are tuples of phenotype name and relative origin and the values are dataframes containing the data for each combination of phenotype name and relative origin.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {(phenotype, vorigin): pheno.loc[:, (pheno.phenotype_name == phenotype) ^ (pheno.vorigin_relative == vorigin)] for phenotype in self.all_phenotypes for vorigin in self.all_relatives}

    def as_pd(self, prettify: bool = True):
        """Returns the data as a Pandas DataFrame.
        Specific to PhenotypeArray objects.
        
        Parameters
        ----------
        prettify : bool, optional
            If True, the multi-index columns will be prettified by replacing -1, 0, 1 with 'proband', 'mother', 'father', respectively.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame representing the data.
        """
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
        """
        Retrieve a subset of the data with the given indices.

        Parameters
        ----------
        args : tuple
            Tuple of indices to retrieve the subset of data. It can be one of the following:
            
            - A dictionary where the keys are column names and the values are the indices of the columns to retrieve.
            - Two positional arguments representing row and column indices, respectively.

        Returns
        -------
        xr.DataArray
            The subset of data corresponding to the given indices.

        Raises
        ------
        KeyError
            If any of the indices provided are invalid.

        """
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
        """
        Set the values of a subset of the data with the given indices.

        Parameters
        ----------
        args : tuple
            Tuple of indices to set the values of the subset of data. It can be one of the following:
            
            - A dictionary where the keys are column names and the values are the indices of the columns to set the values.
            - Two positional arguments representing row and column indices, respectively.
        data : xr.DataArray
            The data to set in the specified subset of data.

        Raises
        ------
        KeyError
            If any of the indices provided are invalid.
        """
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
    """
    Represents a 2D array of binary haplotypes with accompanying row and column indices. 
    Dummy class used for generation of DataArrays and static methods 
    """
    def __new__(cls,
                # n x 2m array of binary haplotypes
                haplotypes: NDArray[Shape["*, *"], Int8] = None,
                variant_indexer: xft.index.HaploidVariantIndex = None,
                sample_indexer: xft.index.SampleIndex = None,
                generation: int = 0,
                n: int = None,
                m: int = None,
                dask: bool = False,
                **kwargs,
                ) -> xr.DataArray:
        """
        Create a new instance of DataArray.

        Parameters
        ----------
        haplotypes : np.ndarray, optional
            A 2D array of binary haplotypes. If not provided, default to None.
        variant_indexer : xft.index.HaploidVariantIndex, optional
            A haploid variant indexer. If not provided, default to None.
        sample_indexer : xft.index.SampleIndex, optional
            A sample indexer. If not provided, default to None.
        generation : int, optional
            The generation number associated with the haplotypes. Default to 0.
        n : int, optional
            The number of samples. Required if `sample_indexer` is not provided.
        m : int, optional
            The number of variants. Required if `variant_indexer` is not provided.
        dask : bool, optional
            Create a Dask array?
        **kwargs :
            Additional arguments to pass to dask.array

        Returns
        -------
        xr.DataArray
            A 2D xarray DataArray with dimensions `sample` and `variant`.

        Raises
        ------
        AssertionError
            If either `variant_indexer` or `m` is not provided, or both are provided.
            If either `sample_indexer` or `n` is not provided, or both are provided.
        
        Warns
        -----
        UserWarning
            If `haplotypes` is not provided, or if `variant_indexer` is not provided and empirical allele frequencies are used.
        """
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
        ## convert to dask array if necessary
        if dask and not isinstance(data,da.Array):
            data = da.asarray(data)
        return xr.DataArray(data=data,
                            dims=['sample', 'variant'],
                            coords=coord_dict,
                            name='HaplotypeArray',
                            attrs={
                                'generation': generation,
                            })


class PhenotypeArray:    
    """
    An array that stores phenotypes for a set of individuals.
    Dummy class used for generation of DataArrays and static methods 

    Parameters
    ----------
    components : ndarray, optional
        n x 2m array of binary haplotypes.
    component_indexer : xft.index.ComponentIndex, optional
        Indexer for components.
    sample_indexer : xft.index.SampleIndex, optional
        Indexer for samples.
    generation : int, optional
        The generation this PhenotypeArray belongs to.
    n : int, optional
        The number of samples.
    k_total : int, optional
        The total number of components.

    Returns
    -------
    xr.DataArray
        The initialized PhenotypeArray.

    Raises
    ------
    AssertionError
        If `components` is provided, then `n` and `k_total` must not be provided.
        If `component_indexer` is provided, then `k_total` must not be provided.
        If `sample_indexer` is provided, then `n` must not be provided.
        If `components` is provided and `sample_indexer` is provided, then the shape of
        `components` must match the size of the sample dimension of `sample_indexer`.
        If `components` is provided and `component_indexer` is provided, then the shape
        of `components` must match the size of the component dimension of `component_indexer`.
        If `component_indexer` is provided, then the size of the component dimension of
        `component_indexer` must match `k_total`.
    """
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
        """
        Create a PhenotypeArray from a product of names.

        Parameters
        ----------
        phenotype_name : iterable
            The names of the phenotypes.
        component_name : iterable
            The names of the components.
        vorigin_relative : iterable
            The relative origins of each component.
        components : xr.DataArray, optional
            The array to use as the components.
        sample_indexer : xft.index.SampleIndex, optional
            The sample indexer to use.
        generation : int, optional
            The generation of the PhenotypeArray.
        haplotypes : xr.DataArray, optional
            The haplotypes to use.
        n : int, optional
            The number of samples to use.

        Returns
        -------
        xr.DataArray
            The new PhenotypeArray.

        Raises
        ------
        AssertionError
            If exactly one of `generation` and `sample_indexer` is provided, or exactly one
            of `haplotypes` and `sample_indexer`/`generation` or `n`/`generation` is provided.
        """
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
