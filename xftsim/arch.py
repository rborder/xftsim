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

import xftsim as xft


from xftsim.utils import paste


class FounderInitialization:
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 initialize_component: Callable = None,
                 ):
        self.component_index = component_index
        self._initialize_component = initialize_component

    ## must take phenotypes and modify by reference
    def initialize_component(self, phenotypes):
        if self._initialize_component is None:
            self._null_initialization(phenotypes)
        else:
            self._initialize_component(phenotypes)

    def _null_initialization(self, phenotypes):
        warnings.warn("No initialization defined")
        phenotypes.loc[:,self.component_index.unique_identifier] = np.full_like(phenotypes.loc[:,self.component_index.unique_identifier].data,
                                                                                fill_value = np.NaN)
class ConstantFounderInitialization(FounderInitialization):
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 constants: Iterable = None,
                 ):
        self.component_index = component_index
        self.constants = np.array(constants)
        assert self.constants.shape[0] == component_index.k_total, "Noncomformable arguments"
        def initialize_constant(phenotypes: xr.DataArray):
            n = phenotypes.xft.n
            phenotypes.loc[:,self.component_index.unique_identifier] = np.tile(self.constants, (n,1))

        super().__init__(component_index = component_index,
                         initialize_component = initialize_constant)

class ZeroFounderInitialization(ConstantFounderInitialization):
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 ):
        super.__init__(component_index, np.zeros(component_index.k_total))

class GaussianFounderInitialization(FounderInitialization):
    """docstring for FounderInitialization"""
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 variances: Iterable = None,
                 sds: Iterable = None,
                 means: Iterable = None,
                 ):
        assert (sds is not None) ^ (variances is not None)
        if variances is not None:
            sds = np.sqrt(np.array(variances))
        else:
            sds = np.array(sds)
        if means is None:
            means=np.zeros_like(sds)
        self.sds = np.array(sds)
        self.means = np.array(means)
        self.component_index = component_index
        assert component_index.k_total == self.sds.shape[0], "scale parameter and component_index length disagree"
        
        def initialize_gaussian(phenotypes: xr.DataArray):
            n = phenotypes.xft.n
            k = self.component_index.k_total
            random_phenotypes = (np.random.randn(n*k).reshape((n,k)) * self.sds) + self.means
            # print(random_phenotypes.shape)
            # print(self.component_index)
            phenotypes.xft[None, self.component_index] = random_phenotypes

        super().__init__(component_index = component_index,
                         initialize_component = initialize_gaussian)

class ArchitectureComponent:
    def __init__(self,
                 compute_component: Callable = None,
                 input_phenotypes: xft.index.ComponentIndex = None,
                 output_phenotypes: xft.index.ComponentIndex = None,
                 input_haplotypes: Union[Bool, xft.index.HaploidVariantIndex] = False,
                 founder_initialization: Callable = None
                 ):
        self._compute_component = compute_component
        self.input_haplotypes = input_haplotypes
        self.input_phenotypes = input_phenotypes
        self.output_phenotypes = output_phenotypes
        self.founder_initialization = founder_initialization

    ## function that accesses haplotypes and/or phenotypes and modifies phenotypes by reference
    def compute_component(self,
                          haplotypes: xr.DataArray = None,
                          phenotypes: xr.DataArray = None,
                          ) -> None:
        if self._compute_component is None:
            # warnings.warn("c")
            pass
        else:
            self._compute_component(
                haplotypes,
                phenotypes,
            )

    @property
    def merged_phenotype_indexer(self):
        return xft.index.XftIndex.reduce_merge([self.input_phenotypes, 
                                                self.output_phenotypes])

    @property
    def input_phenotype_name(self):
        return np.array(self.input_phenotypes.phenotype_name)

    @property
    def input_component_name(self):
        return np.array(self.input_phenotypes.component_name)

    @property
    def input_vorigin_relative(self):
        return np.array(self.input_phenotypes.vorigin_relative)

    @property
    def output_phenotype_name(self):
        return np.array(self.output_phenotypes.phenotype_name)

    @property
    def output_component_name(self):
        return np.array(self.output_phenotypes.component_name)

    @property
    def output_vorigin_relative(self):
        return np.array(self.output_phenotypes.vorigin_relative)

    @property
    def phenotype_name(self):
        return self.merged_phenotype_indexer.phenotype_name

    @property
    def component_name(self):
        return self.merged_phenotype_indexer.component_name

    @property
    def vorigin_relative(self):
        return self.merged_phenotype_indexer.vorigin_relative

    # @property
    # def merged_phenotype_indexer(self):
    #     return xft.index.ComponentIndex(phenotype_name=self.phenotype_name,
    #                                    component_name=self.component_name,
    #                                    vorigin_relative=self.vorigin_relative)



#### Functions / classes for creating phenogenetic architectures

class PlaceholderComponent(ArchitectureComponent):
    def __init__(self,
                 components: xft.index.ComponentIndex = None,
                 metadata: Dict = dict(),
                 ):
        input_phenotypes = components
        output_phenotypes = components
        super().__init__(input_phenotypes=input_phenotypes,
                         output_phenotypes=output_phenotypes,
                         input_haplotypes=False,
                         )



# additive architecture with no parental effects
class AdditiveGeneticComponent(ArchitectureComponent):
    def __init__(self,
                 beta: xft.effect.AdditiveEffects = None,
                 metadata: Dict = dict(),
                 ):
        self.effects = beta
        input_phenotypes = xft.index.ComponentIndex_from_product([],[],[])
        output_phenotypes = xft.index.ComponentIndex_from_product(
                                                                 beta.phenotype_name,
                                                                 ['additiveGenetic'],
                                                                 [-1],
                                                                 )
        super().__init__(input_phenotypes=input_phenotypes,
                         output_phenotypes=output_phenotypes,
                         input_haplotypes=True,
                         )

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray) -> None:
        n = haplotypes.shape[0]
        heritable_components = (haplotypes.data @ self.effects.beta_raw_haploid) + np.tile(self.effects.offset, (n,1))
        phenotypes.loc[:,self.output_phenotypes.unique_identifier] = heritable_components

    @property
    def true_rho_beta(self):
        return np.corrcoef(self.effects._beta, rowvar=False)

    @property
    def true_cov_beta(self):
        return np.cov(self.effects._beta, rowvar=False)


# additive architecture with no parental effects
class AdditiveNoiseComponent(ArchitectureComponent):
    def __init__(self,
                 variances: Iterable = None,
                 sds: Iterable = None,
                 phenotype_name: Iterable = None
                 ):
        assert (variances is None) ^ (
            sds is None), "Provide only variances or sds"
        self.variances = variances
        self.sds = sds
        if variances is None:
            self.variances = np.array(sds)**2
        if sds is None:
            self.sds = np.array(variances)**.5
        input_phenotypes = xft.index.ComponentIndex_from_product([],[],[])
        output_phenotypes = xft.index.ComponentIndex_from_product(
                                                                 np.array(phenotype_name),
                                                                 ['additiveNoise'],
                                                                 [-1],
                                                                 )
        super().__init__(input_phenotypes=input_phenotypes,
                         output_phenotypes=output_phenotypes,
                         input_haplotypes=False,
                         )

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray):
        n = phenotypes.shape[0]
        noise = np.hstack([np.random.normal(0, scale, (n,1)) for scale in self.sds])
        phenotypes.loc[:,self.output_phenotypes.unique_identifier] = noise





class LinearTransformationComponent(ArchitectureComponent):
    def __init__(self,
                 input_phenotypes: xft.index.ComponentIndex = None,
                 output_phenotypes: xft.index.ComponentIndex = None,
                 coefficient_matrix: NDArray=None,
                 # transformation_array: xr.DataArray = None,
                 normalize: bool = True,
                 founder_initialization: FounderInitialization = None,
                 ):
        self.v_input_dimension = input_phenotypes.k_total
        self.v_output_dimension = output_phenotypes.k_total
        self.normalize = normalize
        # assert (coefficient_matrix is not None) ^ (transformation_array is not None), "provide coefficient_matrix with indexes XOR transformation_array"
        # if coefficient_matrix is not None:
        #     assert input_component_index is not None, "provide input_component_index"
        #     assert output_component_index is not None, "provide output_component_index"
        # else:
        #     coefficient_matrix
        if coefficient_matrix is None:
            self.coefficient_matrix = np.zeros((self.v_output_dimension, 
                                                self.v_input_dimension))
        self.coefficient_matrix = coefficient_matrix

        super().__init__(input_phenotypes=input_phenotypes,
                         output_phenotypes=output_phenotypes,
                         founder_initialization=founder_initialization,
                         )

    @property
    def linear_transformation(self):
        ## ugly code
        inputs = paste(self.input_phenotypes.coord_frame.iloc[:,1:].T.to_numpy(), sep=' ')
        if self.normalize:
            inputs = np.char.add('normalized_',inputs)
        outputs = paste(self.output_phenotypes.coord_frame.iloc[:,1:].T.to_numpy())
        return pd.DataFrame(self.coefficient_matrix,
                            columns=inputs,
                            index=outputs)

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray):
        y = phenotypes.loc[:,self.input_phenotypes.unique_identifier]
        if self.normalize:
            y = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x), 1, y)
        new_component = y @ self.coefficient_matrix
        phenotypes.loc[:,self.output_phenotypes.unique_identifier] = new_component

    def __repr__(self):
        ## ugly
        return "<"+self.__class__.__name__+">" + "\n" + self.linear_transformation.__repr__()

class VerticalComponent(LinearTransformationComponent):
    def __init__(self,
                 input_component_index: xft.index.ComponentIndex = None,
                 output_component_index: xft.index.ComponentIndex = None,
                 coefficient_matrix: NDArray=None,
                 normalize: bool = True,
                 founder_variances: Iterable = None,
                 founder_initialization: FounderInitialization = None,
                 ):
        assert (founder_variances is None) ^ (founder_initialization is None), "provide founder_initialization XOR founder_variances"
        if founder_initialization is None:
            founder_initialization = GaussianFounderInitialization(input_component_index,
                                                                   variances=founder_variances)
        super().__init__(input_phenotypes=input_component_index,
                         output_phenotypes=output_component_index,
                         founder_initialization=founder_initialization,
                         coefficient_matrix = coefficient_matrix,
                         normalize= normalize,
                         )

# class MaternalVerticalComponent(LinearTransformationComponent):
#     def __init__(self,
#                  phenotype_name: Iterable,
#                  component_name: Iterable,
#                  output_phenotypes: Iterable,
#                  coefficient_matrix: NDArray=None,
#                  output_component = 'maternalVertical',
#                  normalize: bool = True,
#                  ):
#         super().__init__(phenotype_name=phenotype_name,
#                        component_name=component_name,
#                        vorigin_relative=0,
#                        output_phenotypes=output_phenotypes,
#                        coefficient_matrix=coefficient_matrix,
#                        output_component = output_component,
#                        normalize=normalize
#                        )

# class PaternalVerticalComponent(LinearTransformationComponent):
#     def __init__(self,
#                  phenotype_name: Iterable,
#                  component_name: Iterable,
#                  output_phenotypes: Iterable,
#                  coefficient_matrix: NDArray=None,
#                  output_component = 'paternalVertical',
#                  normalize: bool = True,
#                  ):
#         super().__init__(phenotype_name=phenotype_name,
#                        component_name=component_name,
#                        vorigin_relative=1,
#                        output_phenotypes=output_phenotypes,
#                        coefficient_matrix=coefficient_matrix,
#                        output_component = output_component,
#                        normalize=normalize
#                        )

# class MeanVerticalComponent(LinearTransformationComponent):
#     def __init__(self,
#                  phenotype_name: Iterable,
#                  component_name: Iterable,
#                  output_phenotypes: Iterable,
#                  coefficient_matrix: NDArray=None,
#                  output_component = 'MeanVertical',
#                  normalize: bool = True,
#                  ):
#         vorigin_relative = np.tile([0,1], len(component_name))
#         phenotype_name = np.repeat(phenotype_name, 2)
#         component_name = np.repeat(component_name, 2)
#         print(vorigin_relative)

#         super().__init__(phenotype_name=phenotype_name,
#                        component_name=component_name,
#                        vorigin_relative=vorigin_relative,
#                        output_phenotypes=output_phenotypes,
#                        coefficient_matrix=np.hstack([coefficient_matrix,coefficient_matrix]),
#                        output_component = output_component,
#                        normalize=normalize,
#                        multiplier=.5
#                        )

class HorizontalComponent(LinearTransformationComponent):
    def __init__(self,
                 phenotype_name: Iterable,
                 component_name: Iterable,
                 coefficient_matrix: NDArray=None,
                 normalize: bool = True,
                 output_component = 'Horizontal',
                 ):
        vorigin_relative = -1

        super().__init__(phenotype_name=phenotype_name,
                       component_name=component_name,
                       vorigin_relative=vorigin_relative,
                       output_phenotypes=phenotype_name,
                       coefficient_matrix=coefficient_matrix,
                       output_component = output_component,
                       normalize=normalize,
                       multiplier=1,
                       )


class SumComponent(ArchitectureComponent):
    def __init__(self,
                 phenotype_name: Iterable,
                 sum_components: Iterable = ['additiveGenetic','additiveNoise'],
                 vorigin_relative: Iterable = [-1],
                 output_component: str = 'phenotype',
                 ):
        input_frame = xft.index.ComponentIndex_from_product(phenotype_name, 
                                                           sum_components,
                                                           vorigin_relative).coord_frame
        output_frame = input_frame.copy().loc[~input_frame[['phenotype_name','vorigin_relative']].duplicated()]
        output_frame['component_name'] = output_component
        
        self.input_haplotypes = False
        self.input_phenotypes = xft.index.ComponentIndex_from_frame(input_frame)
        self.output_phenotypes = xft.index.ComponentIndex_from_frame(output_frame)
        self._vorigin_relative = vorigin_relative
        self._phenotype_name = phenotype_name
        self.founder_initialization = None
        ## need to finish
    def compute_component(self,
        haplotypes: xr.DataArray,
        phenotypes: xr.DataArray):
        ## TODO make faster later, UGLY atm
        inputs = self.input_phenotypes.coord_frame
        outputs = self.output_phenotypes.coord_frame
        outputs.set_index('phenotype_name', inplace=True, drop=False)
        ## iterate over vorigin 
        for vo in np.unique(self._vorigin_relative):
            input_index = inputs.loc[inputs.vorigin_relative.values==vo,:]
            output_index = outputs.loc[outputs.vorigin_relative.values==vo,:]
            new_data = phenotypes.loc[:,inputs.component.values].groupby('phenotype_name').sum(skipna=False)
            assignment_indicies = output_index.loc[new_data.phenotype_name.values,:].component.values
            phenotypes.loc[:,assignment_indicies] = new_data.values



class BinarizingComponent(ArchitectureComponent):
    def __init__(self,
                 thresholds: Iterable,
                 phenotype_name: Iterable,
                 liability_component: str = 'phenotype',
                 vorigin_relative: Iterable = [-1], ## TODO: make consistent with providing index
                 output_component: str = 'binary_phenotype',
                 ):
        # assert len(thresholds) == len(phenotype_name)
        self.thresholds=np.array(thresholds).ravel()
        # if not isinstance(thresholds, dict):
            # thresholds = {pheno:threshold for (pheno,threshold) in zip (phenotype_name,thresholds)}
        input_frame = xft.index.ComponentIndex_from_product(phenotype_name, 
                                                            [liability_component],
                                                            vorigin_relative).coord_frame
        output_frame = input_frame.copy().loc[~input_frame[['phenotype_name','vorigin_relative']].duplicated()]
        output_frame['component_name'] = output_component
        
        self.input_haplotypes = False
        self.input_phenotypes = xft.index.ComponentIndex_from_frame(input_frame)
        self.output_phenotypes = xft.index.ComponentIndex_from_frame(output_frame)
        self._vorigin_relative = vorigin_relative
        self._phenotype_name = phenotype_name
        self.founder_initialization = None
        ## need to finish
    def compute_component(self,
        haplotypes: xr.DataArray,
        phenotypes: xr.DataArray):
        ## TODO make faster later, UGLY atm
        y = phenotypes.loc[:,self.input_phenotypes.unique_identifier].data
        new_component = (y>self.thresholds).astype(int) 
        phenotypes.loc[:,self.output_phenotypes.unique_identifier] = new_component



# class PhenotypicTransformationComponent(ArchitectureComponent):
#     def __init__(self,
#                  compute_component: Callable = None,
#                  phenotypes: xft.index.ComponentIndex = None,
#                  ):
#         self._compute_component = compute_component
#         self.input_haplotypes = False
#         self.input_phenotypes = phenotypes
#         self.output_phenotypes = phenotypes




##  Architecture
## 
## an Architecture object consists of four pieces:
## 
##  - components: an iterable collection of ArchitectureComponent objects
##  - initialize_next_generation: function taking current phenotypes, mating
##    assignment, haplotypes, returns new empty Phenotype structure
##  - initialize_founder_generation (optional): function taking 
##    haplotypes, returns new empty Phenotype structure
##  - metadata (optional): dict
class Architecture:
    def __init__(self,
        components: Iterable = None,
        metadata: Dict = dict(),
        depth: int = 1,
        expand_components: bool = False,
        ):
        self.metadata = metadata
        self.components = components
        self.depth = depth
        self.expand_components = expand_components

    @property
    def founder_initializations(self):
        return [x.founder_initialization for x in self.components if x.founder_initialization is not None]

    @property
    def merged_component_indexer(self):
        merged = xft.index.XftIndex.reduce_merge([x.merged_phenotype_indexer for x in self.components])
        if not self.expand_components:
            return merged
        elif self.expand_components:
            phenotype_name = np.unique(merged.phenotype_name)
            component_name = np.unique(merged.component_name)
            vorigin_relative = np.unique(merged.vorigin_relative)
            return xft.index.ComponentIndex_from_product(phenotype_name,
                                                         component_name,
                                                         vorigin_relative)


    def initialize_phenotype_array(self, 
                                   haplotypes: xr.DataArray,
                                   control: dict = None,
                                   ) -> xr.DataArray:
        sample_indexer = haplotypes.xft.get_sample_indexer()
        return xft.struct.phenotypeArray(component_indexer = self.merged_component_indexer,
                                        sample_indexer = haplotypes.xft.get_sample_indexer(),
                                        generation = haplotypes.attrs['generation'])

    def initialize_founder_phenotype_array(self, 
                                           haplotypes: xr.DataArray,
                                           control: dict = None,
                                           ) -> xr.DataArray:
        phenotype_array = self.initialize_phenotype_array(haplotypes)
        for initialization in self.founder_initializations:
            initialization.initialize_component(phenotype_array)
        return phenotype_array

                                           

    # def initialize_next_generation(self, 
    #                                haplotypes: xr.DataArray = None,
    #                                mating: xft.mate.MateAssignment = None,
    #                                phenotypeHistory: Dict = None,
    #                                ) -> xr.DataArray:
    #     assert haplotypes.xft._is_haplotypeArray()
    # #     if self._initialize_next_generation is None:
    # #         raise NotImplementedError
    # #     else:
    # #         self._initialize_next_generation(haplotypes, mating, phenotypeHistory)

    # def initialize_founder_generation(self, 
    #                                   haplotypes: xr.DataArray = None,
    #                                   ) -> xr.DataArray:
    #     assert haplotypes.xft._is_haplotypeArray()
    #     if self._initialize_founder_generation is None:
    #         self._initialize_next_generation(haplotypes, mating=None, phenotypes_previous=None)
    #     else:
    #         self._initialize_founder_generation(haplotypes)


    def compute_phenotypes(self,
                           haplotypes: xr.DataArray = None,
                           phenotypes: xr.DataArray = None,
                           control: dict = None,
                           ) -> None:
        if self.components is None:
            raise NotImplementedError
        for component in self.components:
            component.compute_component(haplotypes, phenotypes)


class InfinitessimalArchitecture:
    def __init__(self):
        NotImplementedError ## TODO


class SpikeSlabArchitecture:
    def __init__(self):
        NotImplementedError ## TODO

