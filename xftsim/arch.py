import warnings
import functools
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final, Tuple
from numpy.typing import ArrayLike
from functools import cached_property
import dask.array as da

import xftsim as xft


from xftsim.utils import paste


class FounderInitialization:
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 initialize_component: Callable = None,
                 ):
        self.component_index = component_index
        self._initialize_component = initialize_component

    # must take phenotypes and modify by reference
    def initialize_component(self, phenotypes):
        if self._initialize_component is None:
            self._null_initialization(phenotypes)
        else:
            self._initialize_component(phenotypes)

    def _null_initialization(self, phenotypes):
        warnings.warn("No initialization defined")
        phenotypes.loc[:, self.component_index.unique_identifier] = np.full_like(phenotypes.loc[:, self.component_index.unique_identifier].data,
                                                                                 fill_value=np.NaN)

class FounderInitialization:
    """Base class for founder initialization."""
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 initialize_component: Callable = None,
                 ):
        """
        Parameters
        ----------
        component_index: xft.index.ComponentIndex, optional
            Component index specifying the phenotype components.
        initialize_component: Callable, optional
            Function to initialize the founder haplotypes for each phenotype.
            Must take phenotypes as input and modify them by reference.
        """
        self.component_index = component_index
        self._initialize_component = initialize_component

    def initialize_component(self, phenotypes: xr.DataArray):
        """
        Initialize founder haplotypes for a single phenotype component.

        Parameters
        ----------
        phenotypes: xr.DataArray
            Phenotypes for a single phenotype component.

        Raises
        ------
        Warning
            If no initialization method is defined.
        """
        if self._initialize_component is None:
            self._null_initialization(phenotypes)
        else:
            self._initialize_component(phenotypes)

    def _null_initialization(self, phenotypes: xr.DataArray):
        """
        A null initialization method that does nothing except warn the user.

        Parameters
        ----------
        phenotypes: xr.DataArray
            Phenotypes for a single phenotype component.

        Raises
        ------
        Warning
            If no initialization method is defined.
        """
        warnings.warn("No initialization defined")
        phenotypes.loc[:, self.component_index.unique_identifier] = np.full_like(
            phenotypes.loc[:, self.component_index.unique_identifier].data, fill_value=np.NaN)


class ConstantFounderInitialization(FounderInitialization):
    """Founder initialization that sets all haplotypes to constant values."""
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 constants: Iterable = None,
                 ):
        """
        Parameters
        ----------
        component_index: xft.index.ComponentIndex, optional
            Component index specifying the phenotype components.
        constants: Iterable, optional
            Iterable of constant values to set the haplotypes to.
        """
        self.component_index = component_index
        self.constants = np.array(constants)
        assert self.constants.shape[0] == component_index.k_total, "Noncomformable arguments"

        def initialize_constant(phenotypes: xr.DataArray):
            n = phenotypes.xft.n
            phenotypes.loc[:, self.component_index.unique_identifier] = np.tile(
                self.constants, (n, 1))

        super().__init__(component_index=component_index,
                         initialize_component=initialize_constant)


class ZeroFounderInitialization(ConstantFounderInitialization):
    """Founder initialization that sets all haplotypes to zero."""
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 ):
        """
        Parameters
        ----------
        component_index: xft.index.ComponentIndex, optional
            Component index specifying the phenotype components.
        """
        super().__init__(component_index, np.zeros(component_index.k_total))


class GaussianFounderInitialization(FounderInitialization):
    """
    A class for initializing founder haplotypes by drawing iid samples from
    normal distributions with the specified means and standard deviations.

    Parameters
    ----------
    component_index: xft.index.ComponentIndex, optional
        A ComponentIndex object containing the indexing information
        of the components. If not provided, then the initialization
        will be null.
    variances: Iterable, optional
        An iterable object of length k_total specifying the variances
        of the Gaussian distribution. Either variances or sds must be
        provided.
    sds: Iterable, optional
        An iterable object of length k_total specifying the standard
        deviations of the Gaussian distribution. Either variances or
        sds must be provided.
    means: Iterable, optional
        An iterable object of length k_total specifying the means of the
        Gaussian distribution. If not provided, then the means will be
        set to 0.

    Raises
    ------
    AssertionError
        If neither variances nor sds is provided or if the length of
        component_index does not match the length of sds.

    Attributes
    ----------
    sds: numpy.ndarray
        An array of standard deviations.
    means: numpy.ndarray
        An array of means.
    component_index: xft.index.ComponentIndex
        An object containing the indexing information of the components.
    """
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
            means = np.zeros_like(sds)
        self.sds = np.array(sds)
        self.means = np.array(means)
        self.component_index = component_index
        assert component_index.k_total == self.sds.shape[
            0], "scale parameter and component_index length disagree"

        def initialize_gaussian(phenotypes: xr.DataArray):
            n = phenotypes.xft.n
            k = self.component_index.k_total
            random_phenotypes = (np.random.randn(
                n * k).reshape((n, k)) * self.sds) + self.means
            # print(random_phenotypes.shape)
            # print(self.component_index)
            phenotypes.xft[None, self.component_index] = random_phenotypes

        super().__init__(component_index=component_index,
                         initialize_component=initialize_gaussian)


class ArchitectureComponent:
    """
    Class representing a component of a genetic architecture.

    Parameters
    ----------
    compute_component : Callable, optional
        Function that accesses haplotypes and/or phenotypes and modifies phenotypes by reference, by default None.
    input_cindex : xft.index.ComponentIndex, optional
        Index of the input component, by default None.
    output_cindex : xft.index.ComponentIndex, optional
        Index of the output component, by default None.
    input_haplotypes : bool or xft.index.HaploidVariantIndex, optional
        Boolean or HaploidVariantIndex indicating if input haplotypes are used, by default False.
    founder_initialization : Callable, optional
        Function that initializes founder haplotypes for the component, by default None.

    Attributes
    ----------
    _compute_component : Callable or None
        Function that accesses haplotypes and/or phenotypes and modifies phenotypes by reference.
    input_haplotypes : bool or xft.index.HaploidVariantIndex
        Boolean or HaploidVariantIndex indicating if input haplotypes are used.
    input_cindex : xft.index.ComponentIndex
        Index of the input component.
    output_cindex : xft.index.ComponentIndex
        Index of the output component.
    founder_initialization : Callable or None
        Function that initializes founder haplotypes for the component.
    """
    def __init__(self,
                 compute_component: Callable = None,
                 input_cindex: xft.index.ComponentIndex = None,
                 output_cindex: xft.index.ComponentIndex = None,
                 input_haplotypes: Union[Bool,
                                         xft.index.HaploidVariantIndex] = False,
                 founder_initialization: Callable = None
                 ):
        self._compute_component = compute_component
        self.input_haplotypes = input_haplotypes
        self.input_cindex = input_cindex
        self.output_cindex = output_cindex
        self.founder_initialization = founder_initialization

    @staticmethod
    def default_input_cindex(*args, **kwargs):
        """
        Static method to define the default input component index.
        """
        pass

    @staticmethod
    def default_output_cindex(*args, **kwargs):
        """
        Static method to define the default output component index.
        """
        pass

    # function that accesses haplotypes and/or phenotypes and modifies phenotypes by reference
    def compute_component(self,
                          haplotypes: xr.DataArray = None,
                          phenotypes: xr.DataArray = None,
                          ) -> None:
        """
        Function that accesses haplotypes and/or phenotypes and modifies phenotypes by reference.

        Parameters
        ----------
        haplotypes : xr.DataArray, optional
            Haplotypes to be accessed, by default None.
        phenotypes : xr.DataArray, optional
            Phenotypes to be accessed and modified, by default None.
        """
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
        return xft.index.XftIndex.reduce_merge([self.input_cindex,
                                                self.output_cindex])

    @property
    def input_phenotype_name(self):
        return np.array(self.input_cindex.phenotype_name)

    @property
    def input_component_name(self):
        return np.array(self.input_cindex.component_name)

    @property
    def input_vorigin_relative(self):
        return np.array(self.input_cindex.vorigin_relative)

    @property
    def output_phenotype_name(self):
        return np.array(self.output_cindex.phenotype_name)

    @property
    def output_component_name(self):
        return np.array(self.output_cindex.component_name)

    @property
    def output_vorigin_relative(self):
        return np.array(self.output_cindex.vorigin_relative)

    @property
    def phenotype_name(self):
        return self.merged_phenotype_indexer.phenotype_name

    @property
    def component_name(self):
        return self.merged_phenotype_indexer.component_name

    @property
    def vorigin_relative(self):
        return self.merged_phenotype_indexer.vorigin_relative


# Functions / classes for creating phenogenetic architectures
class PlaceholderComponent(ArchitectureComponent):
    def __init__(self,
                 components: xft.index.ComponentIndex = None,
                 metadata: Dict = dict(),
                 ):
        input_cindex = components
        output_cindex = components
        super().__init__(input_cindex=input_cindex,
                         output_cindex=output_cindex,
                         input_haplotypes=False,
                         )

class AdditiveGeneticComponent(ArchitectureComponent):
    """
    A genetic component with additive effects.

    Parameters
    ----------
    beta : xft.effect.AdditiveEffects, optional
        Additive effects, by default None.
    metadata : Dict, optional
        Additional metadata, by default an empty dictionary.

    Attributes
    ----------
    effects : xft.effect.AdditiveEffects
        Additive effects.
    """

    def __init__(self,
                 beta: xft.effect.AdditiveEffects = None,
                 metadata: Dict = dict(),
                 ) -> None:
        """
        Initialize a new AdditiveGeneticComponent object.

        Parameters
        ----------
        beta : xft.effect.AdditiveEffects, optional
            Additive effects, by default None.
        metadata : Dict, optional
            Additional metadata, by default an empty dictionary.
        """
        self.effects = beta
        input_cindex = xft.index.ComponentIndex.from_product([], [], [])
        output_cindex = beta.component_indexer
        super().__init__(input_cindex=input_cindex,
                         output_cindex=output_cindex,
                         input_haplotypes=True,
                         )

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray) -> None:
        """
        Compute the additive genetic component of the phenotype.

        Parameters
        ----------
        haplotypes : xr.DataArray
            Haplotypes to be used in the computation.
        phenotypes : xr.DataArray
            Phenotypes to be modified.
        """
        n = haplotypes.shape[0]
        heritable_components = da.dot(haplotypes.data, self.effects.beta_raw_haploid) + np.tile(self.effects.offset, (n, 1))
        phenotypes.loc[:,
                       self.output_cindex.unique_identifier] = heritable_components

    @property
    def true_rho_beta(self):
        """
        Compute the correlation coefficient matrix of the additive effects.

        Returns
        -------
        ndarray
            Correlation coefficient matrix of the additive effects.
        """
        return np.corrcoef(self.effects._beta, rowvar=False)

    @property
    def true_cov_beta(self):
        """
        Compute the covariance matrix of the additive effects.

        Returns
        -------
        ndarray
            Covariance matrix of the additive effects.
        """
        return np.cov(self.effects._beta, rowvar=False)


class AdditiveNoiseComponent(ArchitectureComponent):
    """
    An independent Gaussian noise component.

    Parameters
    ----------
    variances : Iterable, optional
        Variances of the noise components, by default None.
    sds : Iterable, optional
        Standard deviations of the noise components, by default None.
    phenotype_name : Iterable, optional
        Names of the phenotypes, by default None.

    Attributes
    ----------
    variances : ndarray
        Variances of the noise components.
    sds : ndarray
        Standard deviations of the noise components.
    """

    def __init__(self,
                 variances: Iterable = None,
                 sds: Iterable = None,
                 phenotype_name: Iterable = None
                 ) -> None:
        """
        Initialize a new AdditiveNoiseComponent object.

        Parameters
        ----------
        variances : Iterable, optional
            Variances of the noise components, by default None.
        sds : Iterable, optional
            Standard deviations of the noise components, by default None.
        phenotype_name : Iterable, optional
            Names of the phenotypes, by default None.
        """
        assert (variances is None) ^ (
            sds is None), "Provide only variances or sds"
        self.variances = variances
        self.sds = sds
        if variances is None:
            self.variances = np.array(sds)**2
        if sds is None:
            self.sds = np.array(variances)**.5
        input_cindex = xft.index.ComponentIndex.from_product([], [], [])
        output_cindex = xft.index.ComponentIndex.from_product(
            np.array(phenotype_name),
            ['additiveNoise'],
            [-1],
        )
        super().__init__(input_cindex=input_cindex,
                         output_cindex=output_cindex,
                         input_haplotypes=False,
                         )

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray) -> None:
        """
        Compute the noise component of the phenotype.

        Parameters
        ----------
        haplotypes : xr.DataArray
            Haplotypes, not used in the computation.
        phenotypes : xr.DataArray
            Phenotypes to be modified.
        """
        n = phenotypes.shape[0]
        noise = np.hstack([np.random.normal(0, scale, (n, 1))
                           for scale in self.sds])
        phenotypes.loc[:, self.output_cindex.unique_identifier] = noise


class LinearTransformationComponent(ArchitectureComponent):
    """
    A linear transformation component. Maps input phenotypes to output phenotypes using linear
    map represented by `coefficient_matrix`.

    Parameters
    ----------
    input_cindex : xft.index.ComponentIndex, optional
        Input component index, by default None.
    output_cindex : xft.index.ComponentIndex, optional
        Output component index, by default None.
    coefficient_matrix : ndarray, optional
        Coefficient matrix, by default None.
    normalize : bool, optional
        If True, normalize the input by subtracting the mean and dividing by the standard deviation,
        by default True.
    founder_initialization : FounderInitialization, optional
        Founder initialization, by default None.

    Attributes
    ----------
    v_input_dimension : int
        Input dimension.
    v_output_dimension : int
        Output dimension.
    normalize : bool
        If True, normalize the input by subtracting the mean and dividing by the standard deviation.
    coefficient_matrix : ndarray
        Coefficient matrix.
    """
    def __init__(self,
                 input_cindex: xft.index.ComponentIndex = None,
                 output_cindex: xft.index.ComponentIndex = None,
                 coefficient_matrix: NDArray=None,
                 normalize: bool = True,
                 founder_initialization: FounderInitialization = None,
                 ):
        self.v_input_dimension = input_cindex.k_total
        self.v_output_dimension = output_cindex.k_total
        self.normalize = normalize

        if coefficient_matrix is None:
            self.coefficient_matrix = np.zeros((self.v_output_dimension,
                                                self.v_input_dimension))
        self.coefficient_matrix = coefficient_matrix

        super().__init__(input_cindex=input_cindex,
                         output_cindex=output_cindex,
                         founder_initialization=founder_initialization,
                         )

    @property
    def linear_transformation(self) -> pd.DataFrame:
        """
        Get the linear transformation matrix.

        Returns
        -------
        pd.DataFrame
            Linear transformation matrix.
        """
        # ugly code
        inputs = paste(
            self.input_cindex.coord_frame.iloc[:, 1:].T.to_numpy(), sep=' ')
        if self.normalize:
            inputs = np.char.add('normalized_', inputs)
        outputs = paste(
            self.output_cindex.coord_frame.iloc[:, 1:].T.to_numpy())
        return pd.DataFrame(self.coefficient_matrix,
                            columns=inputs,
                            index=outputs)

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray) -> None:
        """
        Compute the linear transformation component of the phenotype.

        Parameters
        ----------
        haplotypes : xr.DataArray
            Haplotypes, not used in the computation.
        phenotypes : xr.DataArray
            Phenotypes to be modified.
        """
        y = phenotypes.loc[:, self.input_cindex.unique_identifier]
        if self.normalize:
            y = np.apply_along_axis(lambda x: (
                x - np.mean(x)) / np.std(x), 1, y)
        new_component = y @ self.coefficient_matrix
        phenotypes.loc[:, self.output_cindex.unique_identifier] = new_component

    def __repr__(self):
        # ugly
        return "<" + self.__class__.__name__ + ">" + "\n" + self.linear_transformation.__repr__()


class VerticalComponent(LinearTransformationComponent):
    """
    A vertical transmission component. Requires a way to generate "transmitted" components 
    in the founder generation. 

    Parameters
    ----------
    input_cindex : xft.index.ComponentIndex, optional
        Input component index, by default None.
    output_cindex : xft.index.ComponentIndex, optional
        Output component index, by default None.
    coefficient_matrix : ndarray, optional
        Coefficient matrix, by default None.
    normalize : bool, optional
        If True, normalize the input by subtracting the mean and dividing by the standard deviation,
        by default True.
    founder_variances : Iterable, optional
        Variances of the founders, by default None.
    founder_initialization : FounderInitialization, optional
        Founder initialization, by default None.

    Attributes
    ----------
    v_input_dimension : int
        Input dimension.
    v_output_dimension : int
        Output dimension.
    normalize : bool
        If True, normalize the input by subtracting the mean and dividing by the standard deviation.
    coefficient_matrix : ndarray
        Coefficient matrix.
    """

    def __init__(self,
                 input_cindex: xft.index.ComponentIndex = None,
                 output_cindex: xft.index.ComponentIndex = None,
                 coefficient_matrix: NDArray=None,
                 normalize: bool = True,
                 founder_variances: Iterable = None,
                 founder_initialization: FounderInitialization = None,
                 ):
        assert (founder_variances is None) ^ (
            founder_initialization is None), "provide founder_initialization XOR founder_variances"
        if founder_initialization is None:
            founder_initialization = GaussianFounderInitialization(input_cindex,
                                                                   variances=founder_variances)
        super().__init__(input_cindex=input_cindex,
                         output_cindex=output_cindex,
                         founder_initialization=founder_initialization,
                         coefficient_matrix=coefficient_matrix,
                         normalize=normalize,
                         )


class HorizontalComponent(LinearTransformationComponent):
    def __init__(self,
                 input_cindex: xft.index.ComponentIndex,
                 output_cindex: xft.index.ComponentIndex,
                 coefficient_matrix: NDArray=None,
                 normalize: bool = True,
                 ):

        super().__init__(input_cindex=input_cindex,
                         output_cindex=output_cindex,
                         coefficient_matrix=coefficient_matrix,
                         normalize=normalize,
                         )


class SumTransformation(ArchitectureComponent):
    """
    Sum components to generate phenotypes.

    Parameters
    ----------
    input_cindex : xft.index.ComponentIndex
        Input component index.
    output_cindex : xft.index.ComponentIndex
        Output component index.

    Attributes
    ----------
    input_haplotypes : bool
        If True, haplotypes are input.
    input_cindex : xft.index.ComponentIndex
        Input component index.
    output_cindex : xft.index.ComponentIndex
        Output component index.
    founder_initialization : None
        Founder initialization.
    """
    def __init__(self,
                 input_cindex: xft.index.ComponentIndex,
                 output_cindex: xft.index.ComponentIndex,
                 ):
        """
        Initialize a new SumTransformation object.

        Parameters
        ----------
        input_cindex : xft.index.ComponentIndex
            Input component index.
        output_cindex : xft.index.ComponentIndex
            Output component index.
        """
        self.input_haplotypes = False
        self.input_cindex = input_cindex
        self.output_cindex = output_cindex
        self.founder_initialization = None

    @staticmethod
    def construct_input_cindex(phenotype_name: Iterable,
                               sum_components: Iterable = [
                                   'additiveGenetic', 'additiveNoise'],
                               vorigin_relative: Iterable = [-1],
                               ) -> xft.index.ComponentIndex:
        """
        Construct input component index.

        Parameters
        ----------
        phenotype_name : Iterable
            Phenotype name.
        sum_components : Iterable, optional
            Components to sum, by default ['additiveGenetic', 'additiveNoise'].
        vorigin_relative : Iterable, optional
            Relative vorigins, by default [-1].

        Returns
        -------
        xft.index.ComponentIndex
            Component index.
        """
        return xft.index.ComponentIndex.from_product(phenotype_name,
                                                     sum_components,
                                                     vorigin_relative)

    @staticmethod
    def construct_output_cindex(phenotype_name: Iterable,
                                sum_components: Iterable = [
                                    'additiveGenetic', 'additiveNoise'],
                                vorigin_relative: Iterable = [-1],
                                output_name='phenotype') -> xft.index.ComponentIndex:
        """
        Construct output component index.

        Parameters
        ----------
        phenotype_name : Iterable
            Phenotype name.
        sum_components : Iterable, optional
            Components to sum, by default ['additiveGenetic', 'additiveNoise'].
        vorigin_relative : Iterable, optional
            Relative vorigins, by default [-1].
        output_name : str, optional
            Output name, by default 'phenotype'.

        Returns
        -------
        xft.index.ComponentIndex
            Component index.
        """
        input_frame = SumTransformation.construct_input_cindex(phenotype_name,
                                                               sum_components,
                                                               vorigin_relative).coord_frame
        output_frame = input_frame.copy(
        ).loc[~input_frame[['phenotype_name', 'vorigin_relative']].duplicated()]
        output_frame['component_name'] = output_name
        return xft.index.ComponentIndex.from_frame(output_frame)

    @staticmethod
    def construct_cindexes(phenotype_name: Iterable,
                           sum_components: Iterable = [
                               'additiveGenetic', 'additiveNoise'],
                           vorigin_relative: Iterable = [-1],
                           output_component: str = 'phenotype',
                           ) -> Tuple[xft.index.ComponentIndex, xft.index.ComponentIndex]:
        """
        Construct input and output ComponentIndex objects for SumTransformation.

        Parameters:
        -----------
        phenotype_name : Iterable
            Names of the phenotypes.
        sum_components : Iterable, optional (default=["additiveGenetic", "additiveNoise"])
            Names of the components to be summed.
        vorigin_relative : Iterable, optional (default=[-1])
            Relative origin of the component with respect to the phenotype.
        output_component : str, optional (default="phenotype")
            Name of the output component.

        Returns:
        --------
        Tuple[xft.index.ComponentIndex, xft.index.ComponentIndex]:
            A tuple containing input and output ComponentIndex objects.
        """
        input_cindex = SumTransformation.construct_input_cindex(phenotype_name,
                                                                sum_components,
                                                                vorigin_relative)
        output_cindex = SumTransformation.construct_output_cindex(phenotype_name,
                                                                  sum_components,
                                                                  vorigin_relative,
                                                                  output_component)
        return (input_cindex, output_cindex)

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray) -> None:
        """
        Compute the sum of the input components and assign them to the output component.

        Parameters:
        -----------
        haplotypes : xr.DataArray
            Haplotypes.
        phenotypes : xr.DataArray
            Phenotypes.

        Returns:
        --------
        None
        """
        # TODO make faster later, UGLY atm
        inputs = self.input_cindex.coord_frame
        outputs = self.output_cindex.coord_frame
        outputs.set_index('phenotype_name', inplace=True, drop=False)
        # iterate over vorigin
        for vo in np.unique(self.input_cindex.vorigin_relative.values):
            input_index = inputs.loc[inputs.vorigin_relative.values == vo, :]
            output_index = outputs.loc[outputs.vorigin_relative.values == vo, :]
            new_data = phenotypes.loc[:, inputs.component.values].groupby(
                'phenotype_name').sum(skipna=False)
            assignment_indicies = output_index.loc[new_data.phenotype_name.values,
                                                   :].component.values
            phenotypes.loc[:, assignment_indicies] = new_data.values


class BinarizingTransformation(ArchitectureComponent):
    """An architecture component that binarizes specified phenotypes based on specified thresholds
    under a liability-threshold model.

    Attributes:
    ----------
    thresholds : Iterable
        A list or array of thresholds used for binarization.
    input_cindex : xft.index.ComponentIndex
        The input component index.
    output_cindex : xft.index.ComponentIndex
        The output component index.
    phenotype_name : Iterable
        The name of the phenotype.
    liability_component : str
        The liability component to be used. Default is 'phenotype'.
    vorigin_relative : Iterable
        The relative V origin. Default is [-1].
    output_component : str
        The name of the output component. Default is 'binary_phenotype'.

    Methods:
    -------
    construct_input_cindex(phenotype_name: Iterable,
                           liability_component: str = 'phenotype',
                           vorigin_relative: Iterable = [-1],) -> xft.index.ComponentIndex
        Constructs the input component index based on given phenotype names.
    construct_output_cindex(phenotype_name: Iterable,
                            output_component: str = 'binary_phenotype',
                            vorigin_relative: Iterable = [-1],) -> xft.index.ComponentIndex
        Constructs the output component index based on given phenotype names.
    construct_cindexes(phenotype_name: Iterable,
                       liability_component: str = 'phenotype',
                       output_component: str = 'binary_phenotype',
                       vorigin_relative: Iterable = [-1],) -> Tuple[xft.index.ComponentIndex, xft.index.ComponentIndex]
        Constructs both the input and output component indexes based on given phenotype names.
    compute_component(self,
                      haplotypes: xr.DataArray,
                      phenotypes: xr.DataArray) -> None:
        Computes the binary phenotype based on the given thresholds.

    """
    def __init__(self,
                 thresholds: Iterable,
                 input_cindex: xft.index.ComponentIndex,
                 output_cindex: xft.index.ComponentIndex,
                 phenotype_name: Iterable,
                 liability_component: str = 'phenotype',
                 # TODO: make consistent with providing index
                 vorigin_relative: Iterable = [-1],
                 output_component: str = 'binary_phenotype',
                 ):
        # assert len(thresholds) == len(phenotype_name)
        self.thresholds = np.array(thresholds).ravel()
        # if not isinstance(thresholds, dict):
        # thresholds = {pheno:threshold for (pheno,threshold) in zip (phenotype_name,thresholds)}

        self.input_haplotypes = False
        self.input_cindex = input_cindex
        self.output_cindex = output_cindex
        self.founder_initialization = None

    @staticmethod
    def construct_input_cindex(phenotype_name: Iterable,
                               liability_component: str = 'phenotype',
                               vorigin_relative: Iterable = [-1],):
        """
        Constructs the input component index for the binarizing transformation.

        Parameters
        ----------
        phenotype_name : Iterable
            Names of the phenotypes.
        liability_component : str, optional
            Name of the liability component. Default is "phenotype".
        vorigin_relative : Iterable, optional
            v-origin relative. Default is [-1].

        Returns
        -------
        xft.index.ComponentIndex
            The input component index.
        """
        return xft.index.ComponentIndex.from_product(phenotype_name,
                                                     [liability_component],
                                                     vorigin_relative)

    @staticmethod
    def construct_output_cindex(phenotype_name: Iterable,
                                output_component: str = 'binary_phenotype',
                                vorigin_relative: Iterable = [-1],
                                ):
        """
        Constructs the output component index for the binarizing transformation.

        Parameters
        ----------
        phenotype_name : Iterable
            Names of the phenotypes.
        output_component : str, optional
            Name of the output component. Default is "binary_phenotype".
        vorigin_relative : Iterable, optional
            v-origin relative. Default is [-1].

        Returns
        -------
        xft.index.ComponentIndex
            The output component index.
        """
        return xft.index.ComponentIndex.from_product(phenotype_name,
                                                     [output_component],
                                                     vorigin_relative)

    @staticmethod
    def construct_cindexes(phenotype_name: Iterable,
                           liability_component: str = 'phenotype',
                           output_component: str = 'binary_phenotype',
                           vorigin_relative: Iterable = [-1],
                           ):
        """
        Constructs both input and output component indexes for the binarizing transformation.

        Parameters
        ----------
        phenotype_name : Iterable
            Names of the phenotypes.
        liability_component : str, optional
            Name of the liability component. Default is "phenotype".
        output_component : str, optional
            Name of the output component. Default is "binary_phenotype".
        vorigin_relative : Iterable, optional
            v-origin relative. Default is [-1].

        Returns
        -------
        Tuple[xft.index.ComponentIndex, xft.index.ComponentIndex]
            The input and output component indexes.
        """
        input_cindex = SumComponent.construct_input_cindex(phenotype_name,
                                                           liability_component,
                                                           vorigin_relative)
        output_cindex = SumComponent.construct_output_cindex(phenotype_name,
                                                             output_component,
                                                             vorigin_relative)
        return (input_cindex, output_cindex)

    def compute_component(self,
                          haplotypes: xr.DataArray,
                          phenotypes: xr.DataArray):
        """
        Computes the binarizing transformation.

        Parameters
        ----------
        haplotypes : xr.DataArray
            The haplotypes.
        phenotypes : xr.DataArray
            The phenotypes.
        """
        # TODO make faster later, UGLY atm
        y = phenotypes.loc[:, self.input_cindex.unique_identifier].data
        new_component = (y > self.thresholds).astype(int)
        phenotypes.loc[:, self.output_cindex.unique_identifier] = new_component


# Architecture
##
# an Architecture object consists of four pieces:
##
# - components: an iterable collection of ArchitectureComponent objects
# - initialize_next_generation: function taking current phenotypes, mating
# assignment, haplotypes, returns new empty Phenotype structure
# - initialize_founder_generation (optional): function taking
# haplotypes, returns new empty Phenotype structure
# - metadata (optional): dict
class Architecture:
    """
    Class representing a phenogenetic architecure

    Parameters
    ----------
    components: Iterable, optional
        An iterable collection of ArchitectureComponent objects
    metadata: Dict, optional
        A dictionary for holding metadata about the Architecture object
    depth: int, optional
        The generational depth of the architecture, default to 1
    expand_components: bool, optional
        A boolean flag indicating whether to expand the components, default to False
    
    Attributes
    ----------
    metadata: Dict
        A dictionary for holding metadata about the Architecture object
    components: Iterable
        An iterable collection of ArchitectureComponent objects
    depth: int
        The depth of the architecture
    expand_components: bool
        A boolean flag indicating whether to expand the components
    
    Methods
    -------
    founder_initializations() -> List:
        Get a list of the founder initialization of each component
    merged_component_indexer() -> xft.index.ComponentIndex:
        Get the merged component indexer
    initialize_phenotype_array(haplotypes: xr.DataArray, control: dict = None) -> xr.DataArray:
        Initialize a new phenotype array
    initialize_founder_phenotype_array(haplotypes: xr.DataArray, control: dict = None) -> xr.DataArray:
        Initialize a new founder phenotype array
    compute_phenotypes(haplotypes: xr.DataArray = None, phenotypes: xr.DataArray = None, control: dict = None) -> None:
        Compute phenotypes for the given haplotypes and phenotypes
    """

    def __init__(self,
                 components: Iterable = None,
                 metadata: Dict = dict(),
                 depth: int = 1,
                 expand_components: bool = False,
                 ):
        """
        Construct an Architecture object

        Parameters
        ----------
        components: Iterable, optional
            An iterable collection of ArchitectureComponent objects
        metadata: Dict, optional
            A dictionary for holding metadata about the Architecture object
        depth: int, optional
            The depth of the architecture, default to 1
        expand_components: bool, optional
            A boolean flag indicating whether to expand the components, default to False
        """
        self.metadata = metadata
        self.components = components
        self.depth = depth
        self.expand_components = expand_components

    @property
    def founder_initializations(self) -> List:
        """
        Get a list of the founder initialization of each component
        """
        return [x.founder_initialization for x in self.components if x.founder_initialization is not None]

    @property
    def merged_component_indexer(self) -> xft.index.ComponentIndex:
        """
        Get the merged ComponentIndex indexer across all archtecure components
        """
        merged = xft.index.XftIndex.reduce_merge(
            [x.merged_phenotype_indexer for x in self.components])
        if not self.expand_components:
            return merged
        elif self.expand_components:
            phenotype_name = np.unique(merged.phenotype_name)
            component_name = np.unique(merged.component_name)
            vorigin_relative = np.unique(merged.vorigin_relative)
            return xft.index.ComponentIndex.from_product(phenotype_name,
                                                         component_name,
                                                         vorigin_relative)

    def initialize_phenotype_array(self,
                                   haplotypes: xr.DataArray,
                                   control: dict = None,
                                   ) -> xr.DataArray:
        """
        Initialize a phenotype array from haplotypes under the specified architecture.

        Parameters
        ----------
        haplotypes : xr.DataArray
            Input haplotypes.
        control : dict, optional
            Dictionary containing control parameters.

        Returns
        -------
        xr.DataArray
            Phenotype array with the merged component indexer and sample indexer.
        """
        sample_indexer = haplotypes.xft.get_sample_indexer()
        return xft.struct.PhenotypeArray(component_indexer=self.merged_component_indexer,
                                         sample_indexer=haplotypes.xft.get_sample_indexer(),
                                         generation=haplotypes.attrs['generation'])

    def initialize_founder_phenotype_array(self,
                                           haplotypes: xr.DataArray,
                                           control: dict = None,
                                           ) -> xr.DataArray:
        """
        Initialize a founder generation phenotype array from haplotypes under the specified architecture.
        In the absense of vertical transmission, this is equivalent to `initialize_phenotype_array()`.

        Parameters
        ----------
        haplotypes : xr.DataArray
            Input haplotypes.
        control : dict, optional
            Dictionary containing control parameters.

        Returns
        -------
        xr.DataArray
            Phenotype array with the merged component indexer and sample indexer.
        """
        phenotype_array = self.initialize_phenotype_array(haplotypes)
        for initialization in self.founder_initializations:
            initialization.initialize_component(phenotype_array)
        return phenotype_array

    def compute_phenotypes(self,
                           haplotypes: xr.DataArray = None,
                           phenotypes: xr.DataArray = None,
                           control: dict = None,
                           ) -> None:
        """
        Compute phenotypes.

        Parameters
        ----------
        haplotypes : xr.DataArray, optional
            Input haplotypes.
        phenotypes : xr.DataArray, optional
            Input phenotypes.
        control : dict, optional
            Dictionary containing control parameters.
        """
        if self.components is None:
            raise NotImplementedError
        for component in self.components:
            component.compute_component(haplotypes, phenotypes)


class InfinitessimalArchitecture:
    def __init__(self):
        NotImplementedError  # TODO


class SpikeSlabArchitecture:
    def __init__(self):
        NotImplementedError  # TODO


# class MaternalVerticalComponent(LinearTransformationComponent):
#     def __init__(self,
#                  phenotype_name: Iterable,
#                  component_name: Iterable,
#                  output_cindex: Iterable,
#                  coefficient_matrix: NDArray=None,
#                  output_component = 'maternalVertical',
#                  normalize: bool = True,
#                  ):
#         super().__init__(phenotype_name=phenotype_name,
#                        component_name=component_name,
#                        vorigin_relative=0,
#                        output_cindex=output_cindex,
#                        coefficient_matrix=coefficient_matrix,
#                        output_component = output_component,
#                        normalize=normalize
#                        )

# class PaternalVerticalComponent(LinearTransformationComponent):
#     def __init__(self,
#                  phenotype_name: Iterable,
#                  component_name: Iterable,
#                  output_cindex: Iterable,
#                  coefficient_matrix: NDArray=None,
#                  output_component = 'paternalVertical',
#                  normalize: bool = True,
#                  ):
#         super().__init__(phenotype_name=phenotype_name,
#                        component_name=component_name,
#                        vorigin_relative=1,
#                        output_cindex=output_cindex,
#                        coefficient_matrix=coefficient_matrix,
#                        output_component = output_component,
#                        normalize=normalize
#                        )

# class MeanVerticalComponent(LinearTransformationComponent):
#     def __init__(self,
#                  phenotype_name: Iterable,
#                  component_name: Iterable,
#                  output_cindex: Iterable,
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
#                        output_cindex=output_cindex,
#                        coefficient_matrix=np.hstack([coefficient_matrix,coefficient_matrix]),
#                        output_component = output_component,
#                        normalize=normalize,
#                        multiplier=.5
#                        )
