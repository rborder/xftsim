import warnings
import numpy as np
import numba as nb
import pandas as pd
import nptyping as npt
import xarray as xr
import scipy as sp
from nptyping import NDArray, Int8, Int64, Float64, Float, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
from dataclasses import dataclass, field
import functools
from numpy.typing import ArrayLike
from scipy import stats
import funcy

import xftsim as xft

@funcy.decorator
def profiled(call, level: int = 1, message: str = None):
    """
    A decorator that prints the duration of a function call when the specified logging level is met.
    
    Parameters
    ----------
    call : function
        The function being decorated.
    level : int, optional
        The logging level at which the duration of the function call is printed.
        Defaults to 1.
    message : str, optional
        A custom message to display in the log output. If not provided, the name of
        the decorated function will be used.
    
    Returns
    -------
    TYPE
        Description
    """
    if message is not None:
        msg = message
    else:
        msg = call._func.__name__
    if level <= xft.config.get_plevel():
        with funcy.print_durations(msg, threshold=xft.config.get_pdurations()):
            return call()
    else:
        return call()


# classes for indexing data structures
def ids_from_n_generation(n: int,
                          generation: int,
                          ):
    """
    Creates an array of individual IDs based on the specified number of elements and generation.
    
    Parameters
    ----------
    n : int
        The number of individuals.
    generation : int
        The generation number.
    
    Returns
    -------
    numpy.ndarray
        An array of individual IDs.
    """
    return np.char.add(str(generation) + "_", np.arange(n).astype(str))


def paste(it, sep="_"):
    """
    Concatenates elements in a list-like object with a specified separator.
    
    Parameters
    ----------
    it : list-like
        The list-like object containing elements to concatenate.
    sep : str, optional
        The separator used to concatenate the elements. Defaults to "_".
    
    Returns
    -------
    numpy.ndarray
        An array of concatenated string elements.
    """
    output = functools.reduce(lambda x, y: np.char.add(np.char.add(np.array(x).astype(str), sep),
                                                       np.array(y).astype(str)), it)
    return np.asarray(output)


def merge_duplicates(it: Iterable):
    """
    Merge duplicates in the input array by checking if any pasted elements are the same. 
    
    Parameters
    ----------
    it : Iterable
        A numpy array with elements to be checked for duplication.
    
    Returns
    -------
    list
        Returns the input list with duplicates merged if present.
    """
    return it
    pasted = paste(it)
    unique_inds = np.unique(pasted, return_index=True)[1]
    if unique_inds.shape[0] < pasted.shape[0]:
        warnings.warn("merging duplicate coords")
    return [x[unique_inds] for x in it]


def ids_from_generation(generation: int,
                        indices: NDArray[Shape["*"], Int64] = None,
                        ):
    """Generates and returns a new array of IDs using the given generation number 
    and the given indices. The new array contains the given indices with the
    generation number prefixed to each index.
    
    Parameters
    ----------
    generation : int
        The generation number to use in the prefix of the IDs.
    indices : NDArray[Shape["*"], Int64], optional
        A numpy array of indices.
    
    Returns
    -------
    ndarray
        A new numpy array of IDs with the given generation number prefixed to each index.
    """
    return np.char.add(str(generation) + "_", indices.astype(str))


def ids_from_generation_range(generation: int,
                              n: NDArray[Shape["*"], Int64] = None,
                              ):
    """
    Returns an array of string IDs of length n, created by concatenating 
    the input generation with an increasing sequence of integers from 0 to n-1.

    Parameters:
    -----------
    generation : int
        An integer representing the generation of the IDs to be created.
    n : NDArray[Shape["*"], Int64], optional (default=None)
        An integer specifying the number of IDs to be generated. If None,
        a range of IDs starting from 0 is created.

    Returns:
    --------
    np.ndarray
        A 1D numpy array containing the IDs in string format.
    """
    return np.char.add(str(generation) + "_", np.arange(n).astype(str))


def exhaustive_permutation(a: NDArray[Shape["*"], Any], 
                           n_sample: int):
    """
    Returns a random permutation of the input array, such that each element 
    is selected exactly once before any element is selected twice, and so forth

    Parameters:
    -----------
    a : NDArray[Shape["*"], Any]
        A numpy array to be permuted.
    n_sample : int
        An integer specifying the size of the permutation to be returned.

    Returns:
    --------
    np.ndarray
        A 1D numpy array containing the permuted elements.
    """
    a = np.array(a)
    n = a.shape[0]
    if n_sample <= n:
        perms = np.random.permutation(a)[:n_sample]
    else:
        perms = np.concatenate(
            [np.random.permutation(a) for i in range(1 + (n_sample // n))]
        )
    return perms[:n_sample]


# repeat each ith element of array a integer n_per_a[i] times
# such that each every element appears min(j,  n_per_a[i]) times
# in order before any element appears j+1 times.
# example ((1,2,3,4), (3,2,1,0)) -> (1,2,3,1,2,1)
def exhaustive_enumerate(a: NDArray[Shape["*"], Any],
                         n_per_a: NDArray[Shape["*"], Int64]):
    """
    Repeat each ith element of array a integer n_per_a[i] times
    such that each every element appears min(j,  n_per_a[i]) times
    in order before any element appears j+1 times.

    Parameters:
    -----------
    a : array-like
        1-D array of any shape and data type.
    n_per_a : array-like
        1-D array of int, representing the number of times each element
        in `a` needs to be repeated.

    Returns:
    --------
    out : array-like
        1-D array of shape `(n,)` and the same data type as `a`, where
        each element is repeated as per `n_per_a` in the order before
        any element appears `j+1` times.

    Raises:
    -------
    Warning : If the output array is empty.
    
    Examples:
    ---------
    >>> exhaustive_enumerate(np.array((1, 2, 3, 4)), np.array((3, 2, 1, 0)))
    array([1, 2, 3, 1, 2, 1])
    """
    a = np.array(a)
    n_per_a = np.array(n_per_a)
    output = []
    to_be_sampled = (n_per_a >= 1)
    while np.any(to_be_sampled):
        output += [a[np.where(to_be_sampled)[0]]]
        n_per_a -= 1
        to_be_sampled = (n_per_a >= 1)
    if len(output) > 0:
        return np.concatenate(output)
    else:
        warnings.warn("Output is empty")
        return np.array([], dtype=a.dtype)


def sort_and_paste(x):
    """
    Sorts the input array in ascending order and concatenates the 
    first element with an underscore separator followed by the second
    element.

    Parameters:
    -----------
    x : array-like
        1-D array of any shape and data type.

    Returns:
    --------
    out : array-like
        1-D array of strings with shape `(n,)` and the same length as `x`,
        where each element is formed by concatenating two sorted string
        representations of each element in `x`, separated by an underscore.

    Examples:
    ---------
    >>> sort_and_paste(np.array((3, 1, 2)))
    array(['1_2', '2_3', '1_3'], dtype='<U3')
    """
    sorted = np.sort(x).astype(str)
    return np.char.add(sorted[0],
                       np.char.add('_', sorted[1]))


def merge_duplicate_pairs(a, b, n, sort=False):
    """
    Merge duplicate pairs of values in a and b based on their corresponding values in n.

    Parameters:
    -----------
    a : NDArray[Shape["*"], Any]
        First array to merge.
    b : NDArray[Shape["*"], Any]
        Second array to merge.
    n : NDArray[Shape["*"], Any]
        Array of corresponding values that determine how the duplicates are merged.
    sort : bool, optional
        Whether to sort the values in a and b before merging the duplicates. Default is False.

    Returns:
    --------
    Tuple[NDArray[Shape["*"], Any], NDArray[Shape["*"], Any], NDArray[Shape["*"], Any]]
        The merged arrays, with duplicates removed based on the corresponding values in n.

    """

    a = np.array(a)
    b = np.array(b)
    if sort:
        sorted = np.apply_along_axis(np.sort, 0, np.vstack([a, b]))
        a = sorted[0, :]
        b = sorted[1, :]
    n = np.array(n)
    # if sort:
    # pairings = np.apply_along_axis(sort_and_paste, 1, df.iloc[:,:2])
    # else:
    # pairings = np.apply_along_axis(paste, 1, df.iloc[:,:2])
    df = pd.DataFrame.from_dict({'a': a, 'b': b, 'n': n})
    aggregated = df.groupby(['a', 'b'], as_index=False).sum(numeric_only=False)
    # TODO issue with families mating more than once
    return np.array(aggregated.a), np.array(aggregated.b), np.array(aggregated.n)


# from https://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r
def match(a: List[Hashable], b: List[Hashable]) -> List[int]:
    """
    Finds the indices in b that match the elements in a, and returns the corresponding
    index of each element in b.

    Parameters:
    -----------
    a : List[Hashable]
        List of elements to find matches for.
    b : List[Hashable]
        List of elements to find matches in.

    Returns:
    --------
    List[int]
        A list of indices in b that match the elements in a.

    """
    b_dict = {x: i for i, x in enumerate(b)}
    return np.array([b_dict.get(x, None) for x in a])

# find indices of b meeting condition in order of corresponding elements in a
# useful for select columns of phenotype object
def matching_indices_conditional(
    a: List[Hashable],
    b: List[Hashable],
    condition: NDArray[Shape["*"], Any],
) -> NDArray[Shape["*"], Int64]:
    """
    Returns the indices of matches between a and b arrays, given a boolean 
    condition.
    
    Parameters
    ----------
    a : List[Hashable]
        The first input array.
    b : List[Hashable]
        The second input array.
    condition : NDArray[Shape["*"], Any]
        The boolean condition array to apply.
        
    Returns
    -------
    NDArray[Shape["*"], Int64]
        The matching indices array.
    """
    return np.where(condition)[0][match(a, b[condition])]


# make sure arrays are 2d, but preserve None
def ensure2D(x: NDArray[Any, Any] = None):
    """
    Ensures the input array is 2D, by adding a new dimension if needed.
    
    Parameters
    ----------
    x : NDArray[Any, Any], optional
        The input array, by default None.
    
    Returns
    -------
    NDArray[Any, Any]
        The 2D input array.
    
    Raises
    ------
    ValueError
        If the input array is not valid.
    """
    if x is None:
        return x
    x = np.array(x)
    if len(x.shape) == 2:
        return x
    elif len(x.shape) == 1:
        return x[:, None]
    else:
        raise ValueError("Invalid array")


# map arrays -> list of columns comprising cartestian product as in expand.grid in R
# but in order of pd.MultiIndex.from_product
def cartesian_product(*args):
    """
    Returns a list of columns comprising a cartesian product of input arrays. Emulates
    R function `expand.grid()`
    
    Parameters
    ----------
    *args : NDArray[Any, Any]
        The input arrays.
    
    Returns
    -------
    List[NDArray[Any, Any]]
        The list of columns.
    """
    return [a.flatten() for a in np.meshgrid(*args, indexing="ij")]


import numpy as np
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Dict, Callable, Union


class VariableCount:
    """
    A class to represent random count variables

    ...

    Attributes
    ----------
    draw : Callable
        a function that generates an array of counts
    expectation : float
        expected count
    mating_fraction : float
        the fraction of the population that is expected to mate

    Methods
    -------
    None
    """
    
    def __init__(self,
                 draw: Callable,
                 expectation: float = None,
                 mating_fraction: float = None):
        """
        Constructs all the necessary attributes for the VariableCount object.

        Parameters
        ----------
            draw : Callable
                a function that generates an array of counts
            expectation : float
                expected count
            mating_fraction : float
                the fraction of the population that is expected to mate

        Returns
        -------
            None
        """
        self.draw = draw
        self._expectation = expectation
        self._expectation_impl = (expectation is not None)
        self._mating_fraction = mating_fraction
        self._mating_fraction_impl = (mating_fraction is not None)

    @property
    def expectation(self):
        """
        Getter function for expectation attribute.

        Returns
        -------
        float
            Expected count.
        """
        if self._expectation_impl:
            return self._expectation
        else:
            raise NotImplementedError("'expectation' not implemented")

    @expectation.setter
    def expectation(self, value):
        """
        Setter function for expectation attribute.

        Parameters
        ----------
            value : float
                expected count

        Returns
        -------
            None
        """
        if value is not None:
            self._expectation_impl = True
        else:
            self._expectation_impl = False
        self._expectation = value

    @property
    def mating_fraction(self):
        """
        Getter function for mating_fraction attribute.

        Returns
        -------
        float
            The fraction of the population that is expected to mate.
        """
        if self._mating_fraction_impl:
            return self._mating_fraction
        else:
            raise NotImplementedError("'mating_fraction' not implemented")

    @mating_fraction.setter
    def mating_fraction(self, value):
        """
        Setter function for mating_fraction attribute.

        Parameters
        ----------
            value : float
                The fraction of the population that is expected to mate.

        Returns
        -------
            None
        """
        if value is not None:
            self._mating_fraction_impl = True
        else:
            self._mating_fraction_impl = False
        self._mating_fraction = value


class ConstantCount(VariableCount):
    """
    Class representing a constant count of individuals in a population.

    Attributes
    ----------
    draw : Callable
        a function that generates an array of counts
    expectation : float
        expected count
    mating_fraction : float
        the fraction of the population that is expected to mate

    Parameters
    ----------
    count : int
        The constant count of individuals in the population.
    """
    def __init__(self, count: int):
        assert type(count) == int and count >= 0
        super().__init__(
            draw=lambda n: np.repeat(count, n),
            expectation=count,
            mating_fraction=float(bool(count)),
        )


class PoissonCount(VariableCount):
    """
    Class representing a Poisson-distributed count of individuals in a population.
    Attributes
    ----------
    draw : Callable
        a function that generates an array of counts
    expectation : float
        expected count
    mating_fraction : float
        the fraction of the population that is expected to mate

    Parameters
    ----------
    rate : float
        The Poisson rate parameter.
    """
    def __init__(self, rate: float):
        assert rate >= 0, "Invalid rate"
        super().__init__(
            draw=lambda n: np.random.poisson(rate, n),
            expectation=rate,
            mating_fraction=1 - np.exp(-rate)
        )


class ZeroTruncatedPoissonCount(VariableCount):
    """
    Class representing a zero-truncated Poisson-distributed count of individuals in a population.

    Attributes
    ----------
    draw : Callable
        a function that generates an array of counts
    expectation : float
        expected count
    mating_fraction : float
        the fraction of the population that is expected to mate

    Parameters
    ----------
    rate : float
        The Poisson rate parameter prior to zero-truncation.
    """
    def __init__(self, rate: float):
        assert rate >= 0, "Invalid rate"
        min_unif = sp.stats.poisson.cdf(0, mu=rate)
        super().__init__(
            draw=lambda n: sp.stats.poisson.ppf(np.random.uniform(min_unif, 1, n),
                                                mu=rate).astype(int),
            expectation=(rate) / (1 - min_unif),
            mating_fraction=1
        )

# todo mating fraction -> nonzero fraction


class NegativeBinomialCount(VariableCount):
    """
    Class representing a negative binomial-distributed count of individuals in a population.

    Attributes
    ----------
    draw : Callable
        a function that generates an array of counts
    expectation : float
        expected count
    mating_fraction : float
        the fraction of the population that is expected to mate

    Parameters
    ----------
    r : float
        The number of successes in the negative binomial distribution.
    p : float
        The probability of success in the negative binomial distribution.
    """
    def __init__(self, r: float, p: float):
        assert r > 0 and 1 >= p >= 0, "Invalid parameters"
        super().__init__(
            draw=lambda n: np.random.poisson(rate, r),
            expectation=r * (1 - p) / p**2,
            mating_fraction=1. - (1 - p) * p**r
        )


class MixtureCount(VariableCount):
    """
    Class representing a mixture of VariableCounts of individuals in a population.

    Attributes
    ----------
    draw : Callable
        a function that generates an array of counts
    expectation : float
        expected count
    mating_fraction : float
        the fraction of the population that is expected to mate

    Parameters
    ----------
    componentCounts : Iterable
        An iterable of VariableCount instances, representing the components of the mixture.
    mixture_probabilities : NDArray[Shape["*"], Float64]
        An array of probabilities associated with each component in the mixture.
    """
    def __init__(self,
                 componentCounts: Iterable,
                 mixture_probabilities: NDArray[Shape["*"], Float64],
                 ):
        def draw(n):
            choices = np.random.choice(
                len(componentCounts), size=n, replace=True, p=mixture_probabilities)
            output = np.zeros(n, dtype=int)
            for i in range(len(componentCounts)):
                output[choices == i] = componentCounts[i].draw(
                    np.sum(choices == i))
            return output
        if all([component._expectation_impl for component in componentCounts]):
            expectation = np.sum([component.expectation * p for (component, p)
                                  in zip(componentCounts, mixture_probabilities)])
        else:
            expectation = None
        if all([component._mating_fraction_impl for component in componentCounts]):
            mating_fraction = np.sum([component._mating_fraction * p for (
                component, p) in zip(componentCounts, mixture_probabilities)])
        else:
            mating_fraction = None
        super().__init__(
            draw=draw,
            expectation=expectation,
            mating_fraction=mating_fraction,
        )


# standardize columns of array
def standardize_array(a: ArrayLike):
    """
    Standardizes columns of a 2D array.

    Parameters:
    -----------
    a: ArrayLike
        Input 2D array.

    Returns:
    --------
    np.ndarray
        Standardized 2D array.

    Raises:
    -------
    None
    """
    with np.errstate(invalid='ignore'):
        a = ensure2D(a)
        return ensure2D(np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x),
                                            0, a))

# map int8 haploid genotypes, float afs to float32 standardized genotypes


@nb.jit(parallel=True)
def _standardize_array_hw(haplotypes,
                          af):
    """
    Maps int8 haploid genotypes and float allele frequency to float32 standardized genotypes.

    Parameters:
    -----------
    haplotypes: np.ndarray
        Input array of int8 haploid genotypes.
    af: np.ndarray
        Input array of allele frequencies.

    Returns:
    --------
    np.ndarray
        Standardized genotypes.

    Raises:
    -------
    None
    """
    genotypes = np.empty(shape=(haplotypes.shape[0],
                                haplotypes.shape[1] // 2), dtype=np.float32)
    means = (2 * af).astype(np.float32)
    stds = np.sqrt(2 * af * (1 - af)).astype(np.float32)
    for j in range(0, haplotypes.shape[1] // 2):
        genotypes[:, j] = (haplotypes[:, 2 * j] +
                           haplotypes[:, 2 * j + 1] - means[j]) / stds[j]
    return genotypes
# wrap to prevent segfaults


def standardize_array_hw(haplotypes: NDArray[Shape["*,*"], Int8],
                         af: NDArray[Shape["*"], Float]):
    """
    Wraps _standardize_array_hw to prevent segfaults.

    Parameters:
    -----------
    haplotypes: NDArray[Shape["*,*"], Int8]
        Input array of int8 haploid genotypes.
    af: NDArray[Shape["*"], Float]
        Input array of allele frequencies.

    Returns:
    --------
    np.ndarray
        Standardized genotypes.

    Raises:
    -------
    None
    """
    with np.errstate(invalid='ignore'):
        # flatten af if needed
        af = np.array(af)
        assert af.shape[0] == af.size
        af = af.ravel()
        # make sure haplotype array is haploid and conformable with af
        haplotypes = np.array(haplotypes)
        assert haplotypes.shape[1] % 2 == 0
        assert haplotypes.shape[1] // 2 == af.shape[0]
        return _standardize_array_hw(haplotypes, af)


def unique_identifier(frame, index_variables):
    """
    Returns a unique identifier string generated from index variables of a dataframe.

    Parameters:
    -----------
    frame: pd.DataFrame
        Input dataframe.
    index_variables: List[str]
        List of column names to be used as index.

    Returns:
    --------
    str
        Unique identifier string.

    Raises:
    -------
    None
    """
    return paste([frame[variable].values for variable in index_variables], sep=".")


def cov2cor(A):
    """
    Converts covariance matrix to correlation matrix.

    Parameters:
    -----------
    A: Union[np.ndarray, pd.DataFrame, xr.DataArray]
        Input covariance matrix.

    Returns:
    --------
    Union[np.ndarray, pd.DataFrame, xr.DataArray]
        Correlation matrix.

    Raises:
    -------
    None
    """
    with np.errstate(invalid='ignore'):
        S = np.diag(1 / np.sqrt(np.diag(A)))
        if isinstance(A, xr.DataArray):
            cols = A.xft.as_pd().columns
            # there must be a better way than this...
            output = pd.DataFrame(
                S @ A @ S).set_index(A.index).T.set_index(A.index)
        elif isinstance(A, pd.DataFrame):
            output = pd.DataFrame(
                S @ A @ S).set_index(A.index).T.set_index(A.index)
        else:
            output = S @ A @ S
        return output


def to_simplex(*args):
    """
    Converts input values to a simplex vector.

    Parameters:
    -----------
    *args: Union[float, int]
        Input values.

    Returns:
    --------
    np.ndarray
        Simplex vector.

    Raises:
    -------
    ValueError
        If all input values are less than or equal to zero.
    """
    a = np.array(args).ravel()
    if np.all(a <= 0): 
        raise ValueError()
    while np.any(a < 0):
        a += np.min(a)
    return a / sum(a)

def to_proportions(*args):
    """
    Converts input values to proportional values.

    Parameters:
    -----------
    *args: Union[float, int]
        Input values.

    Returns:
    --------
    np.ndarray
        Proportional values.

    Raises:
    -------
    None
    """
    simplex = to_simplex(args)
    for i in range(simplex.shape[0]):
        if np.any(simplex < 1):
            simplex *= 1/np.min(simplex)
    return simplex