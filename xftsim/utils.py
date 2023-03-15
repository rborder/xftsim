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


# classes for indexing data structures
def ids_from_n_generation(n: int,
                          generation: int,
                          ):
    return np.char.add(str(generation) + "_", np.arange(n).astype(str))


def paste(it, sep="_"):
    output = functools.reduce(lambda x, y: np.char.add(np.char.add(np.array(x).astype(str), sep),
                                                       np.array(y).astype(str)), it)
    return np.asarray(output)


def merge_duplicates(it):
    return it
    pasted = paste(it)
    unique_inds = np.unique(pasted, return_index=True)[1]
    if unique_inds.shape[0] < pasted.shape[0]:
        warnings.warn("merging duplicate coords")
    return [x[unique_inds] for x in it]


def ids_from_generation(generation: int,
                        indices: NDArray[Shape["*"], Int64] = None,
                        ):
    return np.char.add(str(generation) + "_", indices.astype(str))


def ids_from_generation_range(generation: int,
                              n: NDArray[Shape["*"], Int64] = None,
                              ):
    return np.char.add(str(generation) + "_", np.arange(n).astype(str))


# random sampling with replacement with replacement only occuring after
# each element has been selected once, then twice, then so on.
def exhaustive_permutation(a: NDArray[Shape["*"], Any], n_sample: int):
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
    sorted = np.sort(x).astype(str)
    return np.char.add(sorted[0],
                       np.char.add('_', sorted[1]))

# def paste(x):
    # xx = x.astype(str)
    # return np.char.add(xx[0],
    # np.char.add('_', xx[1]))


def merge_duplicate_pairs(a, b, n, sort=False):
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
    b_dict = {x: i for i, x in enumerate(b)}
    return np.array([b_dict.get(x, None) for x in a])

# find indices of b meeting condition in order of corresponding elements in a
# useful for select columns of phenotype object


def matching_indices_conditional(
    a: List[Hashable],
    b: List[Hashable],
    condition: NDArray[Shape["*"], Any],
) -> NDArray[Shape["*"], Int64]:
    return np.where(condition)[0][match(a, b[condition])]


# make sure arrays are 2d, but preserve None
def ensure2D(x: NDArray[Any, Any] = None):
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
    return [a.flatten() for a in np.meshgrid(*args, indexing="ij")]


import numpy as np
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Dict, Callable, Union


class VariableCount:
    def __init__(self,
                 draw: Callable,
                 expectation: float = None,
                 mating_fraction: float = None):
        self.draw = draw
        self._expectation = expectation
        self._expectation_impl = (expectation is not None)
        self._mating_fraction = mating_fraction
        self._mating_fraction_impl = (mating_fraction is not None)

    @property
    def expectation(self):
        if self._expectation_impl:
            return self._expectation
        else:
            raise NotImplementedError("'expectation' not implemented")

    @expectation.setter
    def expectation(self, value):
        if value is not None:
            self._expectation_impl = True
        else:
            self._expectation_impl = False
        self._expectation = value

    @property
    def mating_fraction(self):
        if self._mating_fraction_impl:
            return self._mating_fraction
        else:
            raise NotImplementedError("'mating_fraction' not implemented")

    @mating_fraction.setter
    def mating_fraction(self, value):
        if value is not None:
            self._mating_fraction_impl = True
        else:
            self._mating_fraction_impl = False
        self._mating_fraction = value


class ConstantCount(VariableCount):
    def __init__(self, count: int):
        assert type(count) == int and count >= 0
        super().__init__(
            draw=lambda n: np.repeat(count, n),
            expectation=count,
            mating_fraction=float(bool(count)),
        )


class PoissonCount(VariableCount):
    def __init__(self, rate: float):
        assert rate >= 0, "Invalid rate"
        super().__init__(
            draw=lambda n: np.random.poisson(rate, n),
            expectation=rate,
            mating_fraction=1 - np.exp(-rate)
        )


class ZeroTruncatedPoissonCount(VariableCount):
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
    def __init__(self, r: float, p: float):
        assert r > 0 and 1 >= p >= 0, "Invalid parameters"
        super().__init__(
            draw=lambda n: np.random.poisson(rate, r),
            expectation=r * (1 - p) / p**2,
            mating_fraction=1. - (1 - p) * p**r
        )


class MixtureCount(VariableCount):
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
    with np.errstate(invalid='ignore'):
        a = ensure2D(a)
        return ensure2D(np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x),
                                            0, a))

# map int8 haploid genotypes, float afs to float32 standardized genotypes


@nb.jit(parallel=True)
def _standardize_array_hw(haplotypes,
                          af):
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
    return paste([frame[variable].values for variable in index_variables], sep=".")


def cov2cor(A):
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
