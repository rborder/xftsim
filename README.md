<img src="./xftsimlogo.svg" width="20%"> 

[![Documentation Status](https://readthedocs.org/projects/xftsim/badge/?version=latest)](https://xftsim.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/xftsim.svg)](https://badge.fury.io/py/xftsim)

# eXtensible Forward Time SIMulator
`xftsim` simulates complex phenotype/genotype data with an emphasis on short timescale phenomena. `xftsim` is designed with two primary goals:

 - make it easy for statistical geneticists to perform reproducible and systematic sensitivity analyses to better understand limitations and assumptions

 - enable evaulation of methods for analyzing complex traits under realistically complex generative models

## Installation

`xftsim` is on PyPI. It can be installed using pip and conda:

```bash
conda create --name xftsim python=3.9.6
conda activate xftsim
pip install xftsim
```

Alternatively, you can install the latest development version from github:

```bash
git clone https://github/rborder/xftsim
cd xftsim
git checkout dev
pip -e . install
```

To enable full functionality (i.e., automatic generation of causal diagrams), you must install [pygraphviz](https://pygraphviz.github.io).

`xftsim` has been tested on MacOS 13.4 and the following GNU/Linux distributions: Ubuntu 22.04, Ubuntu 24.04, PopOS 22.04, and RHEL 7 using Python version 3.9.6. 

<details>
<summary>`xftsim` depends on the following packages</summary>summary

```
asciitree==0.3.3
attrs==23.2.0
cattrs==23.2.3
certifi==2023.11.17
cffi==1.17.1
charset-normalizer==3.3.2
chembl-webresource-client==0.10.9
click==8.1.7
cloudpickle==3.1.0
contourpy==1.2.0
csrgraph==0.1.28
cycler==0.12.1
dask==2024.8.0
dask-expr==1.1.10
dask-glm==0.3.2
dask-ml==2024.4.4
Deprecated==1.2.14
distributed==2024.8.0
easydict==1.13
exceptiongroup==1.2.0
fasteners==0.19
fonttools==4.46.0
fsspec==2024.10.0
funcy==2.0
gensim==4.3.2
idna==3.6
importlib-metadata==8.5.0
importlib-resources==6.1.1
iniconfig==2.0.0
jinja2==3.1.4
joblib==1.3.2
kiwisolver==1.4.5
llvmlite==0.39.1
locket==1.0.0
MarkupSafe==3.0.2
matplotlib==3.8.2
msgpack==1.1.0
multipledispatch==1.0.0
networkx==2.8.8
node2vec==0.4.6
nodevectors==0.1.23
nptyping==2.5.0
numba==0.56.4
numcodecs==0.12.1
numpy==1.23.5
packaging==23.2
pandas==2.1.4
pandas-plink==2.2.9
partd==1.4.2
Pillow==10.1.0
platformdirs==4.2.0
pluggy==1.5.0
psutil==6.1.0
pyarrow==18.0.0
pycparser==2.22
pyparsing==3.1.1
pytest==8.3.3
python-dateutil==2.8.2
pytz==2023.3.post1
PyYAML==6.0.2
rdkit==2023.9.5
requests==2.31.0
requests-cache==1.2.0
scikit-learn==1.3.2
scipy==1.11.4
seaborn==0.13.0
sgkit==0.9.0
six==1.16.0
smart-open==7.0.1
sortedcontainers==2.4.0
sparse==0.15.4
tblib==3.0.0
threadpoolctl==3.2.0
tomli==2.0.2
toolz==1.0.0
tornado==6.4.1
tqdm==4.66.2
typing-extensions==4.10.0
tzdata==2023.3
url-normalize==1.4.3
urllib3==2.1.0
wrapt==1.16.0
xarray==2024.7.0
xftsim==0.2.0
zarr==2.18.2
zict==3.0.0
zipp==3.20.2
zstandard==0.23.0
```
</details>

## Getting started

To get started, [check out the documentation](https://xftsim.readthedocs.io)!

For a minimal test simulation you can run one of the built in demos:
```python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import xftsim as xft

demo = xft.sim.DemoSimulation('BGRM')
demo.run(3)

xft.utils.print_tree(demo.results)
```

## Notice

`xftsim` is under active development. Please let us know if there are features missing or bugs!

<!-- 
## Quickstart: simulating bivariate cross-assortative mating

Here we simulate 


```python

import xftsim as xft
import numpy as np

N = 8000
M = 4000
pnames = ['height', 'wealth', 'eduyears']
h2 = np.array([.6,.0,.0])


founder_haplotypes = xft.founders.founder_haplotypes_uniform_AFs(n = N, 
                                                                 m = M)

genetic_effects = xft.effect.AdditiveEffects(beta = np.hstack(list(map(lambda x: np.random.normal(0, x, (M,1)), np.sqrt(h2)))),
                                             phenotype_name = pnames,
                                             vid = founder_haplotypes.vid,
                                             AF = founder_haplotypes.xft.af_empirical,
                                             standardized=True,
                                             scaled=True,
                                             m_causal=M)

arch_genetic = xft.arch.AdditiveGeneticComponent(beta = genetic_effects)
arch_noise = xft.arch.AdditiveNoiseComponent(variances=[.4, 1/3, 1/3], 
                                             phenotype_name=pnames)
arch_sum = xft.arch.SumComponent(pnames, sum_components=['additiveGenetic', 'additiveNoise'])




amr = xft.mate.LinearAssortativeMatingRegime(r = .3, 
                                             component_index = xft.index.ComponentIndex_from_product(pnames,
                                              ['phenotype'],
                                              [-1]),
                                             offspring_per_pair=xft.utils.ZeroTruncatedPoissonCount(2))

rmap = xft.reproduce.RecombinationMap(p=.25,
                                      vid=founder_haplotypes.vid,
                                      chrom=founder_haplotypes.chrom)

sim = xft.sim.Simulation(founder_haplotypes = founder_haplotypes,
                         mating_regime = amr,
                         recombination_map = rmap,
                         architecture=xft.arch.Architecture([arch_genetic, arch_noise, arch_sum]),
                         statistics = [xft.stats.MatingStatistics(),
                                       xft.stats.SampleStatistics(),
                                       xft.stats.HasemanElstonEstimator(),
                                       ],  
                         post_processors = [lambda sim: print(sim.results['mating_statistics']),
                                            xft.proc.LimitMemory(n_haplotype_generations=2)],
                         reproduction_method=xft.reproduce.Meiosis)

```





 -->
