import warnings
import numpy as np
import pandas as pd
import math
import functools

from typing import Iterable, Union
from nptyping import NDArray, Shape, Float, Int, Object 
from collections.abc import Sequence

import xftsim as xft


import traceback
import warnings
import sys


import sgkit as sg

import sgkit.io.vcf as vcf

vcf.vcf_to_zarr('1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz',
                '1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.zarr')