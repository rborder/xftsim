import pandas as pd
import os, sys


def get_ceu_map():
    package_root = os.path.dirname(sys.modules['xftsim'].__file__)
    return pd.read_csv(package_root + '/../maps/ceu.hg19.map')
