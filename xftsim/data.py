import pandas as pd
import os, sys


def get_ceu_map():
    """
    Load the CEU haplotype map.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the CEU haplotype map.

    """
    package_root = os.path.dirname(sys.modules['xftsim'].__file__)
    return pd.read_csv(package_root + '/../maps/ceu.hg19.map')
