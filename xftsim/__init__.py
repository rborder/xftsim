import numpy as np

class Config:
    """
    A class to store configuration settings. Instantiated as xftsim.config when package is loaded

    Attributes
    ----------
    nthreads : int
        Number of threads to use for parallel execution.
    print_level : int
        Verbosity level for print statements.
    print_durations_threshold : float
        Threshold for printing durations.
    """

    def __init__(self):
        """
        Initialize the Config object with default settings.
        """
        self.nthreads = 4
        self.print_level = 2
        self.print_durations_threshold = 0. #np.inf

    def get_pdurations(self):
        """
        Get the current print durations threshold.

        Returns
        -------
        float
            The print durations threshold.
        """
        return self.print_durations_threshold

    def get_plevel(self):
        """
        Get the current print level.

        Returns
        -------
        int
            The print level.
        """
        return self.print_level


config = Config()

from . import utils       ## utility functions   
#from . import data        ## download recombination maps etc
from . import index       ## indexing
from . import struct      ## data structures
from . import effect      ## genetic effects
from . import arch        ## phenogenetic architectures
from . import mate        ## mate assignment
from . import reproduce   ## sexual reproduction and phenotypic transmission
from . import founders    ## creation / import of founder haplotypes
from . import sim         ## simulation object
from . import stats       ## estimation
from . import proc        ## post-processing
from . import io          ## input/output

