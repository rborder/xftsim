import numpy as np

class Config:
    def __init__(self):
        self.nthreads = 4
        self.print_level = 2
        self.print_durations_threshold = 0. #np.inf

    def get_pdurations(self):
        return self.print_durations_threshold

    def get_plevel(self):
        return self.print_level


config = Config()

from . import utils       ## utility functions   
from . import data        ## download recombination maps etc
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

