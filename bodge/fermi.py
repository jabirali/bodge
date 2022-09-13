import numpy as np

from .hamiltonian import Hamiltonian
from .math import *
from .typing import *


class Fermi:
    # - Iterate only over f_n != 0
    # - Generate identity blocks I_k
    # - Perform (parallelizable) expansion of F_nk = [f_n T_n(X)]_k
    # - Add g_n factors...
    # for f_n, T_n in zip()
    pass
