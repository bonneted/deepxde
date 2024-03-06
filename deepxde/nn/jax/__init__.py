"""Package for jax NN modules."""

__all__ = ["SPINN","PFNN","FNN", "NN"]

from .fnn import FNN, PFNN, SPINN
from .nn import NN
