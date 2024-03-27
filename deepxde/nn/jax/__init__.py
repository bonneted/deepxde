"""Package for jax NN modules."""

__all__ = ["PFNN","FNN", "NN", "SPINN"]

from .fnn import FNN, PFNN, SPINN
from .nn import NN
