import numpy as np
from . import Transformer_inplace


class SqrtTransform(Transformer_inplace):
    """
    Square root transformation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _transform(self, X):
        return np.sqrt(X)

