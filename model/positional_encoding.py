import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        PE  = np.zeros((seq_len, d_model))
        pos = np.arange(seq_len).reshape(-1, 1)
        i   = np.arange(0, d_model, 2)
        div = np.power(10000, i / d_model)

        PE[:, 0::2] = np.sin(pos / div)
        PE[:, 1::2] = np.cos(pos / div)

        return np.round(PE, 5)