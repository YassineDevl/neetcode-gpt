import numpy as np
from numpy.typing import NDArray


class Solution:

  def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.max(z)
    s = 0
    for i in range(len(z)):
        s += np.exp(z[i] - m)
    L = []
    for i in range(len(z)):
        L.append(np.exp(z[i] - m) / s)
    return np.round(L, 4)