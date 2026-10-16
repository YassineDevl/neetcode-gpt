import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
      
        n = len(x)

        # Forward
        z = 0
        for i in range(n):
            z += x[i] * w[i]
        z = z + b
        y_hat = 1 / (1 + np.exp(-z))

        # Gradient commun (meme pour tous les poids)
        delta = (y_hat - y_true) * y_hat * (1 - y_hat)

        # dL/dw_i = delta * x_i  (un par poids)
        gradientL = np.zeros(n)
        for i in range(n):
            gradientL[i] = delta * x[i]

        # dL/db = delta  (un seul scalaire, pas de boucle)
        gradientb = delta

        return (np.round(gradientL, 5), round(gradientb, 5))
    pass