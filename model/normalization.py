import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
    
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        eps = 1e-5
        mu = np.mean(x)
        var = np.var(x)
        x_hat = (x - mu) / np.sqrt(var + eps)
        your_answer = gamma * x_hat + beta
        return np.round(your_answer, 5)
        pass
