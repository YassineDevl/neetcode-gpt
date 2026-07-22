import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x            = np.array(x)
        gamma        = np.array(gamma)
        beta         = np.array(beta)
        running_mean = np.array(running_mean)
        running_var  = np.array(running_var)

        if training:
            mu  = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            running_mean = (1 - momentum) * running_mean + momentum * mu
            running_var  = (1 - momentum) * running_var  + momentum * var
        else:
            mu  = running_mean
            var = running_var

        x_hat = (x - mu) / np.sqrt(var + eps)
        y     = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )