import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        N = len(x)
        prediction =0
        # w: 1D weight array (same length as x)
        for i in range(len(x)):
            prediction += x[i] * w[i] 
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        z = prediction + b
        if activation == "sigmoid" :
            result = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            result = np.maximum(0,z)
        
        return np.round(result,5)
    
     

        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        pass
