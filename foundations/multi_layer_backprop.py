import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
      
        # Conversion en arrays numpy
        x      = np.array(x)
        W1, b1 = np.array(W1), np.array(b1)
        W2, b2 = np.array(W2), np.array(b2)
        y_true = np.array(y_true)

        # ── FORWARD ──────────────────────────────────
        z1 = x @ W1.T + b1          # [hidden]
        a1 = np.maximum(0, z1)      # [hidden]  ReLU
        z2 = a1 @ W2.T + b2         # [output]

        # Loss MSE
        loss = np.mean((z2 - y_true) ** 2)

        # ── BACKWARD ─────────────────────────────────
        # 1. Gradient de la loss vers z2
        n   = len(y_true)
        dz2 = (2 / n) * (z2 - y_true)          # [output]

        # 2. Gradients W2, b2
        dW2 = np.outer(dz2, a1)                 # [output x hidden]
        db2 = dz2                               # [output]

        # 3. Remonter a travers W2
        da1 = dz2 @ W2                          # [hidden]

        # 4. Masque ReLU — tue le gradient la ou z1 <= 0
        dz1 = da1 * (z1 > 0)                   # [hidden]

        # 5. Gradients W1, b1
        dW1 = np.outer(dz1, x)                 # [hidden x input]
        db1 = dz1                               # [hidden]

        return {
            'loss': round(float(loss), 4),
            'dW1':  np.round(dW1, 4).tolist(),
            'db1':  np.round(db1, 4).tolist(),
            'dW2':  np.round(dW2, 4).tolist(),
            'db2':  np.round(db2, 4).tolist(),
        }