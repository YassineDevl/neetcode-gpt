import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean          = torch.mean(x)
                    std           = torch.std(x)
                    dead          = (x <= 0).all(dim=0)
                    dead_fraction = dead.float().mean()
                    stats.append({
                        "mean":          round(mean.item(), 4),
                        "std":           round(std.item(), 4),
                        "dead_fraction": round(dead_fraction.item(), 4)
                    })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        model.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = torch.mean(grad)
                std  = torch.std(grad)
                norm = torch.norm(grad)
                stats.append({
                    "mean": round(mean.item(), 4),
                    "std":  round(std.item(), 4),
                    "norm": round(norm.item(), 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        for stat in activation_stats:
            if stat["std"] < 0.1:
                return "vanishing_gradients"
            if stat["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"