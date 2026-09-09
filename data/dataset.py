import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        starts = torch.randint(0, len(tokens) - context_length, (batch_size,))
        X = [tokens[s : s + context_length] for s in starts]
        Y = [tokens[s+1 : s+1 + context_length] for s in starts]
        return X, Y