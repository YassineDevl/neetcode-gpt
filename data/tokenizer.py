from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            # Compter les paires
            counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                counts[pair] = counts.get(pair, 0) + 1

            # Meilleure paire — max fréquence, puis plus petite lexicographiquement
            max_count = max(counts.values())
            best = min(
                [p for p in counts if counts[p] == max_count],
                key=lambda p: p
            )
            merges.append(list(best))

            # Remplacer les occurrences
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == best[0] and tokens[i+1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges