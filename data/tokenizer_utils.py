from typing import List, Dict

class Solution:

    def greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            match = None
            for length in range(len(text)-i, 0, -1):
                substring = text[i:i+length]
                if substring in vocab:
                    match = substring
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self.greedy_tokenize(str(n), vocab) for n in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self.greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        token_count = self.count_tokens(text, vocab)
        word_count  = len(text.split())
        return round(token_count / word_count, 4)