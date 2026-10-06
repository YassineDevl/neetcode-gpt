class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init
        for _ in range(iterations):            
        # Derivative:         f'(x) = 2x
                derivative = 2 * minimizer
                minimizer = minimizer - learning_rate * derivative
        # Round final answer to 5 decimal places
        return round(minimizer, 5)
        pass
