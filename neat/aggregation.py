from typing import Callable
import numpy as np

class Aggregation:
    @staticmethod
    def sum_aggregation(x: np.ndarray) -> np.ndarray:
        """ Implements the sum aggregation for an iterable of values. """
        return np.sum(x, axis=1)

    @staticmethod
    def to_fn(agg_fn: str) -> Callable:
        """
        Converts the given activation function as string to the real function.

        Args:
            agg_fn (str): aggregation function as str
        """
        if agg_fn == "sum":
            return Aggregation.sum_aggregation
        else:
            raise ValueError(f"#ERROR_AGGREGATION: Given agg_fn '{agg_fn}' is unknown!")