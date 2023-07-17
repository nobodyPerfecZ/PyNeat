from typing import Callable
import numpy as np

class Activation:
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid activation function. """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ Implements the relu activation function. """
        return np.maximum(0.0, z)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """ Implements the tanh activation function. """
        return np.tanh(z)
    
    @staticmethod
    def to_fn(act_fn: str) -> Callable:
        """
        Converts the given activation function as string to the real function.

        Args:
            act_fn (str): activation function as str
        """
        if act_fn == "sigmoid":
            return Activation.sigmoid
        elif act_fn == "relu":
            return Activation.relu
        elif act_fn == "tanh":
            return Activation.tanh
        else:
            raise ValueError(f"#ERROR_ACTIVATION: Given act_fn '{act_fn}' is unknown!")