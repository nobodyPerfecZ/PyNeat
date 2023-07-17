from enum import IntEnum
from typing import Union
import numpy as np

from neat.aggregation import Aggregation
from neat.activation import Activation

class NodeType(IntEnum):
    """
    Enums:
        SENSOR := Input Nodes
        HIDDEN := Hidden Nodes
        OUTPUT := Output Nodes
    """
    SENSOR = 1,
    HIDDEN = 2,
    OUTPUT = 3,
    
    def __str__(self):
        if self == NodeType.SENSOR:
            return "Sensor"
        elif self == NodeType.HIDDEN:
            return "Hidden"
        elif self == NodeType.OUTPUT:
            return "Output"

class Node:
    """ Represents the Node Gene of the NEAT Algorithm. """
    def __init__(
        self, 
        id: int, 
        type: NodeType, 
        mu: float = 0.0, 
        sigma: float = 0.5,
        bias_min_value: float = -30.0,
        bias_max_value: float = 30.0,
        mut_prob: float = 0.5,
        agg_fn: Aggregation = Aggregation.sum_aggregation,
        act_fn: Activation = Activation.sigmoid
    ):
        """
        Args:
            id (int): identifier of the node
            type (NodeType): type of the node (Input)
            mu (float, optional): expected value of the gaussian distribution. Defaults to 0.0.
            sigma (float, optional): variance of the gaussian distribution. Defaults to 0.5.
            bias_min_value (float, optional): minimum allowed value for the bias. Defaults to -30.0.
            bias_max_value (float, optional): maximum allowed value for the bias. Defaults to 30.0.
            mut_prob (float, optional): probability of changing the bias of the node. Defaults to 0.5.
            agg_fn (Aggregation, optional): aggregation function to use. Defaults to Aggregation.sum_aggregation.
            act_fn (Activation, optional): activation function to use. Defaults to Activation.sigmoid.
        """
        self.id = id
        self.type = type
        self.__mu = mu
        self.__sigma = sigma
        self.__bias_min_value = bias_min_value
        self.__bias_max_value = bias_max_value
        self.__mut_prob = mut_prob
        self.bias = np.random.normal(self.__mu, self.__sigma)
        self.agg_fn = agg_fn
        self.act_fn = act_fn

    def mutate(self):
        """ Changes the bias of the specific node by a random (gaussian distributed) noise. """
        if np.random.rand() <= self.__mut_prob:
            # Case: Do the mutation
            self.bias += np.random.normal(self.__mu, self.__sigma)

            if self.bias < self.__bias_min_value:
                # Case: Bias is smaller than the minimum allowed value
                self.bias = self.__bias_min_value
            elif self.bias > self.__bias_max_value:
                # Case: Bias is bigger than the maximum allowed value
                self.bias = self.__bias_max_value

    def distance(self, other: "Node") -> float:
        """
        Returns the bias difference between two nodes.

        Args:
            other (Node): other node to compare

        Returns:
            float: bias difference between two nodes
        """
        return abs(self.bias - other.bias)

    @staticmethod
    def create(id: int, type: NodeType, cfg: dict) -> "Node":
        """
        Create a node with the given HP from cfg.

        Args:
            id (int): identifier of the node
            type (NodeType): type of the node
            cfg (dict): config, which contains HPs
        """
        # Extract HPs from config
        mu = cfg["Node"]["mu"]
        sigma = cfg["Node"]["sigma"]
        bias_min_value = cfg["Node"]["bias_min_value"]
        bias_max_value = cfg["Node"]["bias_max_value"]
        mut_prob = cfg["Node"]["mut_prob"]
        agg_fn = Aggregation.to_fn(cfg["Node"]["agg_fn"])
        act_fn = Activation.to_fn(cfg["Node"]["act_fn"])

        return Node(id, type, mu, sigma, bias_min_value, bias_max_value, mut_prob, agg_fn, act_fn)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: Union["Node", int]):
        if isinstance(other, Node):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other

    def __str__(self) -> str:
        text = f"Node {self.id}\n{str(self.type)}\nBias: {self.bias}\n"
        return text

    def __repr__(self) -> str:
        text = f"Node(id={self.id}, type={self.type}), bias={self.bias}"
        return text