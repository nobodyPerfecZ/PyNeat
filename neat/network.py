
import numpy as np
from neat.aggregation import Aggregation
from neat.connection import Connection
from neat.genome import Genome
from neat.node import Node, NodeType


class FeedForwardNetwork:
    def __init__(self, inputs: list[int], outputs: list[int], node_evals: list[tuple]):
        # list of node ids from the input nodes
        self.input_nodes = inputs
        # list of node ids from the output nodes
        self.output_nodes = outputs
        # order to evaluate the nodes from the NN, each entry := (out_node_id, act_fn, agg_fn, bias, [(in_node_id, weight)])
        self.node_evals = node_evals
        # to safe all values of each node
        self.values = dict((node_id, 0.0) for node_id in inputs + outputs)
        self.values.update(dict((node_id[0], 0.0) for node_id in node_evals))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass of the FeedForwardNetwork.

        Args:
            inputs (np.ndarray): array of shape (?, ?)
        
        Returns:
            np.ndarray: result of shape (?, ?)
        """
        # assert len(self.input_nodes) == len(inputs), "#ERROR_NETWORK: inputs should have the same length as input nodes!"
        if inputs.ndim == 1:
            # Case: Dimension of inputs is (N,) -> Reshape to (1, N)
            inputs = inputs.reshape((1, len(inputs)))

        for k in self.input_nodes:
            self.values[k] = inputs[:, k]
        for out_id, act_fn, agg_fn, bias, connects in self.node_evals:
            node_inputs = []
            for in_id, weights in connects:
                node_inputs.append(self.values[in_id] * weights)
            s = agg_fn(np.stack(node_inputs, axis=1))
            self.values[out_id] = act_fn(bias + s)
        outputs = [self.values[i] for i in self.output_nodes]
        return np.stack(outputs, axis=1)

    @staticmethod
    def get_layers(first_inputs: list[int], connects: list[tuple[int, int]]):
        layers = []
        visited = set(first_inputs)
        while True:
            # Find candidate nodes c for the next layer. These nodes should connect a node in s to a node not in s
            # Get all child nodes
            c = set(b for (a, b) in connects if a in visited and b not in visited)
            # Keep only the used nodes whose entire input is contained in s
            t = set()
            for n in c:
                if all(a in visited for (a, b) in connects if b == n):
                    t.add(n)
            if not t:
                break
            layers.append(t)
            visited = visited.union(t)
        return layers

    @staticmethod
    def create(genome: Genome) -> "FeedForwardNetwork":
        connect_ids: list[tuple[int, int]] = [connect.id for connect in genome.connects if connect.enabled]
        first_input_ids: list[int] = []  # store the ids of the input nodes
        output_ids: list[int] = []  # store the ids of the output nodes
        for node in genome.nodes:
            if node.type == NodeType.SENSOR:
                first_input_ids += [node.id]
            elif node.type == NodeType.OUTPUT:
                output_ids += [node.id]
        layers = FeedForwardNetwork.get_layers(first_input_ids, connect_ids)
        # Order of evaluating the nodes in a neural network, each entry := (out_node_id, act_fn, agg_fn, bias, [(in_node_id, connection.weight)])
        node_evals = []
        for layer in layers:
            for node_id in layer:
                inputs = []  # remember: [(in_node_id, connection.weight)]
                for (in_id, out_id) in connect_ids:
                    if out_id == node_id:
                        # Case: Found a connection that points to the given node
                        connect: Connection = genome.connect_dict[(in_id, out_id)]
                        inputs.append((in_id, connect.weight))
                node: Node = genome.nodes[node_id]
                node_evals.append((node_id, node.act_fn, node.agg_fn, node.bias, inputs))
        return FeedForwardNetwork(first_input_ids, output_ids, node_evals)