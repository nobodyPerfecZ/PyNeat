from neat.activation import Activation
from neat.aggregation import Aggregation
from neat.node import Node, NodeType
from neat.connection import Connection
import networkx as nx
import numpy as np
import copy

class Genome:
    def __init__(
        self,
        id: int,
        n_inputs: int,
        n_outputs: int,
        add_node_prob: float = 0.5, 
        add_connect_prob: float = 0.5,
        change_node_prob: float = 0.5,
        change_connect_prob: float = 0.5,
        # HPs for Node(...)
        node_mu: float = 0.0, 
        node_sigma: float = 0.5,
        bias_min_value: float = -30.0,
        bias_max_value: float = 30.0,
        node_mut_prob: float = 0.5,
        agg_fn: Aggregation = Aggregation.sum_aggregation,
        act_fn: Activation = Activation.sigmoid,
        # HPs for Connection(...)
        connect_mu: float = 0.0, 
        connect_sigma: float = 0.5,
        weight_min_value: float = -30.0,
        weight_max_value: float = 30.0,
        connect_mut_prob: float = 0.5
    ):
        """
        Args:
            id (int): identifier for the genome
            n_inputs (int): number of input nodes
            n_outputs (int): number of output nodes
            add_node_prob (float, optional): probability to make the mutation "add a new node". Defaults to 0.5.
            add_connect_prob (float, optional): probability to make the mutation "add a new connection". Defaults to 0.5.
            change_node_prob (float, optional): probability to make the mutation "change the node bias". Defaults to 0.5.
            change_connect_prob (float, optional): probability to make the mutation "change the connection weight". Defaults to 0.5.
        """
        self.id = id
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.fitness = -np.inf
        self.__add_node_prob = add_node_prob
        self.__add_connect_prob = add_connect_prob
        self.__change_node_prob = change_node_prob
        self.__change_connect_prob = change_connect_prob
        # HPs for Node(...)
        self.__node_mu = node_mu
        self.__node_sigma = node_sigma
        self.__bias_min_value = bias_min_value
        self.__bias_max_value = bias_max_value
        self.__node_mut_prob = node_mut_prob
        self.__agg_fn = agg_fn
        self.__act_fn = act_fn
        # HPs for Connection(...)
        self.__connect_mu = connect_mu
        self.__connect_sigma = connect_sigma
        self.__weight_min_value = weight_min_value
        self.__weight_max_value = weight_max_value
        self.__connect_mut_prob = connect_mut_prob

        # Dictionary where we have key := (in_id, out_id), value := Connection(in_id, out_id)
        self.connect_dict: dict[tuple[int, int], Connection] = dict()
        
        for j in range(n_inputs, n_inputs + n_outputs):
            for i in range(n_inputs):
                self.connect_dict[(i, j)] = Connection(i, j, self.__connect_mu, self.__connect_sigma, self.__weight_min_value, self.__weight_max_value, self.__connect_mut_prob)
        
        # Dictionary where we have key := id, value := Node(id)
        self.node_dict: dict[int, Node] = dict()

        for i in range(n_inputs + n_outputs):
            if i < n_inputs:
                self.node_dict[i] = Node(i, NodeType.SENSOR, self.__node_mu, self.__node_sigma, self.__bias_min_value, self.__bias_max_value, self.__node_mut_prob, self.__agg_fn, self.__act_fn)
            else:
                self.node_dict[i] = Node(i, NodeType.OUTPUT, self.__node_mu, self.__node_sigma, self.__bias_min_value, self.__bias_max_value, self.__node_mut_prob, self.__agg_fn, self.__act_fn)

    @property
    def nodes(self) -> list[Node]:
        return list(self.node_dict.values())

    @property
    def connects(self) -> list[Connection]:
        return list(self.connect_dict.values())
    
    def mutate(self):
        if np.random.rand() <= self.__add_node_prob:
            # Case: Add a new node (between two nodes) to the topology
            self.__add_node()
        if np.random.rand() <= self.__add_connect_prob:
            # Case: Add a new connection to the topology (so that no cycle exists)
            self.__add_connect()
        if np.random.rand() <= self.__change_node_prob:
            # Case: Change the bias of each node
            self.__change_node()
        if np.random.rand() <= self.__change_connect_prob:
            # Case: Change the weights of each node
            self.__change_connect()

    def __add_node(self):
        """ Add a new node (between two nodes) to the topology """
        # Choose a single connection
        connect: Connection = np.random.choice(self.connects)
        in_id, out_id = connect.id
        
        # Disable the single connect
        connect.enabled = False
        
        # Add a new hidden node
        new_id = self.__get_new_node_key()
        node = Node(new_id, NodeType.HIDDEN, self.__node_mu, self.__node_sigma, self.__bias_min_value, self.__bias_max_value, self.__node_mut_prob, self.__agg_fn, self.__act_fn)
        node.bias = 0.0
        self.node_dict[new_id] = node

        # Add two new connections
        connect1 = Connection(in_id, new_id, self.__connect_mu, self.__connect_sigma, self.__weight_min_value, self.__weight_max_value, self.__connect_mut_prob)
        connect2 = Connection(new_id, out_id, self.__connect_mu, self.__connect_sigma, self.__weight_min_value, self.__weight_max_value, self.__connect_mut_prob)
        connect1.weight = 1.0
        connect2.weight = connect.weight
        self.connect_dict[(in_id, new_id)] = connect1
        self.connect_dict[(new_id, out_id)] = connect2

    def __add_connect(self):
        """ Add a new connection (between two nodes) to the topology (without creating cycles) """
        # Create a list of possible in_nodes and out_nodes
        possible_in = []
        possible_out = []
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                # Case: Input node can not be a output node
                possible_in += [node.id]
            elif node.type == NodeType.HIDDEN:
                # Case: Hidden node can be a input or output node
                possible_in += [node.id]
                possible_out += [node.id]
            elif node.type == NodeType.OUTPUT:
                # Case: Output node can not be a input node
                possible_out += [node.id]

        in_id = np.random.choice(possible_in)
        out_id = np.random.choice(possible_out)
        
        if (in_id, out_id) in self.connect_dict:
            # Case: Connection already exists - just enabled it
            self.connect_dict[(in_id, out_id)].enabled = True
            new_created = False
        else:
            # Case: Connection does not exist - Create a new one
            connect = Connection(in_id, out_id, self.__connect_mu, self.__connect_sigma, self.__weight_min_value, self.__weight_max_value, self.__connect_mut_prob)
            self.connect_dict[(in_id, out_id)] = connect
            new_created = True

        # Check if there is a cycle in the the DAG
        G = self._to_graph()
        if len(list(nx.simple_cycles(G))) >= 1:
            # Case: New connection produce a cycle - Remove the new connection
            if new_created:
                del self.connect_dict[(in_id, out_id)]
            else:
                self.connect_dict[(in_id)].enabled = False

    def __change_node(self):
        """ Changes the bias of each node in the genome. """
        for node in self.nodes:
            node.mutate()

    def __change_connect(self):
        """ Changes the weights of each connection in the genome. """
        for connect in self.connects:
            connect.mutate()


    def crossover(self, genome1: "Genome", genome2: "Genome") -> "Genome":
        if genome1.fitness > genome2.fitness:
            # Case: self is better fitted parent
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        for key in parent1.connect_dict.keys():
            if key in parent2.connect_dict:
                # Case: Gene is a matching gene
                connect1 = parent1.connect_dict[key]
                connect2 = parent2.connect_dict[key]
                if (connect1.enabled and connect2.enabled) or ((not connect1.enabled) and (not connect2.enabled)):
                    # Case: Both genes are enabled or disabled - randomly choose which gene to take
                    i = np.random.choice([1, 2])
                    parent = parent1 if i == 1 else parent2
                    connect = parent.connect_dict[key]
                    in_node, out_node = parent.node_dict[key[0]], parent.node_dict[key[1]]
                elif connect1.enabled:
                    # Case: only gene from parent1 is enabled - choose the gene
                    connect = parent1.connect_dict[key]
                    in_node, out_node = parent1.node_dict[key[0]], parent1.node_dict[key[1]]
                elif connect2.enabled:
                    # Case: only gene from parent2 is enabled - choose the gene
                    connect = parent2.connect_dict[key]
                    in_node, out_node = parent2.node_dict[key[0]], parent2.node_dict[key[1]]

                # i = np.random.choice([1, 2])
                # parent = parent1 if i == 1 else parent2
                # connect = parent.connect_dict[key]
                # connect.enabled = True
                # in_node, out_node = parent.node_dict[key[0]], parent.node_dict[key[1]]
            else: 
                # Case: Gene is a disjoint or excess gene
                # Take from the more fitted gene
                connect = parent1.connect_dict[key]
                in_node, out_node = parent1.node_dict[key[0]], parent1.node_dict[key[1]]
            
            # Add connection and nodes to child
            self.connect_dict[key] = copy.deepcopy(connect)
            
            if key[0] not in self.node_dict:
                # Case: in_node not in child
                self.node_dict[key[0]] = copy.deepcopy(in_node)
            if key[1] not in self.node_dict:
                # Case: out_node not in child
                self.node_dict[key[1]] = copy.deepcopy(out_node)
        return self

    def distance(self, other: "Genome") -> float:
        """
        Calculates the distance between two genomes.
        In the NEAT-Paper it is calculated by the following formula:
            lambda_connects = (c1 * #num_excess_genes) / (MAX_GENES) + (c2 * #num_disjoint_genes) / (#MAX_GENES) + (c3 * (#weight_difference(matching_genes)))

        We additionally add the node distance to the formula:
            lambda_node = (n1 * #num_excess_nodes) / (MAX_NODES) + (n2 * #num_disjoint_genes) / (#MAX_NODES) + (n3 * (#bias_difference(matching_nodes)))
            lambda = lambda_connects + lambda_node
        For simplicity we assume c1 = c2 = c3 = n1 = n2 = n3 = 1.0.
        
        Args:
            other (Genome): other genome to compare

        Returns:
            float: difference between two genomes
        """
        # Calculate the node distance
        bias_difference = 0
        disjoint_and_excess_nodes = 0
        for key1 in self.node_dict:
            if key1 not in other.node_dict:
                # Case: node is a disjoint/excess node in first parent
                disjoint_and_excess_nodes += 1
            else:
                # Case: node is a matching node (get the bias difference)
                node1 = self.node_dict[key1]
                node2 = other.node_dict[key1]
                bias_difference += node1.distance(node2)
        for key2 in other.node_dict:
            if key2 not in self.node_dict:
                # Case: connect is a disjoint/excess node in second parent
                disjoint_and_excess_nodes += 1
        n_nodes = max(len(self.nodes), len(other.nodes))
        node_distance = (disjoint_and_excess_nodes + bias_difference) / n_nodes

        # Calculate the connection distance
        weight_difference = 0
        disjoint_and_excess_genes = 0
        for key1 in self.connect_dict:
            if key1 not in other.connect_dict:
                # Case: connect is a disjoint/excess gene in first parent
                disjoint_and_excess_genes += 1
            else:
                # Case: connect is a matching gene (get the weight differences)
                connect1 = self.connect_dict[key1]
                connect2 = other.connect_dict[key1]
                weight_difference += connect1.distance(connect2)
        for key2 in other.connect_dict:
            if key2 not in self.connect_dict:
                # Case: connect is a disjoint/excess gene in second parent
                disjoint_and_excess_genes += 1
        n_connects = max(len(self.connects), len(other.connects))
        connect_distance = (disjoint_and_excess_genes + weight_difference) / n_connects
        return node_distance + connect_distance

    def __get_new_node_key(self) -> int:
        """
        Returns:
            int: next key value (unique index) for a new node
        """
        return max(self.node_dict.keys()) + 1
    
    def _to_graph(self) -> nx.Graph:
        """
        Convert the given genome into a directed graph.

        Returns:
            nx.Graph: directed graph
        """
        # Create an empty directed Graph
        G = nx.DiGraph()
    
        # Create the list of node ids
        nodes = [node.id for node in self.nodes]

        # Create the list of connections
        edges = [connect.id for connect in self.connects if connect.enabled]

        # Add edges and nodes to the graph
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def __node_labels(self) -> dict:
        node_labels = {node.id: round(node.bias, 4) for node in self.nodes}
        return node_labels
    
    def __edge_labels(self) -> dict:
        edge_labels = {connect.id: round(connect.weight, 4) for connect in self.connects if connect.enabled}
        return edge_labels
    
    def __color_map(self) -> dict:
        color_map = dict()
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                color_map[node.id] = "red"
            elif node.type == NodeType.HIDDEN:
                color_map[node.id] = "blue"
            elif node.type == NodeType.OUTPUT:
                color_map[node.id] = "green"
            else:
                raise ValueError("ERROR_GENOME: NodeType is not known!")
        return color_map

    def visualize(
        self, 
        with_bias: bool = False,
        with_weights: bool = False,
        with_colors: bool = False
    ):
        """
        Visualize the given genom as a directed graph.

        Args:
            with_bias (bool): Should the nodes be labeled with the bias?
            with_weights (bool): Should the edges be labeled with the weights?
            with_colors (bool, optional): Should the (...) nodes has the color (...)? Defaults to False.
                - input: red
                - hidden: blue
                - output: green
        """
        G = self._to_graph()
        node_labels = self.__node_labels() if with_bias else None
        edge_labels = self.__edge_labels() if with_weights else None
        color_map = self.__color_map() if with_colors else None
        color_list = [color_map[node] for node in G.nodes()] if color_map else None

        # pos = nx.planar_layout(G)
        pos = nx.shell_layout(G)
        nx.draw(G, pos, node_color=color_list, labels=node_labels)
        
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    @staticmethod
    def create(id: int, n_inputs: int, n_outputs: int, cfg: dict) -> "Genome":
        """
        Creates a genome with the given HP from cfg

        Args:
            id (int): id of the genome
            n_inputs (int): number of input nodes
            n_outputs (int): number of output nodes
            cfg (dict): config, which contains HPs
        """
        # Extract HPs from config
        add_node_prob = cfg["Genome"]["add_node_prob"]
        add_connect_prob = cfg["Genome"]["add_connect_prob"]
        change_node_prob = cfg["Genome"]["change_node_prob"]
        change_connect_prob = cfg["Genome"]["change_connect_prob"]
        node_mu = cfg["Node"]["mu"]
        node_sigma = cfg["Node"]["sigma"]
        bias_min_value = cfg["Node"]["bias_min_value"]
        bias_max_value = cfg["Node"]["bias_max_value"]
        node_mut_prob = cfg["Node"]["mut_prob"]
        agg_fn = Aggregation.to_fn(cfg["Node"]["agg_fn"])
        act_fn = Activation.to_fn(cfg["Node"]["act_fn"])
        connect_mu = cfg["Connection"]["mu"]
        connect_sigma = cfg["Connection"]["sigma"]
        weight_min_value = cfg["Connection"]["weight_min_value"]
        weight_max_value = cfg["Connection"]["weight_max_value"]
        connect_mut_prob = cfg["Connection"]["mut_prob"]

        return Genome(id, n_inputs, n_outputs, add_node_prob, add_connect_prob, change_node_prob, change_connect_prob, node_mu, 
                      node_sigma, bias_min_value, bias_max_value, node_mut_prob, agg_fn, act_fn, connect_mu, connect_sigma, weight_min_value, 
                      weight_max_value, connect_mut_prob)

    def __str__(self) -> str:
        text = "===== Nodes: =====\n"
        for node in self.nodes:
            text += node.__str__()
        text += "===== Connections: =====\n"
        for connect in self.connects:
            text += connect.__str__()
        return text

    def __repr__(self) -> str:
        text = f"Genome(id={self.id}, #nodes={len(self.nodes)}, #connections={len(self.connects)}, fitness={self.fitness})"
        return text
