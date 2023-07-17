from typing import Union
import numpy as np

class Connection:
    """ Represents the Connection Gene of the NEAT Algorithm. """
    
    def __init__(
        self, 
        in_id: int, 
        out_id: int, 
        mu: float = 0.0, 
        sigma: float = 0.5,
        weight_min_value: float = -30.0,
        weight_max_value: float = 30.0,
        mut_prob: float = 0.5
    ):
        """
        Args:
            in_id (int): id of the node, from where the connection starts
            out_id (int): id of the node, where the connection ends
            mu (float, optional): expected value of the gaussian distribution. Defaults to 0.0.
            sigma (float, optional): variance of the gaussian distribution. Defaults to 0.5.
            weight_min_value (float, optional): minimum allowed value for the weight. Defaults to -30.0.
            weight_max_value (float, optional): maximum allowed value for the weight. Defaults to 30.0.
            mut_prob (float, optional): probability of changing the weight of the connection. Defaults to 0.5.
        """
        self.in_id = in_id
        self.out_id = out_id
        self.__mu = mu
        self.__sigma = sigma
        self.__weight_min_value = weight_min_value
        self.__weight_max_value = weight_max_value
        self.__mut_prob = mut_prob
        self.weight = np.random.normal(self.__mu, self.__sigma)
        self.enabled = True

    @property
    def id(self) -> tuple[int, int]:
        return self.in_id, self.out_id

    def mutate(self):
        """ Changes the weights of the specific node by a random (gaussian distributed) noise. """
        if np.random.rand() <= self.__mut_prob:
            # Case: Do the mutation
            self.weight += np.random.normal(self.__mu, self.__sigma)

            if self.weight < self.__weight_min_value:
                # Case: Weight is smaller than the minimum allowed value
                self.weight = self.__weight_min_value
            elif self.weight > self.__weight_max_value:
                # Case: weight is bigger than the maximum allowed value
                self.weight = self.__weight_max_value

    def distance(self, other: "Connection") -> float:
        """
        Returns the weight difference between two connections.
        
        Args:
            other (Connection): other connection to compare

        Returns:
            float: weight difference between two connections
        """
        return abs(self.weight - other.weight)

    @staticmethod
    def create(in_id: int, out_id: int, cfg: dict) -> "Connection":
        """
        Create a connection with the given HP from cfg.

        Args:
            in_id (int): id of the node, from where the connection starts
            out_id (int): id of the node, where the connection ends
            cfg (dict): config, which contains HPs
        """
        # Extract HPs from config
        mu = cfg["Connection"]["mu"]
        sigma = cfg["Connection"]["sigma"]
        weight_min_value = cfg["Connection"]["weight_min_value"]
        weight_max_value = cfg["Connection"]["weight_max_value"]
        mut_prob = cfg["Connection"]["mut_prob"]

        return Connection(in_id, out_id, mu, sigma, weight_min_value, weight_max_value, mut_prob)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: Union["Connection", tuple]):
        if isinstance(other, Connection):
            return self.id == other.id
        elif isinstance(other, tuple):
            return self.id == other

    def __str__(self):
        text = f"In: {self.in_id}\nOut: {self.out_id}\nWeight: {self.weight}\nEnabled: {self.enabled}\n"
        return text
    
    def __repr__(self):
        text = f"Connection(in={self.in_id}, out={self.out_id}, weight={self.weight})"
        return self.__str__()

if __name__ == "__main__":
    print(Connection(1, 3))
    print(Connection(2, 3))