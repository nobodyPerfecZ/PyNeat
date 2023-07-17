from neat.activation import Activation
from neat.aggregation import Aggregation
from neat.network import FeedForwardNetwork
from neat.species import SpeciesSet
from neat.genome import Genome
import gymnasium as gym
import numpy as np
import copy

class Population:
    
    def __init__(
        self,
        n_genomes: int,
        n_parents: int,
        n_inputs: int, 
        n_outputs: int,
        initial_genome: Genome = None,
        # HPs for SpeciesSet(...),
        compatibility_threshold: float = 3.0,
        # HPs for Genome(...)
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
            n_genomes (int): number of genomes inside the population
            n_parents (int): number of parents to be selected for each generation
            n_inputs (int): number of input nodes
            n_outputs (int): number of output nodes
            initial_genome (Genome, optional): Initial Genome to start with. Defaults to None
            
        """
        assert 1 <= n_parents <= n_genomes, "#ERROR_POPULATION: n_parents should be equal or lower than n_genomes and equal or higher than 1!"
        self.__n_genomes: int = n_genomes
        self.__n_parents: int = n_parents
        self.__n_inputs: int = n_inputs
        self.__n_outputs: int = n_outputs
        # HPs for SpeciesSet(...)
        self.__compatibility_threshold = compatibility_threshold
        # HPs for Genome(...)
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
        if initial_genome is None:
            # Case: Start with fully connected input nodes to the output nodes (without hidden nodes)
            self.population: list[Genome] = [Genome(i, self.__n_inputs, self.__n_outputs, self.__add_node_prob, self.__add_connect_prob, 
                                                    self.__change_node_prob, self.__change_connect_prob, self.__node_mu, self.__node_sigma, 
                                                    self.__bias_min_value, self.__bias_max_value, self.__node_mut_prob, self.__agg_fn, 
                                                    self.__act_fn, self.__connect_mu, self.__connect_sigma, self.__weight_min_value, 
                                                    self.__weight_max_value, self.__connect_mut_prob) for i in range(self.__n_genomes)]
        else:
            # Case: Create n_genome times the initial genome
            self.population: list[Genome] = []
            for i in range(self.__n_genomes):
                genome = copy.deepcopy(initial_genome)
                genome.id = i  # Change the id otherwhise all genomes would have the same id!
                self.population += [genome]
        self.species: SpeciesSet = SpeciesSet(self.__compatibility_threshold)
    
    def run_one_generation(self, generation: int):
        """
        Runs one generation of the Evolutionary algorithm (NEAT) (without evaluating each genome inside the population):
        1. Divide each genome in population into species
        2. Select n_parents genomes from the population with the adjusted fitness value
        3. Do Crossover for each parent pair until n_genomes child genomes are created
        4. Do Mutation for each child genome
        
        Args:
            generation (int): number of generation of the Evolutionary algorithm
        """
        # Divide each genome in population into species
        self.species.speciate(self.population, generation)
        assert len(self.species.genomes) == self.__n_genomes

        # Select n_parents genomes from the population with the adjusted fitness value
        self.species.selection(self.__n_parents)
        assert len(self.species.genomes) == self.__n_parents

        # Get the selected genomes from the species
        self.population = self.species.genomes
            
        childs: list[Genome] = []  # population of the next generation
        while len(childs) < self.__n_genomes:
            # Case: Population is smaller than the given population size
            # Shuffle the population randomly
            np.random.shuffle(self.population)

            # Do pairwise crossover over all parents in the population
            for parent1, parent2 in zip(self.population[0::2], self.population[1::2]):
                if len(childs) == self.__n_genomes:
                    # Case: Enough childs are reproduced
                    # Break while loop
                    break
                # Case: Not enough childs are reproduced 
                # Create a new child genome
                new_id = self.__get_new_genome_key(childs)
                child = Genome(new_id, self.__n_inputs, self.__n_outputs, self.__add_node_prob, self.__add_connect_prob, 
                               self.__change_node_prob, self.__change_connect_prob, self.__node_mu, self.__node_sigma, 
                               self.__bias_min_value, self.__bias_max_value, self.__node_mut_prob, self.__agg_fn, 
                               self.__act_fn, self.__connect_mu, self.__connect_sigma, self.__weight_min_value, 
                               self.__weight_max_value, self.__connect_mut_prob)
                child.crossover(parent1, parent2)
                child.mutate()
                childs.append(child)
        self.population = childs

    def evaluate(self, env: gym.Env, n_episodes: int, max_episode_length: int):
        """ Evaluates each genome on a gym environment. """
        for genome in self.population:
            # Create the model from genome
            model = FeedForwardNetwork.create(genome)
            rewards = []
            for _ in range(n_episodes):
                observation, info = env.reset()
                state = observation
                acc_reward = 0.0
                for _ in range(max_episode_length):
                    # Get the next action
                    q_values = model.forward(state)
                    action = np.argmax(q_values, axis=1).item()

                    # Do a step on the environment
                    observation, reward, terminated, truncated, info = env.step(action)
                    next_state = observation
                    acc_reward += reward
                    
                    # Update the state
                    state = next_state
                    
                    if terminated or truncated:
                        break
                rewards += [acc_reward]
            # Set the fitness of the genome
            genome.fitness = np.mean(rewards)

    def get_best_genome(self) -> Genome:
        """ Returns the best genome (highest fitness) from the population. """
        return max(self.population, key=lambda genome: genome.fitness)

    def get_worst_genome(self) -> Genome:
        """ Returns the worst genome (lowest fitness) from the population. """
        return min(self.population, key=lambda genome: genome.fitness)

    def __get_new_genome_key(self, childs: list[Genome] = None) -> int:
        """
        Args:
            childs (list[Genome], optional): list of genomes to also be considered. Defaults to None.

        Returns:
            int: next id of the new genome
        """
        if childs is not None:
            candidates = self.population + childs
        else:
            candidates = self.population
        return max(candidates, key=lambda genome: genome.id).id + 1

    @staticmethod
    def create(
        n_genomes: int,
        n_parents: int,
        n_inputs: int,
        n_outputs: int,
        cfg: dict,
        initial_genome: Genome = None,
    ) -> "Population":
        """
        Creates a population with the given HP from cfg

        Args:
            n_genomes (int): number of genomes in the population (population length)
            n_parents (int): number of parents to be selected from the population 
            n_inputs (int): number of input nodes for each genome
            n_outputs (int): number of output nodes for each genome
            cfg (dict): config, which contains HPs
            initial_genome (Genome, optional): Initial Genome to start with. Defaults to None
        """
        # Extract HPs from config
        compatibility_threshold = cfg["SpeciesSet"]["compatibility_threshold"]
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

        return Population(n_genomes, n_parents, n_inputs, n_outputs, initial_genome, compatibility_threshold, 
                          add_node_prob, add_connect_prob, change_node_prob, change_connect_prob, node_mu, 
                          node_sigma, bias_min_value, bias_max_value, node_mut_prob, agg_fn, act_fn, connect_mu, 
                          connect_sigma, weight_min_value, weight_max_value, connect_mut_prob)