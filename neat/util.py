import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pickle
import yaml
import os

from neat.network import FeedForwardNetwork
from neat.population import Population
from neat.genome import Genome

class Util:

    @staticmethod
    def train_agent(
        gym_env_name: str,
        max_episodes: int,
        max_episode_length: int,
        n_episodes: int,
        n_genomes: int,
        n_parents: int,
        cfg_path: str,
        intermediate_path: str,
        initial_genome: Genome = None,
    ) -> tuple:
        env = gym.make(gym_env_name)
        config = Util.load_config(cfg_path)
        population: Population = Population.create(n_genomes, n_parents, np.prod(env.observation_space.shape), env.action_space.n, config, initial_genome)
        rewards = []
        for generation in range(max_episodes):
            population.evaluate(env, n_episodes, max_episode_length)
            genome = population.get_best_genome()
            training_path = os.path.join(intermediate_path, f"{generation}_{int(genome.fitness)}")
            Util.save_agent(genome, training_path)
            rewards += [genome.fitness]
            print(f"Generation {generation}, Reward of the best model: {genome.fitness}")
            population.run_one_generation(generation)
            if generation == max_episodes -1:
                population.evaluate(env, n_episodes, max_episode_length)
        return population, rewards
    
    @staticmethod
    def eval_agent(
        gym_env_name: str,
        genome: Genome,
        max_episodes: int,
        max_episode_length: int,
        render_intervall: int
    ) -> list[float]:
        env = gym.make(gym_env_name)
        render_env = gym.make(gym_env_name, render_mode="human")
        model = FeedForwardNetwork.create(genome)
        rewards = []

        for generation in range(max_episodes):
            current_env = render_env if generation % render_intervall == 0 else env
            observation, info = current_env.reset()
            state = observation
            acc_reward = 0.0
            for _ in range(max_episode_length):
                # Get the next action
                q_values = model.forward(state)
                action = np.argmax(q_values, axis=1).item()

                # Do a step on the environment
                observation, reward, terminated, truncated, info = current_env.step(action)
                next_state = observation
                acc_reward += reward
                    
                # Update the state
                state = next_state
                    
                if terminated or truncated:
                    break
            rewards += [acc_reward]
            print(f"Generation {generation}, Reward: {acc_reward}")
        return rewards

    @staticmethod
    def save_agent(genome: Genome, path: str):
        """
        Safes a genome as path.pickle file.

        Args:
            genome (Genome): genome to safe as a file
            path (str): directory of the file (without .pickle ending)
        """
        with open(f"{path}.pickle", "wb") as file:
            pickle.dump(genome, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_agent(path: str) -> Genome:
        """
        Loads a genome from path.pickle file.

        Args:
            path (str): directory of the file (without .pickle ending)

        Returns:
            Genome: genome from the file
        """
        with open(f"{path}.pickle", "rb") as file:
            genome: Genome = pickle.load(file)
            return genome

    @staticmethod
    def load_config(path: str) -> dict:
        """
        Load a .yml config file.

        Args:
            path (str): directory of the config file (without .yml ending)

        Returns:
            dict: config
        """
        return yaml.safe_load(open(f"{path}.yml"))
        
    @staticmethod
    def plot_curve(statistic: list[float], xlabel: str, ylabel: str):
        """
        Plots a statistic curve from the training process.

        Args:
            statistic (list[float]): list of values of shape (?,)
            xlabel (str): label of the x-axis
            ylabel (str): label of the y-axis
        """
        plt.step([i for i in range(len(statistic))], statistic)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()