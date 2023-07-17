from typing import Optional
import numpy as np

from neat.genome import Genome

class Species:
    def __init__(self, id: int):
        self.id = id
        self.representative_id = None
        # dictionary that contains each Genome inside the species (key := genome_id, value := Genome(...))
        self.members: dict[int, Genome] = dict()

    @property
    def representative(self) -> Optional[Genome]:
        if self.representative_id is not None:
            # Case: Representative_id is given
            return self.members[self.representative_id]
        else:
            # Case: Representative_id is not given
            return None
    
    @property
    def genomes(self) -> list[Genome]:
        return list(self.members.values())

    def update(self, representative_id: int,  members: dict[int, Genome]):
        """
        Update the species with the new generation and its corresponding representative

        Args:
            representative (int): genome id which represents the species
            members (dict[int, Genome]): members of the species
        """
        assert representative_id in members, "#ERROR_SPECIES: Id of representative is not in members!"
        self.representative_id = representative_id
        self.members = members

    def add(self, genome: Genome):
        """ Add a new (or maybe old) genome to the species """
        self.members[genome.id] = genome

    def remove(self):
        """ Remove all genomes expect from the representative """
        assert self.representative_id is not None, "#ERROR_SPECIES: Expect that representative is in members!"
        for genome in self.genomes:
            if self.representative_id != genome.id:
                # Case: Genome is not representative
                # Remove it!
                del self.members[genome.id]

    def get_adjusted_fitnesses(self) -> dict[int, float]:
        """
        Returns:
            dict[int, float]: dict of adjusted fitness, key:= genome id, value:= adjusted fitness (f_i / #genomes) of each member in the species
        """
        # return {
        #     genome.id: genome.fitness / len(self) for genome in self.genomes
        # }
        # Shift negative fitness values by the minimum value to get no negative fitness values at all
        min_fitness = abs(min(0.0, min(self.genomes, key=lambda genome: genome.fitness).fitness))
        return {
            genome.id: (genome.fitness + min_fitness) / len(self) for genome in self.genomes
        }

    def __len__(self) -> int:
        return len(self.genomes)

    def __str__(self) -> str:
        text = "===== Genomes: =====\n"
        for genome in self.genomes:
            text += f"{genome.__repr__()}\n"
        return text

    def __repr__(self) -> str:
        text = f"Species(id={self.id}, representative_id={self.representative_id}, #genomes={len(self.genomes)})"
        return text

class SpeciesSet:
    def __init__(self, compatibility_threshold: float = 3.0):
        """
        Args:
            compatibility_threshold (float, optional): Distance Threshold to decide, when to create a new species. Defaults to 3.0.
        """
        self.generation = 0
        self.compatibility_threshold = compatibility_threshold
        # Dictionary that contains each species (key := species_id, value := Species(...))
        self.specie_dict: dict[int, Species] = dict()

    @property
    def species(self) -> list[Species]:
        return list(self.specie_dict.values())
    
    @property
    def genomes(self) -> list[Genome]:
        genomes = []
        for specie in self.species:
            genomes += specie.genomes
        return genomes

    def speciate(self, population: list[Genome], generation: int):
        assert len(population) >= 1, "#ERROR_SPECIES: Length of population should be higher or equal to 1!"
        self.__update_generation(generation)
        possible_members = population
        new_species_keys = []  # necessary to detect which species are "old" and which species are new

        if self.generation == 0:
            # Case: First Generation - Yet we have no species
            # Create a new Species for the genome 0 in the population
            possible_members = population[1:]
            new_key = self.__get_new_species_key()
            new_species_keys += [new_key]
            species = Species(new_key)
            species.update(representative_id=population[0].id, members={population[0].id: population[0]})
            self.add(species)
        else:
            # Case: Species from last generation are available
            # Choose a random representative from each specie
            for specie in self.species:
                representative: Genome = np.random.choice(specie.genomes)
                specie.representative_id = representative.id
                specie.remove()  # Delete each genome from the last generation (except representative)

        # Check for each genome the distance and their corresponding specie
        for genome in possible_members:
            specie_founded = False
            for specie in self.species:
                # distance = self.distances(genome, specie.representative)
                distance = genome.distance(specie.representative)
                if distance <= self.compatibility_threshold:
                    # Case: Add genome to the specie
                    specie_founded = True
                    specie.add(genome)
                    break

            if not specie_founded:
                # Case: Create a new specie for the given genome
                new_key = self.__get_new_species_key()
                new_species_keys += [new_key]
                specie = Species(new_key)
                specie.update(representative_id=genome.id, members={genome.id: genome})
                self.add(specie) 

        # Delete the representatives from each specie
        for specie in self.species:
            if specie.id in new_species_keys:
                # Case: New species
                # Do not delete representative from that
                specie.representative_id = None
            elif len(specie) == 1:
                # Case: "Old" specie has only the representative
                # So therefore no new genome was added
                # Remove the specie
                del self.specie_dict[specie.id]
            else:
                # Case: "Old" specie has more than the representative
                # So therefore new genomes was added
                # Remove the representative
                specie = self.specie_dict[specie.id]
                del specie.members[specie.representative_id]
                specie.representative_id = None

    def selection(self, n_parents: int):
        # assert 1 <= n_parents <= sum([len(specie) for specie in self.species]), "#ERROR_SPECIES: n_parents should be higher or equal to 1 and lower or equal to the number of parents in the generation!"
        # Get the fitness values for each genome in each specie
        adjusted_fitnesses = self.get_adjusted_fitnesses()
        total_fitness = sum(adjusted_fitnesses.values())  # total fitness over the entire population
        prob_dist = [fitness / total_fitness for fitness in list(adjusted_fitnesses.values())]
        selected_genomes = np.random.choice(list(adjusted_fitnesses.keys()), size=n_parents, replace=False, p=prob_dist)

        # Remove all genomes that are not in selected_genomes
        for (specie_key, specie) in self.specie_dict.items():
            for genome in specie.genomes:
                if genome.id not in selected_genomes:
                    if len(specie) == 1:
                        # Case: Specie contains only the single genome
                        # Remove the specie
                        del self.specie_dict[specie_key]
                        break
                    else:
                        # Case: Specie contains more than one genome
                        # Remove the genome from the specie
                        del self.specie_dict[specie_key].members[genome.id]

    def add(self, species: Species):
        """ Add a new species to the dict of species. """
        self.specie_dict[species.id] = species

    def get_adjusted_fitnesses(self) -> dict[int, float]:
        """
        Returns:
            dict[int, float]: dictionary of each genome in each specie
        """
        result = {}
        for specie in self.species:
            result.update(specie.get_adjusted_fitnesses())
        return result

    def __get_new_species_key(self) -> int:
        """
        Returns:
            int: next key value (unique index) for a new species
        """
        if self.specie_dict:
            # Case: Species are available
            return max(self.specie_dict.keys()) + 1
        else:
            # Case: No species been created
            return 0

    def __update_generation(self, generation: int):
        """ Updates the generation to the given value. """
        self.generation = generation
    
    @staticmethod
    def create(cfg: dict) -> "SpeciesSet":
        """
        Creates a SpeciesSet with the given HP from cfg

        Args:
            cfg (dict): config, which contains HPs
        """
        # Extract HPs from config
        compatibility_threshold = cfg["SpeciesSet"]["compatibility_threshold"]
        return SpeciesSet(compatibility_threshold)
    
    def __str__(self) -> str:
        text = "===== Species: =====\n"
        for specie in self.species:
            text += f"{specie.__repr__()}\n"
        return text
    
    def __repr__(self) -> str:
        text = f"Generation {self.generation}: SpecieSet(compatibility_threshold={self.compatibility_threshold}, #species={len(self.species)})"
        return text
