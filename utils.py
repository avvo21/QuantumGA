import yaml
import numpy as np

class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Genetic Algorithm
        self.population_size = data["population_size"]
        self.num_generations = data["num_generations"]
        self.n_children = data["n_children"]


        # Mutation
        self.p_mut_init = data["p_mut_init"]
        self.mutation_decay = data["mutation_decay"]
        self.mutation_type = data["mutation_type"]
        self.mutation_sigma = data["mutation_sigma"]
        self.exponential_weight = data["exponential_weight"]
        self.selection_type = data["selection_type"]
        self.use_steady_state = data["use_steady_state"]

        # Crossover

        # Circuit
        self.theta_dimension = data["theta_dimension"]
        self.theta_range = data["theta_range"]
        self.n_shots = data["n_shots"]

        # Dataset
        self.n_params = data["n_params"]
        self.circle_radius= data["circle_radius"]
        self.dataset_noise= data["dataset_noise"]
        self.train_test_split = data["train_test_split"]

class Individual:

    def __init__(self, theta):
        # Initializes the parameter vector and fitness
        self.theta = theta  # real-valued vector
        self.fitness: float = 0.0  # to be calculated, initialized to 0.0

    def clone(self):
        cloned = Individual(np.copy(self.theta))
        cloned.fitness = self.fitness if self.fitness is not None else 0.0
        return cloned

    def mutate(self, p_mut, mutation_type, sigma=None):
        for i in range(len(self.theta)):
            r = np.random.uniform()
            if r < p_mut:
                if mutation_type == "gaussian":
                    # Gaussian mutation with specified sigma
                    if sigma is None:
                        raise ValueError("sigma must be provided for gaussian mutation")
                    self.theta[i] = (self.theta[i] + np.random.normal(loc=0, scale=sigma)) % (2 * np.pi)
                elif mutation_type == "random":
                    # Random uniform mutation
                    self.theta[i] = np.random.uniform(0, 2 * np.pi)
                else:
                    raise ValueError("mutation_type must be 'gaussian' or 'random'")

def generate_binary_mask(n: int) -> np.ndarray:
    """Generate a random binary mask with equal number of 0s and 1s when possible"""

    # Calculate number of 1s and 0s
    ones_count = n // 2
    zeros_count = n - ones_count

    # Create array with the calculated number of 0s and 1s
    mask = np.concatenate([
        np.zeros(zeros_count, dtype=int),
        np.ones(ones_count, dtype=int)
    ])

    # Shuffle randomly
    np.random.shuffle(mask)

    return mask