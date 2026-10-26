from dill.pointers import parent

from circuit import measurement_function
from utils import *
from fitness import cross_entropy_loss
from joblib import Parallel, delayed
import numpy as np
import  matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, config: Config, X: np.ndarray, y: np.ndarray) -> None:

        self.X = X
        self.y = y
        self.config = config
        self.population_size= self.config.population_size
        self.theta_dimension= self.config.theta_dimension
        self.theta_range= tuple(self.config.theta_range)
        self.population = []
        self.p_mut = self.config.p_mut_init
        self.decay = self.config.mutation_decay
        self.generation = 0
        self.best_individual = None

        # Data Tracking for final graph
        self.fitness_history = []
        self.accuracy_history = []
        self.generation_history = []


    def initialize_population(self) -> None:
        for _ in range(self.population_size):
            theta = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1], size=self.theta_dimension)
            individual = Individual(theta)
            self.population.append(individual)


    def compute_fitness(self):
        """Compute fitness for all individuals of the population"""

        # Parallel execution with joblib
        losses = Parallel(n_jobs=-1)(  # Use all available cores
            delayed(cross_entropy_loss)(ind, self.X, self.y, self.config.n_shots)
            for ind in self.population
        )

        # Update fitness field for each individual
        for ind, loss in zip(self.population, losses):
            if loss is None:
                print(f"Warning: cross_entropy_loss returned None for an individual")
                loss = 10.0  # Default high loss value
            ind.fitness = float(loss)

        self.population.sort(key=lambda ind: ind.fitness, reverse=False)

        # Best individual is now at index 0
        if self.best_individual is None:
            self.best_individual = self.population[0]
        elif self.best_individual.fitness > self.population[0].fitness:
            self.best_individual = self.population[0]


    def child_mutation(self, children: list[Individual]) -> list[Individual]:

        # Apply mutation to each child
        for child in children:
            child.mutate(self.p_mut, self.config.mutation_type, self.config.mutation_sigma)

        # Calculate fitness for each child after mutation
        for child in children:
            loss = cross_entropy_loss(child, self.X, self.y, self.config.n_shots)
            if loss is None:
                print(f"Warning: cross_entropy_loss returned None for a child")
                loss = 10.0  # Default high loss value
            child.fitness = float(loss)  # Explicit cast to float

        self.p_mut = self.p_mut * self.decay

        return children

    def select_parents_rank_based(self) -> tuple[
        Individual, Individual]:
        """Select two parents using rank-based selection"""
        n = len(self.population)

        # Population is already sorted in ascending order (best fitness = lowest value at index 0)

        # Vector of ranks (n, n-1, ..., 2, 1)
        rank = np.arange(n, 0, -1)  # [n, n-1, n-2, ..., 2, 1]

        if self.config.selection_type == "linear":
            prob = rank / (n * (n - 1))

        elif self.config.selection_type == "exponential":
            if not (0 < self.config.exponential_weight < 1):
                raise ValueError("Weight w must be between 0 and 1 for exponential selection")

            # Calculate w^(n-rank) for each rank
            exponents = n - rank
            prob = np.power(self.config.exponential_weight, exponents)

        else:
            raise ValueError("selection_type must be 'position', 'linear', or 'exponential'")

        pSum = np.sum(prob)

        # Normalize probabilities
        prob = prob / pSum

        fitIndx = np.arange(n)  # [0, 1, 2, ..., n-1] (already sorted, best at 0)

        pCumSum = np.cumsum(prob)

        # Select two different parents
        selected_indices = []
        for _ in range(2):
            rnd = np.random.rand()

            idx = np.argmax(rnd <= pCumSum)  # Find first index where condition is true

            selected_index = fitIndx[idx]
            selected_indices.append(selected_index)

        # Ensure we select two different parents
        if selected_indices[0] == selected_indices[1] and n > 1:

            # If same parent selected twice, choose a different second parent
            available_indices = [i for i in range(n) if i != selected_indices[0]]

            # Use same probability distribution but exclude the first selected parent
            remaining_prob = np.delete(prob, selected_indices[0])
            remaining_prob = remaining_prob / np.sum(remaining_prob)
            remaining_cumsum = np.cumsum(remaining_prob)

            rnd = np.random.rand()
            idx = np.argmax(rnd <= remaining_cumsum)
            selected_indices[1] = available_indices[idx]

        parent1 = self.population[selected_indices[0]]
        parent2 = self.population[selected_indices[1]]

        return parent1, parent2

    def scattered_crossover(self, n_children: int) -> list[Individual]:
        """Scattered crossover on the population for generate children"""
        children = []
        for _ in range(n_children):

            parent1, parent2 = self.select_parents_rank_based()

            mask = generate_binary_mask(self.config.theta_dimension)

            child = Individual(np.zeros(self.config.theta_dimension))

            for i in range(len(mask)):
                if mask[i]:
                    child.theta[i] = parent1.theta[i]
                else:
                    child.theta[i] = parent2.theta[i]

            children.append(child)

        return children


    def update_population_steady_state(self, children: list[Individual]):
        """Steady-state selection: combine parents and children, select best individuals"""
        
        # Combine current population with children
        combined_population = self.population + children
        
        # Sort by fitness (best first - lowest fitness value)
        combined_population.sort(key=lambda ind: ind.fitness, reverse=False)
        
        # Select the best individuals to form the new population
        self.population = combined_population[:self.population_size]
        
        # Update best individual
        if self.best_individual is None:
            self.best_individual = self.population[0]
        elif self.best_individual.fitness > self.population[0].fitness:
            self.best_individual = self.population[0]


    def update_population(self, children: list[Individual]):
        """Replace random individuals with children, but preserve best individual and avoid children replacing each other"""

        # Create a list of available indices (excluding the best individual at index 0)
        available_indices = list(range(0, len(self.population)))

        # Shuffle the indices to randomize selection
        np.random.shuffle(available_indices)

        # Replace individuals using the shuffled indices
        for i, child in enumerate(children):
            if i < len(available_indices):  # Ensure we don't exceed available positions
                self.population[available_indices[i]] = child

        # Re-sort population from best to worst fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=False)
        if self.best_individual is None:
            self.best_individual = self.population[0]
        elif self.best_individual.fitness > self.population[0].fitness:
            self.best_individual = self.population[0]


    def evaluate_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the accuracy of the best individual on the test set
        """
        if self.best_individual is None:
            return 0.0

        # Parallel execution with joblib
        correct_predictions_list = Parallel(n_jobs=-1)(
            delayed(evaluate_single_sample)(i, X_test, y_test, self.best_individual, self.config.n_shots)
            for i in range(len(X_test))
        )

        # Convert to list to handle joblib return type
        predictions_list = list(correct_predictions_list)
        correct_predictions = sum(pred for pred in predictions_list if pred is not None)
        accuracy = float(correct_predictions / len(X_test))
        return accuracy


    def train(self) -> None:
        print("Training genetic algorithm started.")

        if self.generation == 0:
            print("Initializing population...\n")
            self.initialize_population()
            self.compute_fitness()
            self.generation += 1

        while self.generation <= self.config.num_generations:
            if self.best_individual is not None:

                current_accuracy = self.evaluate_accuracy(self.X, self.y)

                print(f"Generation {self.generation} -> \tBest Fitness: {self.best_individual.fitness:.6f}\t Accuracy: {(current_accuracy *100):.2f}%")

                self.fitness_history.append(self.best_individual.fitness)
                self.accuracy_history.append(current_accuracy * 100)
                self.generation_history.append(self.generation)
            else:
                print(f"Generation {self.generation} -> \tBest Fitness: N/A")

            children = self.scattered_crossover(self.config.n_children)
            children = self.child_mutation(children)

            # Use steady-state or generational replacement based on config
            if self.config.use_steady_state:
                self.update_population_steady_state(children)
            else:
                self.update_population(children)
            #self.evolution_random_replacement(self.config.n_children)

            self.generation += 1

    def plot_training_progress(self) -> None:
        """Plot fitness and accuracy evolution during training"""
        if not self.fitness_history:
            print("No data to plot")
            return

        fig = plt.figure(figsize=(12, 10))
        
        # Create grid layout: 2 plots on top, text below
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.4], hspace=0.4)

        # fitness graph
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.generation_history, self.fitness_history, 'b-', linewidth=2, label='Best Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (Cross-Entropy Loss)')
        ax1.set_title('Fitness Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # accuracy graph
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.generation_history, self.accuracy_history, 'r-', linewidth=2, label='Accuracy')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 100)

        # Configuration parameters text
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')  # Hide axes
        
        # Determine selection method description
        selection_method = "Steady-State Selection" if self.config.use_steady_state else "Rank Selection"
        
        # Build configuration text based on mutation type
        config_text = (
            f"Configuration Parameters:\n"
            f"Population Size: {self.config.population_size}  |  "
            f"Children per Generation: {self.config.n_children}  |  "
            f"Initial Mutation Probability: {self.config.p_mut_init}\n"
            f"Mutation Decay: {self.config.mutation_decay}  |  "
            f"Mutation Type: {self.config.mutation_type}"
        )
        
        # Add mutation sigma only for gaussian mutation
        if self.config.mutation_type == "gaussian":
            config_text += f"  |  Mutation Sigma: {self.config.mutation_sigma}"
        
        config_text += (
            f"\nExponential Weight: {self.config.exponential_weight}  |  "
            f"Selection Type: {self.config.selection_type}  |  "
            f"Population Update: {selection_method}  |  "
            f"Theta Dimension: {self.config.theta_dimension}"
        )
        
        ax3.text(0.5, 0.5, config_text, transform=ax3.transAxes, 
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.show()

        # Saving graph option
        #plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        #print("Training progress plot saved as 'training_progress.png'")


def evaluate_single_sample(i: int, X_test: np.ndarray, y_test: np.ndarray, best_individual: Individual, n_shots: int) -> int:
    """Function to evaluate a single sample for accuracy calculation"""
    # Get probabilities for both classes
    p_0, p_1 = measurement_function(X_test[i], best_individual, n_shots)

    # Predict class with higher probability
    predicted_class = 1 if p_1 > p_0 else 0

    return 1 if predicted_class == y_test[i] else 0