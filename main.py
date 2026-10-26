import time
from genetic import GeneticAlgorithm
from utils import *
from dataset import *
from circuit import *
from fitness import *

def main():

    cfg = Config("config.yaml") #Import config from .yaml file

    X_original, y = generate_circle_dataset(cfg.n_params, cfg.circle_radius, cfg.dataset_noise)
    
    #X_normalized = normalize_X_to_pi(X_original)

    #plot_circle_dataset(X_original, y)

    plot_circle_dataset(X_original, y)
    
    X_train, y_train, X_test, y_test = train_test_split(X_original, y, cfg.train_test_split)

    ga = GeneticAlgorithm(cfg, X_train, y_train)

    # Misura il tempo di training
    start_time = time.time()
    
    ga.train()
    
    end_time = time.time()
    training_time = end_time - start_time

    ga.plot_training_progress()

    print(f"\nTraining completed after {ga.generation -1} generations:")

    if ga.best_individual is not None:
        print(f"\tBest individual fitness: {ga.best_individual.fitness:.6f}")

        train_accuracy = ga.evaluate_accuracy(X_train, y_train)
        print(f"\tTrain accuracy: {train_accuracy:.2%}")
        
        # Evaluate accuracy on test set
        test_accuracy = ga.evaluate_accuracy(X_test, y_test)
        print(f"\tTest accuracy: {test_accuracy:.2%}")
        
        # Calculate and display the generalization gap
        generalization_gap = train_accuracy - test_accuracy
        print(f"\tGeneralization gap: {generalization_gap:.2%}")
        
    else:
        print("\tNo best individual found!")
        

    print(f"\tTotal training time: {training_time:.2f} seconds")
    print(f"\tAverage time per generation: {training_time / max(1, ga.generation - 1):.2f} seconds")


if __name__ == '__main__':
    main()