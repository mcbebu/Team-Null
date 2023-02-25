
# General use libraries 
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd 


from datetime import datetime
from itertools import permutations
from pprint import pprint 

# EA libraries 
from deap import base, creator, tools, algorithms

# -----------------------------------
# Generate synthetic data
def generate_data(size: int = 20): 
    points = [[random.randint(0, 50), random.randint(0, 50)] for i in range(size)]
    
    # generate a lookup from them
    perm = [pair for pair in permutations(points, 2)]
    lookup = []
    for pair in perm:
        lookup.append([pair[0], pair[1], np.sqrt((pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2)])
        # print(pair[0], pair[1])
    return points, lookup



# -----------------------------------
# EA functions using DEAP 

def tsp_solver(locations: list, population_size: int=100, num_generations: int=1000, cxpb: float =0.5, mutpb: float =0.2):
    """ Solves the TSP problem using a genetic algorithm. Uses OOTB algorithms from the DEAP package. 

    INPUTS
    ----------
    locations
        This is a list of (lat, long) coordinates that represent the locations that the driver must visit.

    population_size
        This parameter determines the size of the population used in the genetic algorithm. 
        A larger population size can improve the diversity of the population and increase the chances of finding better solutions, 
        but also requires more computational resources. In general, a good rule of thumb is to set the population size to at least 10 times the number of variables 
        (in this case, the number of locations in the TSP). However, you may need to experiment with different population sizes to find the optimal value for your problem.

    num_generations
        This parameter determines the number of generations (iterations) that the genetic algorithm will run for. 
        A larger number of generations can improve the chances of finding better solutions, but also requires more computational resources. 
        In general, you should run the genetic algorithm for as many generations as necessary to achieve good results, 
        but not so many that it takes an unreasonable amount of time. 
    

    cxpb
        This parameter determines the probability of crossover (recombination) between two individuals in the population. 
        A higher value of cxpb means that crossover is more likely to occur and can improve the diversity of the population, 
        but may also result in premature convergence to suboptimal solutions. A lower value of cxpb means that crossover is 
        less likely to occur and can result in slower convergence but may allow for a more thorough exploration of the search space. 
        In general, a good starting value for cxpb is 0.5, but you may need to experiment with different values to find the optimal value for your problem.

    mutpb
        This parameter determines the probability of mutation of an individual in the population. 
        A higher value of mutpb means that mutation is more likely to occur and can allow for more exploration of the search space, 
        but may also result in premature convergence to suboptimal solutions. 
        A lower value of mutpb means that mutation is less likely to occur and can result in slower convergence 
        but may lead to a more thorough exploration of the search space. 
        In general, a good starting value for mutpb is 0.2, but you may need to experiment with different values to find the optimal value for your problem.

    
    OUTPUTS
    ----------
    pop
        This is the final population of individuals that resulted from running the genetic algorithm. 
        It is a list of individuals, where each individual is a list of integers representing a candidate solution for the TSP problem. The length of the population is determined by the population_size parameter.

    logbook
        This is a record of the statistics collected during the genetic algorithm. 
        It is a deap.tools.Logbook object, which is essentially a dictionary containing the statistics for each generation. 
        The keys of the dictionary are the names of the statistics, and the values are lists of the statistics for each generation.
        ref: https://deap.readthedocs.io/en/master/api/tools.html?highlight=logbook%20object#logbook

    hof
        This is the Hall of Fame object, which is a list of the best individuals found during the run of the genetic algorithm. 
        By default, hof has a length of 0, but you can specify a different size when creating the HallOfFame object. 
        Each individual in hof is a list of integers representing a candidate solution for the TSP problem.
        ref: https://deap.readthedocs.io/en/master/api/tools.html?highlight=logbook%20object#hall-of-fame

    """
    # Set up the problem
    num_locations = len(locations)

    # Define the fintess function
    def point_to_point_time(start_point: list[float, float], end_point: list[float, float], ref=lookup) -> float: 
        """query the dataframe to find the time between two points 
        
        note
        - to be replaced later with pred that does lookup on a model
        """
        for row in ref: 
            if row[0] == start_point and row[1] == end_point: 
                return row[2]

    def tsp_fitness(individual):
        # Calculate the total travel time for the TSP path
        fitness = 0
        for i in range(num_locations):
            location1 = locations[individual[i]]
            location2 = locations[individual[(i+1) % num_locations]]
            fitness += point_to_point_time(location1, location2)
        return (fitness,)

    # Create the DEAP toolbox
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    # Register the functions to generate and manipulate individuals
    toolbox.register("indices", random.sample, range(num_locations), num_locations)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the genetic operators
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Set up the evaluation function
    toolbox.register("evaluate", tsp_fitness)

    # Set up the genetic algorithm
    pop = toolbox.population(n=population_size)

    # Define the statistics to track
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    # Run the genetic algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, num_generations, 
                                       stats=stats, halloffame=hof, verbose=True)

    # Return the population, logbook, and Hall of Fame
    return pop, logbook, hof


# -----------------------------------
# Helper Functions 

def get_best_sequence(individual, locations):
    # Sort the locations based on the individual's order
    ordered_locations = [locations[i] for i in individual]
    # Add the first location to the end to make a loop
    ordered_locations.append(locations[individual[0]])
    return ordered_locations

# Plotting 
def plot_evolution(logbook):

    # Get the fitness values from the logbook
    gen = logbook.select("gen")
    best = logbook.select("min")
    best_score = min(best)

    # Plot the best score vs. generation
    plt.plot(gen, best)
    plt.title("Best score vs. generation")
    plt.xlabel("Generation")
    plt.ylabel("Best score")
    plt.text(gen[-1], best_score, f"Best Time: {best_score:.2f} min", ha="right", va="center")
    plt.show()

def export_csv(location_sequence: list):
    df = pd.DataFrame(location_sequence, columns = ['x', 'y'])
    df['index'] = df.index

    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"locationSequence_{timestamp}.csv"
    df.to_csv(filename)
    print(f"... exported sequence to current directory as: {filename}")

# -----------------------------------
# MAIN 

# Initialize list of locations
locations, lookup = generate_data(size = 20)

# Run TSP 
pop, logbook, hof = tsp_solver(locations = locations, 
                               population_size = 100,
                               num_generations = 100,
                               cxpb = 0.5,
                               mutpb = 0.2
                               )


# Get the best individual
best_individual = hof[0]
best_fitness = best_individual.fitness.values[0]
best_sequence = get_best_sequence(best_individual, locations)


# Viz & export 
plot_evolution(logbook)
print("The best sequence is:")
pprint(best_sequence)
export_csv(best_sequence)


    
