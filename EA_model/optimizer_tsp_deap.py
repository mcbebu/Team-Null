
# General use libraries 
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd 
from query_model import query_model

from datetime import datetime
from itertools import permutations
from pprint import pprint 

# EA libraries 
from deap import base, creator, tools, algorithms

# cache function
from functools import lru_cache
# import warnings
# from sklearn.exceptions import UserWarning
# warnings.filterwarnings("ignore", category=UserWarning)
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

def get_data():
    df = pd.read_csv('location.csv')
    df = df.assign(lat=df.location.str.split(',').str[0])
    df = df.assign(long=df.location.str.split(',').str[1])
    df = df.drop(columns=['location'])
    # return list
    data = df.values.tolist()
    return data

# -----------------------------------
# EA functions using DEAP 
import math
import requests
import json
key = "AIzaSyBN9wjKeVnXXeMK3dtVWFgFYjGfL18MyyA"


@lru_cache(maxsize=2048)
def get_google_duration(lat1, lon1, lat2, lon2):
    direction_url = "https://maps.googleapis.com/maps/api/directions/json?origin={},{}&destination={},{}&mode=driving&key={}"
    response = requests.get(direction_url.format(lat1, lon1, lat2, lon2, key))
    result_json_obj = json.loads(response.text)
    dur = result_json_obj['routes'][0]['legs'][0]['duration']['value']
    
    return float(dur)

def distance(lat1, lon1, lat2, lon2):
    # lat_long = str(lat_long)
    # source, dest = lat_long.split(';')
    # lat1, lon1 = source.split(',')
    # lat2, lon2 = dest.split(',')
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    R = 6371  # radius of the earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return float(distance)



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
    def point_to_point_time(start_point, end_point) -> float: 
        """query the dataframe to find the time between two points 
        
        note
        - to be replaced later with pred that does lookup on a model
        """
        # build a df distance,hour,weekday,source_destination,google_duration
        df = pd.DataFrame(columns=['distance', 'hour', 'weekday', 'source_destination', 'google_duration'])
        dist = distance(start_point[0], start_point[1], end_point[0], end_point[1])
        duration = get_google_duration(start_point[0], start_point[1], end_point[0], end_point[1])
        # duration = 356.51 * dist + 35.303
        # add one row
        df = df.append({'distance': dist, 'hour': 10, 'weekday': True, 'source_destination': f'{start_point[0]},{start_point[1]};{end_point[0]},{end_point[1]}', 'google_duration': duration}, ignore_index=True)
        
        
        df = df.assign(source=df.source_destination.str.split(';').str[0])
        df = df.assign(destination=df.source_destination.str.split(';').str[1])
        # split source lat long into source_lat and source_long
        df = df.assign(source_lat=df.source.str.split(',').str[0])
        df = df.assign(source_long=df.source.str.split(',').str[1])
        df = df.assign(destination_lat=df.destination.str.split(',').str[0])
        df = df.assign(destination_long=df.destination.str.split(',').str[1])

        df = df.drop(columns=['source_destination', 'source', 'destination'])
        
        # random noise from -2 to 2
        noise = np.random.uniform(-2, 2)
        prediction = query_model(df)[0] * 5 + noise
        print(prediction)
        return prediction

    def tsp_fitness(individual):
        # Calculate the total travel time for the TSP path
        # make correct df first
        
        # set weekday column to all True
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

# Plotting  f
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
# locations, lookup = generate_data(size = 20)

# read test.csv
location_df = get_data()
print(location_df)

# Run TSP 
pop, logbook, hof = tsp_solver(locations = location_df, 
                               population_size = 10,
                               num_generations = 40,
                               cxpb = 0.5,
                               mutpb = 0.2
                               )


# Get the best individual
best_individual = hof[0]
best_fitness = best_individual.fitness.values[0]
best_sequence = get_best_sequence(best_individual, location_df)


# Viz & export 
plot_evolution(logbook)
print("The best sequence is:")
pprint(best_sequence)
export_csv(best_sequence)


    
