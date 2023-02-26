
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# -----------------------------------
# Generate mock data

# generate a dataset of 5 samples with random lat long values
# points = np.random.randint(0, 5, size=(5, 2))
points = [
    [0,1],
    [1,2],
    [2,3],
    [3,4],
    [4,5]
]

# create pair-wise permutations of the 5 samples
# generate a lookup from them
from itertools import permutations, combinations
perm = [pair for pair in permutations(points, 2)]
lookup = []
for pair in perm:
    lookup.append([pair[0], pair[1], np.sqrt((pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2)])
    # print(pair[0], pair[1])
print("Lookup:")
print(lookup)

"""
# OLD VERSION; using a dataframe for lookup
# for pair in perm:
#     pair.add(np.sqrt((pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2))


# create a dataframe with the pair-wise permutations
df = pd.DataFrame(perm, columns=[pt_from, pt_to])

# calculate the distance between each pair of points
df[metric] = df.apply(lambda row: np.sqrt((row[pt_from][0] - row[pt_to][0])**2 + (row[pt_from][1] - row[pt_to][1])**2), axis=1)
# print(df)
"""

# -----------------------------------
# EA functions
def create_guess(points): 
    # create a random guess of the order of the points
    guess = np.random.permutation(points)
    # convert to list
    guess = guess.tolist()
    return guess

def create_generation(points, population = 100):
    """
    Creates a generation of guesses based on list of points 
    
    INPUT   
    - list of (lat, long) points

    OUTPUT 
    - list of guesses in the form of a list of lists
    """
    generation = [create_guess(points) for i in range(population)]
    return generation

def point_to_point_time(start_point: list[float, float], end_point: list[float, float], ref=lookup) -> float: 
    """query the dataframe to find the time between two points 
    
    note
    - to be replaced later with pred that does lookup on a model
    """
    for row in ref: 
        if row[0] == start_point and row[1] == end_point: 
            return row[2]
    
def fitness_scores(generation: list, verbose = False) -> list: 
    """
    Calculate the fitness score

    Input: 
    - a generation of guesses in the form of a list of lists
    
    Output: 
    - pairwise list of guess-to-scores for each guess in the generation
    """
    scores = []

    for guess in generation:
        # create pairwise sequence of points in the guess
        pairs = list(zip(guess[:-1], guess[1:]))
        if verbose: print(pairs)
        score = 0
        for pair in pairs: 
            score += point_to_point_time(pair[0], pair[1])
            if verbose: print(pair[0], pair[1], "-score-> ", score)
        scores.append(score)
    
    return list(zip(generation, scores))

def get_breeders(input_generation, take_best_N = 10, take_random_N = 5, verbose = False): 
    """
    Gets the top guesses from a generation 
    and randomly selects a few more

    Notes
    - does not mutate the guesses since order is important

    """
    # sort the guesses by score
    generation_scored = sorted(fitness_scores(input_generation), key=lambda x: x[1])
    generation_sorted = [gen[0] for gen in generation_scored]
    if verbose: print(generation_scored)
    
    # get the best guesses
    breeders = generation_sorted[:take_best_N]
    best_candidate = breeders[0]
    
    # get the random guesses and add to breeders
    for _ in range(take_random_N):
        idx = np.random.randint(0, len(generation_sorted) - 1)
        breeders.append(generation_sorted[idx])

    if verbose: 
        print("Best candidates: ", best_candidates)
        print("Breeders: ", breeders)
    return breeders, best_candidate
