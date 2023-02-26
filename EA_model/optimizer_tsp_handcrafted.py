
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import random
# -----------------------------------
# Generate mock data

points = [[random.randint(0, 50), random.randint(0, 50)] for i in range(20)]

# create pair-wise permutations of the 5 samples
# generate a lookup from them
from itertools import permutations, combinations
perm = [pair for pair in permutations(points, 2)]
lookup = []
for pair in perm:
    lookup.append([pair[0], pair[1], np.sqrt((pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2)])
    # print(pair[0], pair[1])

# print("Lookup:")
# pprint(lookup)

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
    generation_scored = sorted(fitness_scores(input_generation), reverse=True)
    generation_sorted = [gen[0] for gen in generation_scored]
    if verbose: print(generation_scored)
    
    # get the best guesses
    breeders = generation_sorted[:take_best_N]
    best_candidate = breeders[0]
    best_score = generation_scored[0][1]
    
    # get the random guesses and add to breeders
    for _ in range(take_random_N):
        idx = np.random.randint(0, len(generation_sorted) - 1)
        breeders.append(generation_sorted[idx])

    if verbose: 
        print("Best candidates: ", best_candidates)
        print("Breeders: ", breeders)
    return breeders, (best_candidate, best_score)

def breed(parent1, parent2): 
    """
    Breed to candidates to create a new candidates
    
    """
    # create a new candidate
    child = []
    
    # get a random index
    idx = np.random.randint(0, len(parent1) - 1)
    
    # add the first part of the first parent
    child.extend(parent1[:idx])
    
    # add the second part of the second parent, filling in for values that arent already there. maintain sequencing
    for item in parent2:
        if item not in child:
            child.append(item)

    return child

# def breed(parent1, parent2): 
#     # print(parent1)
#     parent_temp = [1 for _ in parent1]
#     list_of_ids_for_parent1 = list(np.random.choice(parent_temp, replace=False, size=len(parent1)//2))
#     # list_of_ids_for_parent1 = np.random.choice(len(parent1), replace=False, size = len(parent1)//2)
    
#     child = [-99 for _ in parent1]
    
#     for ix in list_of_ids_for_parent1:
#         child[ix] = parent1[ix]
#     for ix, gene in enumerate(child):
#         if gene == -99:
#             for gene2 in parent2:
#                 if gene2 not in child:
#                     child[ix] = gene2
#                     break
#     child[-1] = child[0]
#     return child

def breed_generation(breeders, children_per_couple = 1, verbose = False): 
    """
    Breed a generation of candidates to create a new generation

    Pair breeders together and make children for each pair
    pairwise done by doing best-to-worst 

    
    """
    # pair breeders 
    idx_split = len(breeders)//2
    batch_a = breeders[:idx_split]
    batch_b = breeders[idx_split:]
    batch_b.reverse()
    mated_breeders = list(zip(batch_a, batch_b))
    if verbose: 
        print("Mated Breeders:")
        pprint(mated_breeders)

    # make new generation from each breeder pair 
    new_generation = []
    for couple in mated_breeders:
        for _ in range(children_per_couple):
            child = breed(couple[0], couple[1])
            new_generation.append(child)
    
    return new_generation

def evolutionary_solver(
        current_generation, 
        max_generations = 100, 
        take_best_N = 20,
        take_random_N = 3,
        children_per_couple = 1,
        print_every = 10,
        verbose = False):
    
    fitness_tracker = []
    best_candidate = ([na],100000)
    for i in range(max_generations):
        
        # get the breeders
        breeders, epoch_winner = get_breeders(current_generation, take_best_N, take_random_N, verbose)
        
        best_candidate = epoch_winner if epoch_winner[1] < best_candidate[1] else best_candidate
        
        # breed the new generation
        current_generation = breed_generation(breeders, children_per_couple, verbose)
        
        # get the fitness score
        fitness_tracker.append(fitness_scores(current_generation)[0][1])
        
        # print
        if (i % print_every == 0): 
            # print(f"Generation: {i} - Best candidate: {best_candidate}, - Score: {fitness_tracker[-1]}")
            print(f"Generation: {i} - | Score: {fitness_tracker[-1]}")
    
    return best_candidate, fitness_tracker

# -----------------------------------

def plot_fitness(_best_sequence, _fitness_tracker ): 
    """
    Plot the fitness tracker
    """
    # plot the fitness tracker score as a line graph
    plt.plot(_fitness_tracker)
    
    # plot a line for the best score & label it 
    plt.axhline(y= _best_sequence[1], color=r, linestyle=-)

    # print the best_sequence score on the graph
    plt.text(0, _best_sequence[1], f"Best Score: {_best_sequence[1]}")

    # label 
    plt.ylabel("Fitness Score")
    plt.xlabel("Generation")
    # title 
    plt.title("Fitness Score over Generations")

    plt.show()

gen1 = create_generation(points, population = 100)
stops = len(gen1)//2

# [X] TODO - fix best candidate
# [X] TODO - create plotter function

test = [] 
for i in range(1): 
    best_sequence, fitness_tracker = evolutionary_solver(gen1, max_generations = 100, take_best_N = stops, take_random_N = stops//2, children_per_couple = 1, print_every = 10, verbose = False)
    plot_fitness(best_sequence, fitness_tracker)
    test.append(best_sequence[1])


