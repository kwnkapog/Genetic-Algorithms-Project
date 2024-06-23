import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from  scipy.stats import qmc
import pygad


def print_dataframe_info(df):
    """
    Prints information of a dataset on a pandas dataframe. 

    Args:
        df (Pandas dataframe): The dataframe to print information on
    """
    print(f'\nDataset Information:')
    df.info()
    print(f'\nNumber of NULL values per column:')
    print(df.isnull().sum())
    print(f'\nNumber of unique values per column:')
    print(df.nunique())
    
def print_vectorizer_info(vectorizer, matrix, printMatrix):
    """
    Prints information of a Tf-Idf matrix produced by a TfIdfVectorizer. 

    Args:
        vectorizer (TFIdfVectorizer obj): The object of the TfIdfVectorizer class.
        matrix (csr): The matrix containing the encoded inscriptions.
        printMatrix (boolean): Determines whether or not the actual matrix is printed.
    
    Returns:
        dictionary: The indexed dictionary of the vectorizer.
        
    """
    vocab = vectorizer.get_feature_names_out()
    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    print(f"The dictionary of the dataset: {vocab}")
    print(f"The shape of the output matrix: {matrix.shape}")
    if printMatrix: 
        print(f"The matrix: {matrix}")
    print(f"The unique values of the output matrix: {np.unique(matrix)}")
    return vocab_dict

 
def find_k_nearest(k, matrix, incomplete_vector):
    """
    Finds the k closest inscriptions to a target inscription, using cosine similarity to compare them.

    Args:
        k (int): The number of closest neighbours .
        matrix (np array): The matrix containing the tf-idf encoded inscriptions.
        incomplete_vector (np array): The vector containing the incomplete target inscription.

    Returns:
        np array: the indices of the discriptions closest to the incomplete target inscription.
        np array: The values of the cosine similarity between the incomplete target inscription and the ones closest to it.
    """
    similarities = []
    incomplete_vector = np.array(incomplete_vector).reshape(1, -1)
    
    for row in matrix:
        row = row.reshape(1, -1)
        similarity = cosine_similarity(row, incomplete_vector)[0][0]
        similarities.append(similarity)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    top_k_values = np.array([similarities[i] for i in top_k_indices])
    
    return top_k_indices, top_k_values    

def create_initial_populations(num_populations, population_size):
    """
    Creates the initial populations of the genetic algorithm by constructing the appropriate chromosomes, ranged within the specific values. 

    Args:
        num_populations (int): The number of initial populations .
        population_size (int): The number of chromosomes per population.

    Returns:
        np array: An array consisting of all the initialized populations.
        np array: The values of the cosine similarity between the incomplete target inscription and the ones closest to it.
    """
    init_populations = []
    
    for _ in range(num_populations):
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n= population_size)
        samples = qmc.scale(samples, [0,0], [1677,1677])
        samples = np.round(samples).astype(int)
        
        population = [tuple(chromosome) for chromosome in samples]
        init_populations.append(population)
    return np.array(init_populations)

def set_ga_instances(init_populations, num_of_generations, crossover_prop, mutation_prop,fitness_func,early_stopping_callback):
    """
    Creates the instances of the genetic algorithm. 

    Args:
        init_populations (numpy array): A list containig the random initial populations.
        num_of_generations (int): The number of generations each population will run for.
        crossover_prop (float): The propability with which every chromosome is chosen for the crossover operation.
        mutation_prop (float): The propabilitywith which every chromosome is selected for the mutation operation.
        fitness_func (function): THe fitness function.
        early_stopping_callback (function): THe early stopping callback function.

    Returns:
        list: A list consisting of all the initialized instances of the pyGAD class.
    """
    ga_instances = []
    
    for init_population in init_populations:
        ga_instance = pygad.GA(initial_population=init_population,
                                gene_space={'low':0, 'high':1677, 'step':1},
                                num_generations= num_of_generations,
                                crossover_probability= crossover_prop,
                                num_parents_mating = len(init_population),
                                fitness_func=fitness_func,
                                parent_selection_type="rws",
                                keep_elitism=1,
                                crossover_type="single_point",
                                mutation_type="random",
                                mutation_probability=mutation_prop,
                                mutation_by_replacement=True,
                                on_generation=early_stopping_callback)
        ga_instances.append(ga_instance)
    return ga_instances

def run_instances(ga_instances):
    for instance in ga_instances:
        
