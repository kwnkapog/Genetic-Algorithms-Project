import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from  scipy.stats import qmc


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

def create_initial_populations(num_populations, num_chromosomes):
    """
    Creates the initial populations of the genetic algorithm by constructing the appropriate chromosomes, ranged within the specific values. 

    Args:
        num_populations (int): The number of initial populations .
        num_chromosomes (int): The number of chromosomes per population.

    Returns:
        np array: An array consisting of all the initialized populations.
        np array: The values of the cosine similarity between the incomplete target inscription and the ones closest to it.
    """
    populations = []
    
    for _ in range(num_populations):
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n= num_chromosomes)
        samples = qmc.scale(samples, [0,0], [1677,1677])
        samples = np.round(samples).astype(int)
        
        population = [tuple(chromosome) for chromosome in samples]
        populations.append(population)
    return np.array(populations)


    