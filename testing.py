import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches
import utils as ul
import importlib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import qmc
importlib.reload(ul)

# Initialize the pandas dataframe
df = pd.read_csv('iphi2802.csv', delimiter='\t')

# Keep rows of the same region as the target inscription. 
df_filtered= df.query("region_main_id == 1693")

# Initalize the tf-idf Vectorizer and transform the text column of the dataframe.
vectorizer = TfidfVectorizer()
index_matrix = vectorizer.fit_transform(df_filtered['text'].to_list()).toarray()
vocab_dict = ul.print_vectorizer_info(vectorizer, index_matrix, True)

# Find the word closest to the missing word 'αλεξανδρε' and turn the incomplete inscription into a tf-idf vector
missing_word = 'αλεξανδρε'
replaced_word = get_close_matches(missing_word, vocab_dict.values(), n=1)
replaced_inscription = f'{replaced_word} ουδις'
incomplete_vector = vectorizer.transform([replaced_inscription]).toarray()

top_indices, top_values = ul.find_k_nearest(5, index_matrix, incomplete_vector)

def fitness_func(ga_instance, solution, solution_idx):
    left_word_idx, right_word_idx = solution
    
    left_word = vocab_dict[left_word_idx]
    right_word = vocab_dict[right_word_idx]
    replaced_inscription = f'{left_word} αλεξανδρε ουδις {right_word}'
    
    replaced_vector = vectorizer.transform([replaced_inscription]).toarray().reshape(1, -1)
    fit_similarities = []
    
    for index in top_indices:
        inscription = index_matrix[index].reshape(1, -1)
        similarity = cosine_similarity(inscription, replaced_vector)[0][0]
        fit_similarities.append(similarity)
    fitness_value = np.mean(fit_similarities)
    
    mean_value = np.mean(top_values)
    
    # Punishing chromosomes with suboptimal fitness values, to discourage them from continuing(dont know if it is correct)
    if fitness_value < mean_value:
        fitness_value = fitness_value - (0.5 * fitness_value)
    
    return fitness_value


best_fitness_values = []
num_of_insignificant_better = []
num_of_insignificant_better.append(0)

#Stop early when GA converges
def early_stopping_callback(ga_instance):
    best_fitness_values.append(ga_instance.best_solution()[1])
        
    better_ratio = (best_fitness_values[len(best_fitness_values)-1] / best_fitness_values[len(best_fitness_values)-2]) - 1
    if better_ratio < 0.01 and len(best_fitness_values) > 1:
        num_of_insignificant_better.append(num_of_insignificant_better[len(num_of_insignificant_better)-1] + 1)
    else:
        num_of_insignificant_better.append(0)
    
    if num_of_insignificant_better[len(num_of_insignificant_better)-1] == 100:
        return "stop"
        


# TESTING THE FUCK OUT OF EVERYTHING

initial_populations = ul.create_initial_populations(10,20)
ga_instances = ul.set_ga_instances(initial_populations, 1000, 0.6, 0.01, fitness_func, early_stopping_callback)