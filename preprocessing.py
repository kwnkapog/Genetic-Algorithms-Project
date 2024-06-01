import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches
import utils as ul
import importlib
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

