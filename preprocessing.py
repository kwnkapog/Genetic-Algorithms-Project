import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset into a dataframe
df = pd.read_csv('iphi2802.csv', delimiter='\t')

# Print the original shape of the dataset
print(f"Original shape of Dataset: {df.shape}")

# Print information about the dataset
print(f'\nDataset Information:')
df.info()

# Print the number of NULL values in each column
print(f'\nNumber of NULL values per column:')
print(df.isnull().sum())

# Print the number of unique values in each column
print(f'\nNumber of unique values per column:')
print(df.nunique())

# Keep rows where the region_main_id is 1693 
df_filtered= df.query("region_main_id == 1693")

# Initalize the tf-idf Vectorizer and transform the text column of the dataframe.
vectorizer = TfidfVectorizer()
index_matrix = vectorizer.fit_transform(df_filtered['text'].to_list())

# Visualize the output of the vectorizer(words and their idf values)
shape = index_matrix.shape
idf_values = vectorizer.idf_
vocab = sorted(vectorizer.vocabulary_)
matrix = index_matrix.toarray()
matrix_uniques = np.unique(matrix)
