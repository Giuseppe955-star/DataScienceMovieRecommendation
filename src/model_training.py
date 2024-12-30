import pandas as pd
from sklearn.decomposition import TruncatedSVD
import pickle


def train_model(df):
    # Create a pivot table
    pivot_table = df.pivot_table(index='user', columns='name', values='rating', fill_value=0)

    # Decompose the matrix
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_matrix = svd.fit_transform(pivot_table)

    # Save the model
    with open('../models/model.pkl', 'wb') as model_file:
        pickle.dump(svd, model_file)

    return svd, pivot_table
