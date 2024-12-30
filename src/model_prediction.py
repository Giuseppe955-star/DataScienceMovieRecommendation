import pickle
import numpy as np


def make_predictions(user_id, model_path, pivot_table, top_n=3):
    # Load the model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Align pivot_table columns with model components
    svd_matrix = model.transform(pivot_table)

    # Get user index
    user_idx = pivot_table.index.get_loc(user_id)

    # Compute predicted ratings for all movies
    user_ratings = svd_matrix[user_idx].dot(model.components_)

    # Sort movies by predicted rating
    recommended_movies_indices = np.argsort(user_ratings)[::-1][:top_n]
    recommended_movies = pivot_table.columns[recommended_movies_indices]

    return recommended_movies.tolist()

