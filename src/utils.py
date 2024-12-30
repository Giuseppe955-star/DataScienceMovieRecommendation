from data_processing import process_data
from model_training import train_model
from model_prediction import make_predictions
import json

if __name__ == "__main__":
    # Process data
    df, label_encoder = process_data('../data/movies_dataset.csv')

    # Train the model
    svd, pivot_table = train_model(df)

    # Make predictions for users
    predictions = {}
    for user in pivot_table.index:
        predictions[user] = make_predictions(user, '../models/model.pkl', pivot_table)

    # Save predictions to JSON
    with open('../predictions/predictions.json', 'w') as pred_file:
        json.dump({"target": predictions}, pred_file)
