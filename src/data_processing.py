import pandas as pd
from sklearn.preprocessing import LabelEncoder


def process_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Encode categorical columns
    le_theme = LabelEncoder()
    df['theme_encoded'] = le_theme.fit_transform(df['theme'])

    return df, le_theme
