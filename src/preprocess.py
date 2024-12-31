import os
import pandas as pd


async def preprocess_model(file_path):
    """
    Preprocess the dataset by removing unwanted columns, handling missing values,
    and encoding target variables.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: Features (X) and target (y) arrays.
    """
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"The file path is empty or the file does not exist: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Drop unwanted columns
    df = df.drop(columns=["Id"], errors="ignore")

    # Handle missing values
    df = df.dropna()

    # Encode target variable
    df['Species'] = df['Species'].astype('category').cat.codes

    # Split features and target
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    y = df['Species'].values

    return X, y
