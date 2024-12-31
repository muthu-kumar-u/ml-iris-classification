import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

async def train_model(X, y, model_dir):
    """
    Train multiple machine learning models and save them as .pkl files.

    Args:
        X (array): Feature matrix.
        y (array): Target array.
        model_dir (str): Directory to save model files.

    Returns:
        dict: Trained models.
    """

    if not model_dir or not os.path.exists(model_dir):
        raise ValueError(f"The file path is empty or the file does not exist: {model_dir}")

    # Define models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=200),
        "Decision_Tree": DecisionTreeClassifier(),
        "Random_Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
    }

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model

        # Save the model
        model_path = os.path.join(model_dir, f"{name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    return trained_models
