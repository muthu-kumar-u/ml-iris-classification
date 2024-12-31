from sklearn.metrics import accuracy_score

async def evaluate_model(models, X, y):
    """
    Evaluate trained models on the given dataset.

    Args:
        models (dict): Trained models.
        X (array): Feature matrix.
        y (array): Target array.

    Returns:
        dict: Accuracy scores for each model.
    """
    evaluation_results = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        evaluation_results[name] = accuracy

    return evaluation_results
