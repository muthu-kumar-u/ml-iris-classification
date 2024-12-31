from dotenv import load_dotenv
import joblib
import os

load_dotenv(dotenv_path=".env")
MODEL_DIR = os.getenv("MODEL_PATH")

# Helper function to load a model and predict
async def predict_from_model(model_name, data):
    model_path = os.path.join(MODEL_DIR, f'{model_name.replace(" ", "_")}_model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model from the file
    model = joblib.load(model_path)

    # Predict and return the result as a string
    prediction = model.predict([data]) 
    return str(prediction[0])  
