from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from services.model import get_knn_result, get_logistic_result, get_decision_tree_result, get_random_forest_result, get_svm_result

router = APIRouter()

# Define the request body model using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    # Validate the values to ensure they are positive (optional)
    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('All feature values must be positive')
        return v

# Helper function to process and get model prediction
async def get_model_result(model_func, features: IrisFeatures):
    try:
        # Extract features
        data = [features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]
        result = await model_func(data)

        if result is None:
            raise HTTPException(status_code=404, detail="Prediction result not found")

        return {"result": result}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post('/predict/knn')
async def predict_knn(features: IrisFeatures):
    return await get_model_result(get_knn_result, features)


@router.post('/predict/logistic')
async def predict_logistic(features: IrisFeatures):
    return await get_model_result(get_logistic_result, features)


@router.post('/predict/decision-tree')
async def predict_decision_tree(features: IrisFeatures):
    return await get_model_result(get_decision_tree_result, features)


@router.post('/predict/random-forest')
async def predict_random_forest(features: IrisFeatures):
    return await get_model_result(get_random_forest_result, features)


@router.post('/predict/svm')
async def predict_svm(features: IrisFeatures):
    return await get_model_result(get_svm_result, features)
