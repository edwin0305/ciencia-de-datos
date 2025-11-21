from fastapi import FastAPI
from src.predict import predecir_valor

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API Diabetes funcionando"}

@app.post("/predict")
def predict(data: dict):
    prediccion = predecir_valor(data)
    return {"prediction": prediccion}
