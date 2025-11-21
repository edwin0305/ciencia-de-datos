import numpy as np
import joblib

def predecir_valor(data: dict):

    modelo = joblib.load("models/modelo_diabetes.joblib")
    scaler = joblib.load("models/escalador_diabetes.joblib")
    columnas = joblib.load("models/columnas_entrenamiento.joblib")

    valores = [data[col] for col in columnas]

    valores = np.array(valores).reshape(1, -1)

    valores_scaled = scaler.transform(valores)

    pred = int(modelo.predict(valores_scaled)[0])

    return pred
