import pandas as pd
import os

def cargar_datos():
    ruta = os.path.join("data", "raw", "diabetes.csv")
    df = pd.read_csv(ruta)
    return df
