import pandas as pd
import os

def load_diabetes_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_path, "data", "diabetes.csv")
    return pd.read_csv(file_path)
