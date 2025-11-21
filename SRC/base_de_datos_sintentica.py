import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Configurar semillas para reproducibilidad
np.random.seed(42)

# --- Crear dataset sintético ---
def create_realistic_diabetes_dataset(n_samples=500):
    """Crea un dataset sintético con valores realistas basado en el dataset PIMA."""
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=42
    )

    data = pd.DataFrame(X, columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    # Transformación a valores realistas
    data['Pregnancies'] = np.clip((data['Pregnancies'] * 2 + 3).astype(int), 0, 17)
    data['Glucose'] = np.clip((data['Glucose'] * 30 + 100).astype(int), 50, 200)
    data['BloodPressure'] = np.clip((data['BloodPressure'] * 15 + 70).astype(int), 40, 120)
    data['SkinThickness'] = np.clip((data['SkinThickness'] * 10 + 20).astype(int), 7, 99)
    data['Insulin'] = np.clip((data['Insulin'] * 80 + 80).astype(int), 14, 846)
    data['BMI'] = np.clip(data['BMI'] * 10 + 25, 18, 67)
    data['DiabetesPedigreeFunction'] = np.clip(data['DiabetesPedigreeFunction'] * 0.3 + 0.4, 0.08, 2.42)
    data['Age'] = np.clip((data['Age'] * 10 + 30).astype(int), 21, 81)

    data['Outcome'] = y
    return data


# --- Cargar o crear dataset ---
def load_diabetes_data():
    """
    Carga diabetes.csv.
    Si no existe en /data/raw/, lo crea automáticamente.
    """

    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/diabetes.csv"

    if not os.path.exists(file_path):
        print("No existe diabetes.csv, generando dataset sintetico...")
        df = create_realistic_diabetes_dataset(500)
        df.to_csv(file_path, index=False)
        print("Dataset creado correctamente en:", file_path)
    else:
        print("Cargando dataset existente...")

    return pd.read_csv(file_path)


def main():
    print("Generando o cargando dataset de diabetes...")
    df = load_diabetes_data()
    print("Primeras filas del dataset:")
    print(df.head())


if __name__ == "__main__":
    main()
