import pandas as pd
from sklearn.preprocessing import StandardScaler

def limpiar_datos(df):

    columnas_a_corregir = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]

    df[columnas_a_corregir] = df[columnas_a_corregir].replace(0, pd.NA)

    for col in columnas_a_corregir:
        df[col] = df[col].fillna(df[col].median())

    return df


def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
