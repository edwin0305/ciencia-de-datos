import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .data_loader import cargar_datos
from .preprocessing import limpiar_datos, escalar_datos

def entrenar_modelo():

    df = cargar_datos()

    df = limpiar_datos(df)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    columnas = list(X.columns)

    X_scaled, scaler = escalar_datos(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(modelo, "models/modelo_diabetes.joblib")
    joblib.dump(scaler, "models/escalador_diabetes.joblib")
    joblib.dump(columnas, "models/columnas_entrenamiento.joblib")

    with open("reports/metricas.json", "w") as file:
        json.dump({"accuracy": accuracy}, file)

    print("Modelo entrenado correctamente. Accuracy:", accuracy)
