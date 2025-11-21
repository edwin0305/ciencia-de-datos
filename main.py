from fastapi import FastAPI
from src.data_loader import load_diabetes_data

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "API activa"}

@app.get("/data/info")
def get_info():
    df = load_diabetes_data()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "diabetes_rate": float(df["Outcome"].mean())
    }

@app.get("/data/preview")
def preview():
    df = load_diabetes_data()
    return df.head(10).to_dict(orient="records")
