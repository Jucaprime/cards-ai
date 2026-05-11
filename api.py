from fastapi import FastAPI
from firebase_db import db
import pandas as pd
import joblib
import os
print("STARTING APP")
print(os.getcwd())

app = FastAPI()

model = joblib.load("cards_model.pkl")

FEATURES = [
    "home_cards_avg",
    "away_cards_avg",
    "referee_avg",
    "over45_rate",
    "odds_over45",
    "last5_home",
    "last5_away",
    "is_derby"
]

@app.get("/")
def home():
    return {"status": "online"}

@app.post("/predict")
def predict(data: dict):

    try:
        df = pd.DataFrame([data])
        df = df[FEATURES]

        prob = model.predict_proba(df)
        prob_over = float(prob[0][1])

        odd = float(data["odds_over45"])

        ev = (prob_over * odd) - 1

        entrada_valor = ev > 0.10

        result = {
            "probabilidade_over45": round(prob_over * 100, 2),
            "probabilidade_under45": round((1 - prob_over) * 100, 2),
            "odd": odd,
            "ev": round(ev, 3),
            "entrada_valor": entrada_valor
        }

        # salvar no Firebase (AGORA CERTO)
        db.collection("predictions").add({
            "home_cards_avg": data["home_cards_avg"],
            "away_cards_avg": data["away_cards_avg"],
            "probabilidade_over45": result["probabilidade_over45"],
            "ev": result["ev"],
            "entrada_valor": entrada_valor
        })

        return result

    except Exception as e:
        return {"erro": str(e)}