from fastapi import FastAPI
from firebase_db import db
import pandas as pd
import joblib

app = FastAPI()

# carregar modelo
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

        # transformar dataframe
        df = pd.DataFrame([data])

        # ordem correta
        df = df[FEATURES]

        # previsão
        prob = model.predict_proba(df)

        # probabilidade over
        prob_over = float(prob[0][1])

        # odd enviada
        odd = float(data["odds_over45"])

        # calcular EV
        ev = (prob_over * odd) - 1

        # entrada de valor
        entrada_valor = ev > 0.10

        return {

            "probabilidade_over45": round(prob_over * 100, 2),

            "probabilidade_under45": round(
                (1 - prob_over) * 100,
                2
            ),

            "odd": odd,

            "ev": round(ev, 3),

            "entrada_valor": entrada_valor

        }
        db.collection("predictions").add({

    "home_cards_avg": data["home_cards_avg"],
    "away_cards_avg": data["away_cards_avg"],

    "probabilidade_over45": round(prob_over * 100, 2),

    "ev": round(ev, 3),

    "entrada_valor": entrada_valor

})

    except Exception as e:

        return {
            "erro": str(e)
        }