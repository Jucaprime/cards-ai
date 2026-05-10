import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# carregar dados
df = pd.read_csv("matches_cards.csv")

# features
features = [
    "home_cards_avg",
    "away_cards_avg",
    "referee_avg",
    "over45_rate",
    "odds_over45",
    "last5_home",
    "last5_away",
    "is_derby"
]

X = df[features]

# alvo
y = df["target_over45"]

# dividir
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# modelo
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

# treinar
model.fit(X_train, y_train)

# salvar
joblib.dump(model, "cards_model.pkl")

print("Modelo treinado com sucesso!")