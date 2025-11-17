import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("Leyendo dataset...")

# Lectura correcta del CSV
df = pd.read_csv("bank.csv")

if "deposit" not in df.columns:
    raise ValueError("ERROR: La columna objetivo 'deposit' no existe en el CSV. Columnas encontradas: " + str(df.columns.tolist()))

# Separar variables
X = df.drop("deposit", axis=1)
y = df["deposit"].map({"yes": 1, "no": 0})

# Identificar columnas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocesamiento
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Modelo
model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
])

print("Entrenando modelo...")

model.fit(X, y)

# Métricas
y_pred = model.predict(X)
metrics = {
    "accuracy": float(accuracy_score(y, y_pred)),
    "precision": float(precision_score(y, y_pred)),
    "recall": float(recall_score(y, y_pred)),
    "f1": float(f1_score(y, y_pred))
}

print("Métricas:", metrics)

# Guardar modelo
joblib.dump(model, "modelo_banking.pkl")
print("Modelo guardado como modelo_banking.pkl")

# Guardar métricas como JSON
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("metrics.json creado ✔")

# Guardar columnas características del dataset
feature_columns = X.columns.tolist()
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=4)
print("feature_columns.json creado ✔")
