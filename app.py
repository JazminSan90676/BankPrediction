import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='templates')
CORS(app)

MODEL_PATH = 'modelo_banking.pkl'
FEATURES_PATH = 'feature_columns.json'
METRICS_PATH = 'metrics.json'

# Cargar artefactos si existen
model = None
feature_columns = None
metrics = {}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

# Lista de campos originales esperados por el formulario (sin target 'deposit')
EXPECTED_FIELDS = [
    'age','job','marital','education','default','balance','housing','loan',
    'contact','day','month','duration','campaign','pdays','previous','poutcome'
]

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "input" not in data:
        return jsonify({"error": "Falta el campo 'input'"}), 400

    user_input = data["input"]

    try:
        # Convertir diccionario → DF con las columnas exactas del modelo
        X_input = pd.DataFrame([user_input], columns=feature_columns)

        # Predecir
        prediction = model.predict(X_input)[0]

        return jsonify({
            "prediction": int(prediction),
            "message": "El cliente responde positivamente" if prediction == 1 else "El cliente NO responde"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    if metrics:
        return jsonify(metrics)
    return jsonify({'message': 'Metrics not found. Run train_model.py.'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=False en producción
    app.run(host='0.0.0.0', port=port, debug=False)
