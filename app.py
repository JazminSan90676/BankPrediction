import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve)

app = Flask(__name__, template_folder='templates')
CORS(app)

# Directorio estático para gráficas
PLOTS_DIR = os.path.join('static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_PATH = 'modelo_banking.pkl'
FEATURES_PATH = 'feature_columns.json'
METRICS_PATH = 'metrics.json'
DB_PATH = 'predictions.db'

# Cargar artefactos si existen
model = None
feature_columns = None
metrics = {}
model_plots = {
    'confusion': os.path.join(PLOTS_DIR, 'confusion.png'),
    'roc': os.path.join(PLOTS_DIR, 'roc.png'),
    'pr': os.path.join(PLOTS_DIR, 'pr.png')
}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_json TEXT,
        prediction INTEGER,
        probability REAL,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        confusion_png TEXT,
        roc_png TEXT,
        pr_png TEXT
    )
    ''')
    conn.commit()
    conn.close()


def generate_model_plots_and_metrics():
    # If we have training CSV and a loaded model, compute global metrics and plots
    global metrics
    try:
        if not os.path.exists('bank.csv') or model is None:
            return

        df = pd.read_csv('bank.csv')
        if 'deposit' not in df.columns:
            return

        X = df.drop('deposit', axis=1)
        y = df['deposit'].map({'yes': 1, 'no': 0})

        # Ensure feature columns alignment if available
        if feature_columns:
            X = X[feature_columns]

        y_pred = model.predict(X)
        y_prob = None
        try:
            y_prob = model.predict_proba(X)[:, 1]
        except Exception:
            # If predict_proba not available, use predictions as 0/1
            y_prob = y_pred

        # Compute metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred))
        }

        # Confusion matrix plot
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusión')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['No','Yes'])
        plt.yticks(tick_marks, ['No','Yes'])
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.tight_layout()
        plt.savefig(model_plots['confusion'])
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(model_plots['roc'])
        plt.close()

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y, y_prob)
        plt.figure(figsize=(5,4))
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.tight_layout()
        plt.savefig(model_plots['pr'])
        plt.close()

        # Save metrics.json as well so UI existing code still works
        with open(METRICS_PATH, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

    except Exception as e:
        print('Error generating plots/metrics:', e)


# Inicialización: DB y plots
init_db()
generate_model_plots_and_metrics()

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
        prediction = int(model.predict(X_input)[0])
        # Probabilidad para la clase positiva si está disponible
        probability = None
        try:
            probs = model.predict_proba(X_input)
            # probs shape (n_samples, n_classes)
            if probs is not None and len(probs.shape) == 2:
                if probs.shape[1] > 1:
                    probability = float(probs[0][1])
                else:
                    probability = float(probs[0][0])
        except Exception:
            probability = None

        # Guardar en DB: guardamos métricas globales actuales y rutas de las imágenes
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        ts = datetime.utcnow().isoformat()
        input_json = json.dumps(user_input, ensure_ascii=False)
        acc = float(metrics.get('accuracy')) if metrics.get('accuracy') is not None else None
        prec = float(metrics.get('precision')) if metrics.get('precision') is not None else None
        rec = float(metrics.get('recall')) if metrics.get('recall') is not None else None
        f1 = float(metrics.get('f1')) if metrics.get('f1') is not None else None
        # store web-accessible paths
        confusion_web = '/' + model_plots['confusion'].replace('\\','/')
        roc_web = '/' + model_plots['roc'].replace('\\','/')
        pr_web = '/' + model_plots['pr'].replace('\\','/')

        c.execute('''INSERT INTO predictions (timestamp, input_json, prediction, probability, accuracy, precision, recall, f1, confusion_png, roc_png, pr_png)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (ts, input_json, prediction, probability, acc, prec, rec, f1, confusion_web, roc_web, pr_web))
        conn.commit()
        row_id = c.lastrowid
        conn.close()

        return jsonify({
            "id": row_id,
            "prediction": prediction,
            "probability": probability,
            "message": "El cliente responde positivamente" if prediction == 1 else "El cliente NO responde"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    if metrics:
        return jsonify(metrics)
    return jsonify({'message': 'Metrics not found. Run train_model.py.'}), 404


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/history_data')
def history_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, input_json, prediction, probability, accuracy, precision, recall, f1, confusion_png, roc_png, pr_png FROM predictions ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            'id': r[0],
            'timestamp': r[1],
            'input': json.loads(r[2]) if r[2] else {},
            'prediction': r[3],
            'probability': r[4],
            'accuracy': r[5],
            'precision': r[6],
            'recall': r[7],
            'f1': r[8],
            'confusion_png': r[9],
            'roc_png': r[10],
            'pr_png': r[11]
        })

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=False en producción
    app.run(host='0.0.0.0', port=port, debug=False)
