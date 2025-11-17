import pandas as pd

# Intenta leer con distintos separadores
try:
    df = pd.read_csv("bank.csv")
    print("Columnas detectadas con separador por defecto:")
    print(df.columns.tolist())
except:
    print("No se pudo leer con separador por defecto")

try:
    df = pd.read_csv("bank.csv", sep=";")
    print("Columnas detectadas con ';':")
    print(df.columns.tolist())
except:
    print("No se pudo leer con ';'")
