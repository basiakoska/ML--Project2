# -*- coding: utf-8 -*-
"""
CROSS VALIDATION WITH BASELINE MODEL 

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Carica il dataset e standardizzalo
dataset = pd.read_excel(r'C:\Users\Giulia\Desktop\DTU\1\Intro2ml&dm\project1\diabetes_dataset.xlsx')
dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

# Separazione tra attributi predittivi (features) e target
dataset = dataset.drop('Outcome', axis=1)

# Selezionare tutte le colonne tranne 'BloodPressure' per X
X = dataset.drop('BloodPressure', axis=1)
y = dataset['BloodPressure']



# Inizializzazione dei KFold per entrambi i livelli
K1 = K2 = 10
kf1 = KFold(n_splits=K1, shuffle=True)
kf2 = KFold(n_splits=K2, shuffle=True)

baseline_errors = []
model_errors = []

# Iterazione sui fold esterni (outer loop)
for train_index_outer, test_index_outer in kf1.split(X):
    X_train_outer, X_test_outer = X.iloc[train_index_outer], X.iloc[test_index_outer]
    y_train_outer, y_test_outer = y.iloc[train_index_outer], y.iloc[test_index_outer]

    # Baseline Model (calcola la media di y sul training set e predici con questo valore)
    baseline_prediction = np.mean(y_train_outer)
    baseline_error = mean_squared_error(np.repeat(baseline_prediction, len(y_test_outer)), y_test_outer)
    baseline_errors.append(baseline_error)

    # Modello di Regressione Lineare
    fold_model_errors = []

    # Iterazione sui fold interni (inner loop) per il tuning del modello
    for train_index_inner, test_index_inner in kf2.split(X_train_outer):
        X_train_inner, X_val = X_train_outer.iloc[train_index_inner], X_train_outer.iloc[test_index_inner]
        y_train_inner, y_val = y_train_outer.iloc[train_index_inner], y_train_outer.iloc[test_index_inner]

        # Addestramento del modello
        model = LinearRegression()
        model.fit(X_train_inner, y_train_inner)

        # Valutazione sul validation set
        y_pred = model.predict(X_val)
        fold_model_error = mean_squared_error(y_val, y_pred)
        #print(fold_model_error)
        fold_model_errors.append(fold_model_error)
        

    # Calcolo dell'errore del modello per questo fold esterno (media dell'errore sui fold interni)
    model_errors.append(np.mean(fold_model_errors))

# Calcolo dell'errore medio del baseline model e del modello di regressione lineare
baseline_avg_error = np.mean(baseline_errors)
model_avg_error = np.mean(model_errors)

#print(f"Baseline Model Average Error: {baseline_avg_error}")
#print(f"Linear Regression Model Average Error: {model_avg_error}")

print(fold_model_errors)

models = ['Baseline Model', 'Linear Regression Model']
errors = [baseline_avg_error, model_avg_error]

colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'lightcoral', 'cyan', 'orange', 'limegreen', 'tomato', 'deepskyblue']

plt.bar(range(1, K1 + 1), fold_model_errors, color=colors,width=0.3)
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('MSE per Fold in Baseline Model Two-Level Cross-Validation')
plt.show()

#plt.bar(models, errors, color=['blue', 'green'])
#plt.xlabel('Models')
#plt.ylabel('Average Error')
#plt.title('Comparison of Average Errors between Models')
#plt.show()

