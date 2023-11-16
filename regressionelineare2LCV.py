from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Carica il dataset e standardizzalo
dataset = pd.read_excel(r'C:\Users\Giulia\Desktop\DTU\1\Intro2ml&dm\project1\diabetes_dataset.xlsx')
dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

# Separazione tra attributi predittivi (features) e target
X = np.column_stack((dataset['Insulin'], dataset['Age'], dataset['DiabetesPedigreeFunction'], dataset['Glucose'], dataset['SkinThickness'], dataset['Pregnancies'], dataset['BMI']))
y = dataset['BloodPressure'].squeeze().values

N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
M = M + 1

# Definisci i valori di lambda
lambda_values = np.power(10., range(-5, 9))
lambda_values=[0.0001,60,100]
# Definisci il modello Ridge
ridge = Ridge()

# Inizializza le liste per gli errori e i valori di lambda
errors_outer = []
lambda_values_outer = []

# Imposta la cross-validation esterna e interna
K1 = K2 = 10
CV1 = KFold(n_splits=K1, shuffle=True)
CV2 = KFold(n_splits=K2, shuffle=True)

for train_index_outer, test_index_outer in CV1.split(X):
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

    errors_inner = []
    lambda_values_inner = []

    for train_index_inner, test_index_inner in CV2.split(X_train_outer):
        X_train_inner, X_val = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
        y_train_inner, y_val = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

        errors_lambda = []

        for lambda_val in lambda_values:
            ridge.set_params(alpha=lambda_val)  # Imposta il valore di lambda
            ridge.fit(X_train_inner, y_train_inner)
            y_pred_val = ridge.predict(X_val)
            error_val = mean_squared_error(y_val, y_pred_val)
            errors_lambda.append(error_val)

        best_lambda_index = np.argmin(errors_lambda)
        best_lambda = lambda_values[best_lambda_index]

        ridge.set_params(alpha=best_lambda)
        ridge.fit(X_train_outer, y_train_outer)
        y_pred_outer = ridge.predict(X_test_outer)
        error_outer = mean_squared_error(y_test_outer, y_pred_outer)
        errors_inner.append(error_outer)
        lambda_values_inner.append(best_lambda)

    best_error_index = np.argmin(errors_inner)
    best_lambda_outer = lambda_values_inner[best_error_index]
    best_error_outer = errors_inner[best_error_index]

    errors_outer.append(best_error_outer)
    lambda_values_outer.append(best_lambda_outer)

print(errors_outer)
print(lambda_values_outer)

colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'lightcoral', 'cyan', 'orange', 'limegreen', 'tomato', 'deepskyblue']


plt.bar(range(1, K1 + 1), errors_outer, color=colors,width=0.3)
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('MSE per Fold in Linear Regression Two-Level Cross-Validation')
plt.show()