# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:35:54 2023

@author: Giulia
"""
import torch
import pandas as pd
import numpy as np
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset

# Carica il dataset e standardizzalo
dataset = pd.read_excel(r'C:\Users\Giulia\Desktop\DTU\1\Intro2ml&dm\project1\diabetes_dataset.xlsx')
dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

# Separazione tra attributi predittivi (features) e target
X = np.column_stack((dataset['Insulin'], dataset['Age'], dataset['DiabetesPedigreeFunction'], dataset['Glucose'], dataset['SkinThickness'], dataset['Pregnancies'], dataset['BMI']))
y = dataset['BloodPressure'].squeeze().values

hidden_units_values = [1, 2, 3, 4, 5]  # Modify these values based on your tests

n_replicates = 1  # Number of networks trained in each k-fold
max_iter = 10000
K = 10  # Number of folds for cross-validation
loss_fn = torch.nn.MSELoss()

CV_outer = model_selection.KFold(K, shuffle=True)

errors_outer = []  # List to store generalization errors for each outer fold

for k, (train_outer_index, test_outer_index) in enumerate(CV_outer.split(X, y)):
    print(f'\nOuter fold: {k + 1}/{K}')

    X_train_outer, X_test_outer = X[train_outer_index], X[test_outer_index]
    y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]

    CV_inner = model_selection.KFold(K, shuffle=True)
    best_h_per_inner_fold = []

    for train_inner_index, test_inner_index in CV_inner.split(X_train_outer, y_train_outer):
        X_train_inner, X_val = X_train_outer[train_inner_index], X_train_outer[test_inner_index]
        y_train_inner, y_val = y_train_outer[train_inner_index], y_train_outer[test_inner_index]

        errors_inner = []

        for h in hidden_units_values:
            print(f'Testing ANN model with {h} hidden units for inner fold')

            # Convert data to PyTorch tensors
            train_data = TensorDataset(torch.Tensor(X_train_inner), torch.Tensor(y_train_inner))
            train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

            # Initialize the model with different numbers of hidden units
            model = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1)
            )

            # Train the model on the training portion
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for epoch in range(max_iter):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets.view(-1, 1))
                    loss.backward()
                    optimizer.step()

            # Evaluate the model on the validation set
            with torch.no_grad():
                y_pred_val = model(torch.Tensor(X_val))
                error_val = loss_fn(y_pred_val, torch.Tensor(y_val).view(-1, 1))
                print("Err val=",error_val," with ",h," hidden units")
                errors_inner.append((h, error_val.item()))

        # Select the best number of hidden units for the current inner fold
        best_h_per_inner_fold.append(min(errors_inner, key=lambda x: x[1])[0])

    # Train the model on the entire training set using the best h for each inner fold
    best_h = max(set(best_h_per_inner_fold), key=best_h_per_inner_fold.count)

    # Train the model on the entire training set using the best h
    final_model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], best_h),
        torch.nn.Tanh(),
        torch.nn.Linear(best_h, 1)
    )

    optimizer = torch.optim.SGD(final_model.parameters(), lr=0.01)
    for epoch in range(max_iter):
        optimizer.zero_grad()
        outputs = final_model(torch.Tensor(X_train_outer))
        loss = loss_fn(outputs, torch.Tensor(y_train_outer).view(-1, 1))
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    with torch.no_grad():
        y_pred_test = final_model(torch.Tensor(X_test_outer))
        error_test = loss_fn(y_pred_test, torch.Tensor(y_test_outer).view(-1, 1))
        errors_outer.append(error_test.item())

# Print the generalization errors for each outer fold
for h, error in zip(hidden_units_values, errors_outer):
    print(f'Mean Outer Fold Error with {h} hidden units: {error}')

# Print the error for each outer fold along with the corresponding optimal h value
for k, (h, error) in enumerate(zip(best_h_per_inner_fold, errors_outer), 1):
    print(f'Outer Fold {k}: Hidden Units = {h}, Error = {error}')
