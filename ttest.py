# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:18:49 2023

@author: Giulia
"""

from scipy import stats

# Mean-squared-errors dei due modelli
baseline = [0.84099,1.43333,0.96292 ,0.89725,1.09639, 0.94959, 0.71346  ,0.75967,0.82080, 0.81678]  # Esempio di MSE del primo modello
linear_regression = [0.89482, 1.39852, 0.91020, 0.85655, 0.97932, 0.89281, 0.61520, 0.74431, 0.58619, 0.67848]  # Esempio di MSE del secondo modello
ann = [0.86037, 1.08242, 0.78447,  0.81192,  0.84530, 0.88230, 0.76571, 0.82906, 0.73398, 0.79946]


differencesBL = [baseline[i] - linear_regression[i] for i in range(len(baseline))]
t_statistic, p_value = stats.ttest_rel(baseline, linear_regression)
print("Baseline-LinearModel")
print(f"T-Statistic : {t_statistic}")
print(f"P-Value: {p_value}")


differencesAL = [linear_regression[i] - ann[i] for i in range(len(linear_regression))]
t_statistic, p_value = stats.ttest_rel(linear_regression, ann)
print("ANN-LinearModel")
print(f"T-Statistic : {t_statistic}")
print(f"P-Value: {p_value}")

differencesAL = [baseline[i] - ann[i] for i in range(len(baseline))]
t_statistic, p_value = stats.ttest_rel(baseline, ann)
print("ANN-Baseline")
print(f"T-Statistic : {t_statistic}")
print(f"P-Value: {p_value}")