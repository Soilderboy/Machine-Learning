"""
Repeat same as decision_tree.py
    but using baggiing classifier with decisiontreeclassifier as base estimator

Hyperparameter tuning:
    estimators (n_estimators) = [10, 20, 50, 100]
    max_depth = [None, 5, 10]

    design small but meaningful grid to explore different ensembles (n_estimators and max_depth)
    for each dataset, report training and validation accuracy for all combinations of estimators and max depth in a table

Retraining and evaluation: using selected configuration, combine train/valid sets and retrain final model
    Evaluate model on test set and report accuracy and f1 score in a table for all datasets, max depth, and n_estimators

Reflect:
    As num estimators increases, does validation performance consistently improve?
    Do deeper trees always perform better in bagging?
    What does this suggest about interaction between ensemble size and base learner complexity?
"""

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import utils

def tune_evaluate_bagging(x_train, y_train, x_valid, y_valid):
    n_estimators_options = [10, 20, 50, 100]
    max_depth_options = [None, 5, 10]
    best_accuracy = 0
    best_params = {}
    total_models = 0
    results_grid = {} #track all results for table 1
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            total_models += 1
            base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model = BaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_valid)
            accuracy = np.mean(y_pred == y_valid)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
            results_grid[(n_estimators, max_depth)] = accuracy
    return best_params, best_accuracy, total_models, results_grid

def retrain_evaluate_bagging(x_train_valid, y_train_valid, x_test, y_test, best_params):
    base_estimator = DecisionTreeClassifier(max_depth=best_params['max_depth'], random_state=42)
    model = BaggingClassifier(estimator=base_estimator, n_estimators=best_params['n_estimators'], random_state=42, n_jobs=-1)
    model.fit(x_train_valid, y_train_valid)
    y_pred = model.predict(x_test)
    results = utils.evaluate_model(y_pred, y_test)
    return results['accuracy'], results['f1_score']


