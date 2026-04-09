"""
Same as the other two. 
n_estimators = [10, 20, 50, 100]
max_features = [None, 'sqrt', 'log2']
max_depth = [None, 5, 10]

Retraining and Evaluation = combine train/valid sets, retrain final model, report accuracy/F1

max_features =how many features reach tree may consider when searching for best split
    restricting features reduces correlation between trees
    since variance reduction depends on averaging relatively uncorrelated models, stronger feature subsampling can sometimes improve generalization

reflection:
    does validation performance consistently improve with num estimators?
    what about changing max_features?
    tradeoff between feature restriction and model strength?
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import utils

def tune_evaluate_random_forest(x_train, y_train, x_valid, y_valid):
    n_estimators_options = [10, 20, 50, 100]
    max_features_options = [None, 'sqrt', 'log2'] #sqrt means sqrt(num_features) -> same for log2
    max_depth_options = [None, 5, 10]
    best_accuracy = 0
    best_params = {}
    total_models = 0

    for n_estimators in n_estimators_options:
        for max_features in max_features_options:
            for max_depth in max_depth_options:
                total_models += 1
                model = RandomForestClassifier(n_estimators=n_estimators, max_features = max_features, max_depth=max_depth, random_state=42, n_jobs=-1)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_valid)
                accuracy = np.mean(y_pred == y_valid)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth
                    }
    return best_params, best_accuracy, total_models

def retrain_evaluate_random_forest(x_train_valid, y_train_valid, x_test, y_test, best_params):
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(x_train_valid, y_train_valid)
    y_pred = model.predict(x_test)
    results = utils.evaluate_model(y_pred, y_test)
    return results['accuracy'], results['f1_score']