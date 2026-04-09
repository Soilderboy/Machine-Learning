"""
n_estimators = [10, 20, 50, 100]
max_depth = [1, 5, 10] - idea is to not have strong learners so no None

Conceptual note: adaboost, models are added sequentailly, each new learner places more emphasis on previous misclassifications.
    Increasing n_estimators increases overall model capacity, while increasing tree depth makes each weak learner stronger.

Reflection:
    Does increasing # estimaros always improve validation performance?
    Do deeper base learners necessarily perform better in boosting?
    Or can overly strong learners lead to overfitting?
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import utils

def tune_evaluate_adaboost(x_train, y_train, x_valid, y_valid):
    n_estimators_options = [10, 20, 50, 100]
    max_depth_options = [1, 5, 10]
    best_accuracy = 0
    best_params = {}
    total_models = 0

    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            total_models += 1
            base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=42) #can't parallelize
            model.fit(x_train, y_train)
            y_pred = model.predict(x_valid)
            accuracy = np.mean(y_pred == y_valid)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
    return best_params, best_accuracy, total_models

def retrain_evaluate_adaboost(x_train_valid, y_train_valid, x_test, y_test, best_params):
    base_estimator = DecisionTreeClassifier(max_depth=best_params['max_depth'], random_state=42)
    model = AdaBoostClassifier(estimator=base_estimator, n_estimators=best_params['n_estimators'], random_state=42)
    model.fit(x_train_valid, y_train_valid)
    y_pred = model.predict(x_test)
    results = utils.evaluate_model(y_pred, y_test)
    return results['accuracy'], results['f1_score']