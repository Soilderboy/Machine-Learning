"""
Let's think:
Decision tree - we need:
A. hyperparameter tuning: (exhaustive grid search - try all combinations, test on validation set, pick best)
    #criterion - how each works
        entropy - information gain, higher entropy = more mixed, want to reduce entropy
            H = - sum(p_i log2(p_i)) where p_i is proportion of class i in node
            works by calculating entropy, splitting on highest reduction in entropy aka highest information gain
            50% class 0, 50% class 1 -> max entropy
            90% class 0, 10% class 1 -> -(.9log2(.9) + .1log2(.1)) = 0.47 -> lower entropy
        gini - probability-based impurity => low gini means pure node
            G = 1 - sum(p_i^2)
            Pure = 1 - 1^2 = 0
            Mixed = 1 - (.5^2 + .5^2) = 0.5
            90/10 = 1 - (.9^2 + .1^2) = 0.18 -> lower gini
    1. what features to split on - gini, entropy
        min # of samples to split internal node - min_samples_split
            more samples required to split -> less complex tree, more generalization, less overfitting
        min # samples for leaf node - min_samples_leaf
    2. when to stop splitting - max depth
        no max depth means highest complexity (overfitting) 
    3. how to predict - majority class in leaf node
    4. missing values - ignore for now

Reporting - concise summary only
    not required to report full grid of results. Instead for each dataset, summarize
        hyperparameter ranges explored + justification
        total # of cnadidate models evaluated
        selected best hyperparamter configuration
        corresponding validation accuracy

        then discuss how model complexity changes as hyperparamaters vary

B. Best Model Selection
    choose  hyperparameter combination with highest validation accuracy + report configuration
C. Retraining and evaluation
    combine training/validation sets
    retrain classifier using selected hyperparameter configuration on this
    evaluate final model on test set
    report classification accuracy and f1 score

    Reflect: How does selected hyperparameters affect complexity and generalization performance?
        Specifically, bias/variance tradeoff for variables like max_depth, decreasing min samples_split/leaf
        Does no max depth consistently perform best or overfit?
        What patterns do I observe across datasets with different training sizes/clause complexity?
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import utils

def tune_evaluate_decision_tree(x_train, y_train, x_valid, y_valid):
    #define hyperparameter grid
    max_depths= [None, 5, 10]
    min_samples_splits = [2, 5, 10] #just three random values to explore
    min_samples_leafs = [1, 2, 4]
    best_accuracy = 0
    best_params = {}
    total_models = 0

    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                for criterion in ['gini', 'entropy']:
                    total_models += 1
                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_valid)
                    accuracy = np.mean(y_pred == y_valid)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'criterion': criterion
                    }
    return best_params, best_accuracy, total_models

def retrain_evaluate_decision_tree(x_train_valid, y_train_valid, x_test, y_test, best_params):
    model = DecisionTreeClassifier(**best_params, random_state=42)
    model.fit(x_train_valid, y_train_valid)
    y_pred = model.predict(x_test)
    results = utils.evaluate_model(y_pred, y_test)
    return results['accuracy'], results['f1_score']
