"""
Main file, every other file are just classes/functions
"""

import utils
import decision_tree
import bagging
import random_forest
import adaboost
import numpy as np
flag_DTC = False
flag_Bagged = True
flag_PrintTableBagged = True
flag_RF = False
flag_Boost = False
def main():
    base_path = 'project2_data\\all_data\\'
    dataset_combinations = utils.get_dataset_combinations()
    for clauses, data_size in dataset_combinations:

        print(f"Processing dataset with {clauses} clauses and {data_size} data points...")
        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_train_valid_test(clauses, data_size, base_path)
        #Combined set of train/validation for retraining final model
        x_train_valid = np.concatenate((x_train, x_valid), axis=0) #axis 0 to stack vertically
        y_train_valid = np.concatenate((y_train, y_valid), axis=0)
        # decision tree classifier
        if flag_DTC:
            best_params, best_accuracy, total_models = decision_tree.tune_evaluate_decision_tree(x_train, y_train, x_valid, y_valid)
            print(f"Best hyperparameters: {best_params}, Validation Accuracy: {best_accuracy:.4f}, Total Models Evaluated: {total_models}")
            test_accuracy, test_f1 = decision_tree.retrain_evaluate_decision_tree(x_train_valid, y_train_valid, x_test, y_test, best_params)
            print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}\n")

        #bagged decision tree classifiers
        if flag_Bagged:
            best_params_bagging, best_accuracy_bagging, total_models_bagging, results_grid_bagging = bagging.tune_evaluate_bagging(x_train, y_train, x_valid, y_valid)
            print(f"Best hyperparameters for bagging: {best_params_bagging}, Validation Accuracy: {best_accuracy_bagging:.4f}, Total Models Evaluated: {total_models_bagging}")
            test_accuracy_bagging, test_f1_bagging = bagging.retrain_evaluate_bagging(x_train_valid, y_train_valid, x_test, y_test, best_params_bagging)
            print(f"Bagging Test Accuracy: {test_accuracy_bagging:.4f}, Bagging Test F1 Score: {test_f1_bagging:.4f}\n")
            #print table 1 for bagging results on different n_estimators and max_depth combinations
            if flag_PrintTableBagged:
                print("Validation Accuracy for Bagging Classifier (n_estimators, max_depth):")
                for (n_estimators, max_depth), accuracy in results_grid_bagging.items():
                    print(f"n_estimators: {n_estimators}, max_depth: {max_depth} -> Validation Accuracy: {accuracy:.4f}")
                print("\n")
        
        #random forest classifiers
        if flag_RF:
            best_params_rf, best_accuracy_rf, total_models_rf = random_forest.tune_evaluate_random_forest(x_train, y_train, x_valid, y_valid)
            print(f"Best hyperparameters for random forest: {best_params_rf}, Validation Accuracy: {best_accuracy_rf:.4f}, Total Models Evaluated: {total_models_rf}")
            test_accuracy_rf, test_f1_rf = random_forest.retrain_evaluate_random_forest(x_train_valid, y_train_valid, x_test, y_test, best_params_rf)
            print(f"Random Forest Test Accuracy: {test_accuracy_rf:.4f}, Random Forest Test F1 Score: {test_f1_rf:.4f}\n")

        #adaboost classifiers
        if flag_Boost:
            best_params_boost, best_accuracy_boost, total_models_boost = adaboost.tune_evaluate_adaboost(x_train, y_train, x_valid, y_valid)
            print(f"Best hyperparameters for adaboost: {best_params_boost}, Validation Accuracy: {best_accuracy_boost:.4f}, Total Models Evaluated: {total_models_boost}")
            test_accuracy_boost, test_f1_boost = adaboost.retrain_evaluate_adaboost(x_train_valid, y_train_valid, x_test, y_test, best_params_boost)
            print(f"AdaBoost Test Accuracy: {test_accuracy_boost:.4f}, AdaBoost Test F1 Score: {test_f1_boost:.4f}\n")

if __name__ == "__main__":
    main()