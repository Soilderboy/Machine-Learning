"""
Utility functions needed
1.
load_dataset(filepath)
    read csv file
    split into features x and labels y
    return x,y
    remember that last column as label

2.
get dataset combinations()
    return all 15 dataset combinations (clauses,datasize)
        clauses: 300, 500, 1000, 1500, 1800
        data size: 100, 1000, 5000

3.
load train validation test(caluses, data size, base path)
    given clauses and data size, load all three datasets (train, valid, test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test
    ---purpose: not mixing datasets---

4.
evaluate model y_pred, y_true
    calculate accuracy, precision, recall, f1 score
    return as dict
"""

import csv
import os
import numpy as np

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) #skip header
        data = list(reader)
    data = np.array(data) 
    x = data[:, :-1].astype(float) #all columns except last -> float to help normalize/scale features
    y = data[:, -1].astype(int) #last column -> labels as int
    return x, y

def get_dataset_combinations():
    clauses = [300, 500, 1000, 1500, 1800]
    data_sizes = [100, 1000, 5000]
    combinations = []
    for clause in clauses:
        for size in data_sizes:
            combinations.append((clause, size))
    return combinations

def load_train_valid_test(clauses, data_size, base_path):
    train_path = os.path.join(base_path, f'train_c{clauses}_d{data_size}_train.csv')
    valid_path = os.path.join(base_path, f'valid_c{clauses}_d{data_size}_valid.csv')
    test_path = os.path.join(base_path, f'test_c{clauses}_d{data_size}_test.csv')
    
    x_train, y_train = load_dataset(train_path)
    x_valid, y_valid = load_dataset(valid_path)
    x_test, y_test = load_dataset(test_path)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def evaluate_model(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return { 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1 }
