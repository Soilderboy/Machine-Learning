import nltk
#nltk.download('punkt') #download tokenizer models for word_tokenize
#nltk.download('stopwords') #download english stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import csv
import logisticRegression

"""
Implements methods to
    tune lambda and returns best lambda

    train and evaluate logistic regression model with given lambda and training/test data
        returns evaluation metrics (accuracy, precision, recall, f1 score)

    save results in csv
"""
class Evaluation:
    def __init__(self):
        pass

    #tune lambda using validation set, return best lambda
    def tune_lambda(self, X_train, y_train, X_val, y_val, lambda_values, gd_variant='batch'):
        best_lambda = None
        best_accuracy = 0
        for lambda_val in lambda_values:
            model = logisticRegression.LogisticRegression(learning_rate=0.01, max_iterations=500, regularization_lambda=lambda_val)
            model.fit(X_train, y_train, gd_variant=gd_variant)
            metrics = model.evaluate(X_val, y_val)
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_lambda = lambda_val
        return best_lambda
    
    #train final model with given lambda and evaluate on test set, return metrics (should use best lambda from tuning)
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, lambda_val, gd_variant='batch', max_iterations=500):
        model = logisticRegression.LogisticRegression(learning_rate=0.01, max_iterations=max_iterations, regularization_lambda=lambda_val)
        if gd_variant == 'stochastic':
            model.fit(X_train, y_train, gd_variant='stochastic', learning_rate=0.001) #use smaller learning rate to converge
        else:
            model.fit(X_train, y_train, gd_variant)
        metrics = model.evaluate(X_test, y_test)
        return metrics
    
    #save results in csv
    def save_results(self, results, csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'representation', 'gd_variant', 'lambda', 'accuracy', 'precision', 'recall', 'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    #read csv, skip header, return X and y as numpy arrays
    def load_csv_data(self, csv_path):
        X = []
        y = []
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) #skip header
            for row in reader:
                features = list(map(int, row[:-1])) #convert feature values to integers (-1 removes label)
                label = int(row[-1])
                X.append(features)
                y.append(label)
        return np.array(X), np.array(y)
    