"""
Important notes:
    do everything in log space to avoid underflow. thus do not exponentiate log-probs during test time

Bayes Theorem: P(Y|X) = P(X|Y) * P(Y) / P(X)
    P(Y|X) = posterior probability of class Y given features X (what we want to predict)
    P(X|Y) = likelihood of features X given class Y (estimated from training data)
    P(Y) = prior probability of class Y (estimated from training data)
    P(X) = evidence (ignore since it's constant for all classes)

Multinomial NB (multinomial means features are counts)
    - assumes features are generated from word counts in email
    - estimates P(Y) and P(X|Y) from training data, uses Bayes to compute P(Y|X) for prediction
    - uses BOW
overview:
    training phase:
        - build vocabulary
        - for each class, count number of emails in class,
            count number of times each word appears in emails of that class
        - estimate P(Y) = count(class) / total emails
        (laplace smoothing handles zero counts. a=1 in this case)
        - estimate P(X|Y) = (count(word in class) + alpha) / (total words in class + alpha * vocab_size)

    testing phase:
        for new email, compute log P(Y|X) for each class using log P(Y) + sum(log P(X_i|Y)) for all words in email
        predict class with highest log P(Y|X)




Bernoulli NB
    - features are binary
"""
from collections import defaultdict
import os
import nltk
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self):
        pass
    
    #given labels, compute prior probabilities P(Y) for each calss and return as dict {class: prior_prob}
    def prior(self, y):
        #dictionary to store counts of each class
        #prior -> P(Y) = count(class) / total samples
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        total_samples = len(y)
        prior_dict = {}
        for label in class_counts:
            prior_dict[label] = np.log(class_counts[label] / total_samples)
        return prior_dict
    
    #given training data, estimate likelihood P(X|Y) for each class and each feature, return as dict {class: likelihood_vector}
    def likelihood(self, X, y):
        #count word counts per class
        #X = count matrix where rows are emails and columns are words in vocab, y = labels for each email
        #X.shape = (num_samples, num_features), y.shape = (num_samples,)
        class_word_counts = defaultdict(lambda: np.zeros(X.shape[1])) #dict of counts for each class, initialized to zero vector of vocab size
        class_counts = defaultdict(int) #count of samples in each class
        for i in range(X.shape[0]):
            label = y[i]
            class_word_counts[label] += X[i] #add word counts for this sample to class total
            class_counts[label] += 1
        #apply laplace smoothing
        likelihood_dict = {}
        vocab_size = X.shape[1]
        alpha = 1
        #for each class, compute likelihood vector of words, store as log probs
        for label in class_word_counts:
            #P(X|Y) = (count(word in class) + alpha) / (total words in class + alpha * vocab_size)
            total_words_in_class = np.sum(class_word_counts[label])
            likelihood_dict[label] = np.log((class_word_counts[label] + alpha) / (total_words_in_class + alpha * vocab_size))
        return likelihood_dict

    #Store prior and likelihood from training data, then return optimized weights log of prior and likelihood for each class
    def fit(self, X, y):
        self.prior_dict = self.prior(y)
        self.likelihood_dict = self.likelihood(X, y)
        return self.prior_dict, self.likelihood_dict
    
    #predict class for new data point using log P(Y) + sum(log likelihood) for each class, return predicted class
    #we're basically computing log posterior for each class and choosing class with highest log posterior
    def predict(self, X):
        #X is count vector for new email
        predictions = []
        for i in range(X.shape[0]): #for each email
            log_probs = {}
            for label in self.prior_dict: #go through each class (2)
                #log posterior = log P(Y) + second part
                #second part = all features in email, so sum of likelihoods for words present, weighted by count of word (multiplication)
                log_probs[label] = self.prior_dict[label] + np.sum(X[i] * self.likelihood_dict[label]) 
            
            predicted_label = max(log_probs, key=log_probs.get)
            predictions.append(predicted_label)
        return np.array(predictions)
    
    #given test data and true labels, compute metrics, and return as dict
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        if np.sum(y_pred) == 0:
            precision = 0
        else:
            #true positives/predicted positives -> how many predicted spam were actually spam
            precision = np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1)
        if np.sum(y) == 0:
            recall = 0
        else:
            #true positives/actual positives -> how many of actual spam emails did we correctly identify
            recall = np.sum((y_pred==1) & (y==1))/np.sum(y==1)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision*recall) / (precision + recall)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

class BernoulliNaiveBayes:
    def __init__(self):
        pass