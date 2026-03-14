import dataPreparation
import os
import numpy as np
import logisticRegression
import evaluation
import naiveBayes
import csv
#import evaluation

"""
Main script to run data preparation and model evaluation

for now:
    initalize data preparator
    prepare datasets
        loop through each dataset (enron#)
            read training emails
            build vocabulary from training emails
            generate feature matrices for training/test
            add label to matrices
    evaluate models
#logistic regression
Load data
    read csv files into numpy arrays
    separate features and labels

For each dataset - representation (bow, bernoulli)
    load training data -> extract X_train and y_train
    load test csv -> extract X_test and y_test
    split x_tain/y_train into train/validation (70/30)

For each GD variant(batch, mini-batch, stochastic)
    (implement this in evaluation.py)
    lambda tuning loop (test each lambda in {.01, .1, 1, 10})
        create LogisticRegression with lambda
        train on 70%
        evaluate on 30%
    pick lambda with best validation accuracy

    final training:
        create new LogisticRegression with best lambda
        train on full training set (100%)
        evaluate on test set
        record metrics (accuracy, precision, recall, f1)
collect results
    3 datasets x 2 representations x 3 GD variants = 18 models
    each row: Dataset, Representation, GD Variant, Best Lambda, Accuracy, Precision, Recall, F1
    save results to csv
    then print Table 1 with results
"""
#global variables

if __name__ == "__main__":
    #prepare datasets
    dataset_names = ['enron1', 'enron2', 'enron4']
    representations = ['bow', 'bernoulli']
    sets = ['train', 'test']
    #Flag to control whether to regenerate datasets from raw email or load from existing csv
    REGENERATE_DATA = False
    LOGISTICREGRESSION = False
    NAIVEBAYES = False
    PRINTRESULTS = True
    
    if REGENERATE_DATA:
        #initalize data preparator
        preparator = dataPreparation.DataPreparator()


        for dataset in dataset_names:
            #read training emails
            train_folder = os.path.join('dataset', f'{dataset}_train', dataset, 'train')
            #if folder is ham, or spam, assign label 0 or 1
            label = 0 if 'ham' in train_folder else 1
            spam_emails = preparator.read_emails(os.path.join(train_folder, 'spam'))
            ham_emails = preparator.read_emails(os.path.join(train_folder, 'ham'))
            #build vocab
            vocab = preparator.build_vocabulary(spam_emails + ham_emails)
            #generate feature matrices for training/test
            for representation in representations:
                #training set
                if representation == 'bow':
                    train_matrix = preparator.generate_bow_matrix(spam_emails + ham_emails, vocab)
                else:
                    train_matrix = preparator.generate_bernoulli_matrix(spam_emails + ham_emails, vocab)
                #get label by identifying the folder (spam or ham) each email is in
                labels = [1] * len(spam_emails) + [0] * len(ham_emails)
                
                train_csv_path = f"{dataset}_{representation}_train.csv"
                preparator.save_to_csv(train_matrix, labels, vocab, train_csv_path)
            """
            read test spam/ham
            generate test matrices using same vocab built from training set
                if word in vocab, get index and set value in matrix
                if not in vocab, ignore word (treat as unseen word) -> done in dataPreparation functions

            save test csvs
            """
            test_folder = os.path.join('dataset', f'{dataset}_test', dataset, 'test')
            test_spam_emails = preparator.read_emails(os.path.join(test_folder, 'spam'))
            test_ham_emails = preparator.read_emails(os.path.join(test_folder, 'ham'))
            for representation in representations:
                test_labels = [1] * len(test_spam_emails) + [0] * len(test_ham_emails)
                #reutilize same vocab
                if representation == 'bow':
                    test_matrix = preparator.generate_bow_matrix(test_spam_emails + test_ham_emails, vocab)
                else:
                    test_matrix = preparator.generate_bernoulli_matrix(test_spam_emails + test_ham_emails, vocab)
                
                test_csv_path = f"{dataset}_{representation}_test.csv"
                preparator.save_to_csv(test_matrix, test_labels, vocab, test_csv_path)
    if LOGISTICREGRESSION:
        #logistic regression
        results = []
        evaluator = evaluation.Evaluation()
        #load data
        for dataset in dataset_names:
            for representation in representations:
                #load training csv/test csv
                train_file = f"{dataset}_{representation}_train.csv"
                test_file = f"{dataset}_{representation}_test.csv"
                X_train, y_train = evaluation.Evaluation().load_csv_data(train_file)
                X_test, y_test = evaluation.Evaluation().load_csv_data(test_file)

                #split training data into train/validation
                num_train_samples = X_train.shape[0]
                indices = np.arange(num_train_samples) #arange creates array of indices from 0 to n
                np.random.shuffle(indices) #random.shuffle shuffles indices in place
                split_point = int(0.7 * num_train_samples)
                train_indices = indices[:split_point]
                val_indices = indices[split_point:]
                X_train_split = X_train[train_indices]
                y_train_split = y_train[train_indices]
                X_val_split = X_train[val_indices]
                y_val_split = y_train[val_indices]

                #for each GD variant, tune lambda, train final model, evaluate on test set, record metrics
                gd_variants = ['batch', 'mini-batch', 'stochastic']
                lambda_values = [0.01, 0.1, 1, 10]

                for gd_variant in gd_variants:
                    best_lambda = evaluator.tune_lambda(X_train_split, y_train_split, X_val_split, y_val_split, lambda_values, gd_variant)
                    metrics = evaluator.train_and_evaluate(X_train, y_train, X_test, y_test, best_lambda, gd_variant, max_iterations=500) 
                    #record results in list of dicts
                    result = {
                        'dataset': dataset,
                        'representation': representation,
                        'gd_variant': gd_variant,
                        'lambda': best_lambda,
                        **metrics
                    }
                    print(result)
                    #append result to results list
                    results.append(result)
        fieldnames = ['dataset', 'representation', 'gd_variant', 'lambda', 'accuracy', 'precision', 'recall', 'f1']
        evaluator.save_results(results, 'logistic_regression_results.csv', fieldnames)

    if NAIVEBAYES:
        #multinomial naive bayes
        results = []
        evaluator = evaluation.Evaluation()
        for dataset in dataset_names:
            #only bow
            train_file = f"{dataset}_bow_train.csv"
            test_file = f"{dataset}_bow_test.csv"
            X_train, y_train = evaluator.load_csv_data(train_file)
            X_test, y_test = evaluator.load_csv_data(test_file)

            #train model and evaluate on test set
            model = naiveBayes.MultinomialNaiveBayes()
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            result = {
                'dataset': dataset,
                'representation': 'bow',
                **metrics
            }
            print(result)
            results.append(result)

            #bernoulli nb
            train_file = f"{dataset}_bernoulli_train.csv"
            test_file = f"{dataset}_bernoulli_test.csv"
            X_train, y_train = evaluator.load_csv_data(train_file)
            X_test, y_test = evaluator.load_csv_data(test_file)

            model = naiveBayes.BernoulliNaiveBayes()
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            result = {
                'dataset': dataset,
                'representation': 'bernoulli',
                **metrics
            }
            print(result)
            results.append(result)
        fieldnames = ['dataset', 'representation', 'accuracy', 'precision', 'recall', 'f1']
        evaluator.save_results(results, 'naive_bayes_nb_results.csv', fieldnames)

    #print results in table format
    if PRINTRESULTS:
        import pandas as pd
        
        #logistic regression results
        print("\n" + "="*100)
        print("Table 1: Logistic Regression Results")
        print("="*100)
        lr_df = pd.read_csv('logistic_regression_results.csv')
        print(lr_df.to_string(index=False))
        
        #naive bayes results
        print("\n" + "="*100)
        print("Table 2: Naive Bayes results for different variants")
        print("="*100)
        nb_df = pd.read_csv('naive_bayes_nb_results.csv')
        print(nb_df.to_string(index=False))
        
