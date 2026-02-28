import dataPreparation
import os
import numpy as np
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

"""
#global variables

if __name__ == "__main__":
    #initalize data preparator
    preparator = dataPreparation.DataPreparator()
    #prepare datasets
    dataset_names = ['enron1', 'enron2', 'enron4']
    representations = ['bow', 'bernoulli']
    sets = ['train', 'test']

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

        





    #evaluate models
