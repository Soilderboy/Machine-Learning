"""
Transform raw collection of emails into a data matrix
    columns represent features (words from predefined vocabulary)
    rows represent examples (emails)
    each entry in matrix, quantifies presence of a word in a given email

There will be two approaches:
    bag of words: count how many times each word appears in email
        each email represented as a vector of word (counts)
            vector has length n, where n is vocabulary size


    bernoulli: binary representation, presence or absence of word in email
        same vector representation where vector is the vocabulary

First step - building the vocabulary:
    - read all emails
    - tokenize text into words
        - remove stop words (common words with no meaning)
        - build set of unique words across all emails
        - tokenizing helps remove punctuation and special characters, and lowercases all words
    - create mapping of each unique word to index in vector
    - construct data matrix where each row is an email and each column is word


2. Generating feature matrices for each representation
    bag of words
    - initialize matrix of zeros with shape (num_emails, vocab_size)
     - for each email, tokenize and count of each word

    bernoulli
    - initialize binary matrix of zeros with same shape
    - for each email, tokenize and set binary value to 1 if word is present

3. apply transformation to test set
    use vocabulary built from training set to transform test emails into same feature space
    
4. store datasets in csv format
    goal: 12 datasets (3 datasets x 2 representations x train and test)
        6 training sets (one BoW and one Bernoulli per dataset)
        6 test sets (same as above)

    each row = 1 email
    first w columns represent features (each word in vocab)
    last column contains label (spam or ham)
    include header with column names(word1, ... ,wordn, label)

dataset name format: dataset_representation_set.csv
    dataset(enron1, enron2, enron4)
    representation: bow or bernoulli
    set: train or test
submit this csv
"""

import os #file handling
import nltk #natural language processing
import numpy as np
from nltk.tokenize import word_tokenize #tokenization
from nltk.corpus import stopwords 
import csv
from collections import defaultdict

class DataPreparator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) #stopword set for english
    
    #make sure to only use this for training folder until vocabulary is built
    def read_emails(self, folder_path):
        #Read all emails from folder and return list of email texts
        emails = []
        for filename in os.listdir(folder_path): #os.listdir returns list of all files in directory
            if filename.endswith('.txt'): 
                #open file by passing full path, read content, and append to emails list
                with open(os.path.join(folder_path, filename), 'r', encoding='latin-1') as file:
                    emails.append(file.read())
        return emails 

    def preprocess_text(self, text):
        #tokenize text, remove stop words, return list of clean words
        tokens = word_tokenize(text.lower()) #tokenize and lowercase
        #keep alphabetic tokens not in stop words, remove punctuation and special characters
        words = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                words.append(token)
        return words
    
    def build_vocabulary(self, training_emails): 
        #Create vocabulary from training email, return mapping of word to index
        vocab = set()
        for email in training_emails:
            words = self.preprocess_text(email)
            vocab.update(words)
        vocab = sorted(vocab) #alphabetical order
        vocabMapping = {}
        for index, word in enumerate(vocab):#enumerate gives index and word for a set
            vocabMapping[word] = index
        return vocabMapping
    
    def generate_bow_matrix(self, emails, vocabMapping):
        #generate bag of words matrix for given emails and vocabulary mapping
        #returns matrix
        numEmails = len(emails)
        vocabSize = len(vocabMapping)
        bowMatrix = np.zeros((numEmails, vocabSize), dtype=int) #intialize matrix of zeros
        for i, email in enumerate(emails):
            words = self.preprocess_text(email)
            for word in words:
                if word in vocabMapping:
                    index = vocabMapping[word]
                    bowMatrix[i][index] += 1
        return bowMatrix
    
    def generate_bernoulli_matrix(self, emails, vocabMapping):
        #generate bernoulli matrix
        #given emails and vocab mapping, returns binary matrix
        numEmails = len(emails)
        vocabSize = len(vocabMapping)
        bernoulliMatrix = np.zeros((numEmails, vocabSize), dtype=int) #dtype means data type
        for i, email in enumerate(emails):
            words = self.preprocess_text(email)
            for word in words:
                if word in vocabMapping:
                    index = vocabMapping[word]
                    bernoulliMatrix[i][index] = 1
        return bernoulliMatrix
    
    def save_to_csv(self, matrix, labels, vocabMapping, filename):
        #combine feature matrix and labels, save to csv
        #create header
        header = sorted(vocabMapping.keys()) + ['label']
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i in range(len(matrix)):
                row = list(matrix[i]) + [labels[i]]
                writer.writerow(row)
    
    
    