important things to keep in mind

don't mix datasets
(don't train on train_c500_d100 and test on test_c500_d5000)
# The Following Contains all Results from Terminal for Table 2

# Results from Decision Tree Classifier

C:\Users\DragonFire\Desktop\Coding\Machine Learning\proj2>python main.py
Processing dataset with 300 clauses and 100 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.6533, Total Models Evaluated: 72
Test Accuracy: 0.6281, Test F1 Score: 0.6263

Processing dataset with 300 clauses and 1000 data points...
Best hyperparameters: {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'}, Validation Accuracy: 0.6493, Total Models Evaluated: 72
Test Accuracy: 0.6723, Test F1 Score: 0.7093

Processing dataset with 300 clauses and 5000 data points...
Best hyperparameters: {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.7382, Total Models Evaluated: 72
Test Accuracy: 0.7803, Test F1 Score: 0.7884

Processing dataset with 500 clauses and 100 data points...
Best hyperparameters: {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 4, 'criterion': 'entropy'}, Validation Accuracy: 0.6482, Total Models Evaluated: 72
Test Accuracy: 0.6734, Test F1 Score: 0.6948

Processing dataset with 500 clauses and 1000 data points...
Best hyperparameters: {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'}, Validation Accuracy: 0.6938, Total Models Evaluated: 72
Test Accuracy: 0.6818, Test F1 Score: 0.6867

Processing dataset with 500 clauses and 5000 data points...
Best hyperparameters: {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.7566, Total Models Evaluated: 72
Test Accuracy: 0.7886, Test F1 Score: 0.7981

Processing dataset with 1000 clauses and 100 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.7588, Total Models Evaluated: 72
Test Accuracy: 0.7337, Test F1 Score: 0.7440

Processing dataset with 1000 clauses and 1000 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 4, 'criterion': 'entropy'}, Validation Accuracy: 0.8064, Total Models Evaluated: 72
Test Accuracy: 0.7889, Test F1 Score: 0.7965

Processing dataset with 1000 clauses and 5000 data points...
Best hyperparameters: {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.8435, Total Models Evaluated: 72
Test Accuracy: 0.8601, Test F1 Score: 0.8652

Processing dataset with 1500 clauses and 100 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 4, 'criterion': 'entropy'}, Validation Accuracy: 0.8744, Total Models Evaluated: 72
Test Accuracy: 0.8794, Test F1 Score: 0.8800

Processing dataset with 1500 clauses and 1000 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.9165, Total Models Evaluated: 72
Test Accuracy: 0.9270, Test F1 Score: 0.9279

Processing dataset with 1500 clauses and 5000 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.9482, Total Models Evaluated: 72
Test Accuracy: 0.9565, Test F1 Score: 0.9566

Processing dataset with 1800 clauses and 100 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.9749, Total Models Evaluated: 72
Test Accuracy: 0.9146, Test F1 Score: 0.9187

Processing dataset with 1800 clauses and 1000 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}, Validation Accuracy: 0.9710, Total Models Evaluated: 72
Test Accuracy: 0.9780, Test F1 Score: 0.9781

Processing dataset with 1800 clauses and 5000 data points...
Best hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 4, 'criterion': 'entropy'}, Validation Accuracy: 0.9842, Total Models Evaluated: 72
Test Accuracy: 0.9873, Test F1 Score: 0.9873

# Bagged Decision Tree Classifier
Processing dataset with 300 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 50, 'max_depth': 5}, Validation Accuracy: 0.7286, Total Models Evaluated: 12
Bagging Test Accuracy: 0.7739, Bagging Test F1 Score: 0.7867

Processing dataset with 300 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.8689, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8904, Bagging Test F1 Score: 0.8922

Processing dataset with 300 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.8985, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9185, Bagging Test F1 Score: 0.9230

Processing dataset with 500 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.7990, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8543, Bagging Test F1 Score: 0.8599

Processing dataset with 500 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': 10}, Validation Accuracy: 0.8784, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8714, Bagging Test F1 Score: 0.8705

Processing dataset with 500 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9129, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9432, Bagging Test F1 Score: 0.9437

Processing dataset with 1000 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.8593, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8894, Bagging Test F1 Score: 0.8866

Processing dataset with 1000 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9460, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9430, Bagging Test F1 Score: 0.9436

Processing dataset with 1000 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9523, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9636, Bagging Test F1 Score: 0.9635

Processing dataset with 1500 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 50, 'max_depth': 5}, Validation Accuracy: 0.9799, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9698, Bagging Test F1 Score: 0.9706

Processing dataset with 1500 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9795, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9825, Bagging Test F1 Score: 0.9824

Processing dataset with 1500 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9889, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9904, Bagging Test F1 Score: 0.9904

Processing dataset with 1800 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 10, 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9698, Bagging Test F1 Score: 0.9700

Processing dataset with 1800 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9910, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9925, Bagging Test F1 Score: 0.9925

Processing dataset with 1800 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9966, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9976, Bagging Test F1 Score: 0.9976

# Random Forest Classifier

Processing dataset with 300 clauses and 100 data points...
Best hyperparameters for random forest: {'n_estimators': 50, 'max_features': 'sqrt', 'max_depth': 5}, Validation Accuracy: 0.7186, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.7186, Random Forest Test F1 Score: 0.7255

Processing dataset with 300 clauses and 1000 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': None, 'max_depth': None}, Validation Accuracy: 0.8819, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.8839, Random Forest Test F1 Score: 0.8866

Processing dataset with 300 clauses and 5000 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': 10}, Validation Accuracy: 0.8970, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9091, Random Forest Test F1 Score: 0.9118

Processing dataset with 500 clauses and 100 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': 5}, Validation Accuracy: 0.8442, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.8543, Random Forest Test F1 Score: 0.8543

Processing dataset with 500 clauses and 1000 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'log2', 'max_depth': 5}, Validation Accuracy: 0.9265, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9410, Random Forest Test F1 Score: 0.9413

Processing dataset with 500 clauses and 5000 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'log2', 'max_depth': 10}, Validation Accuracy: 0.9402, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9566, Random Forest Test F1 Score: 0.9573

Processing dataset with 1000 clauses and 100 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': None}, Validation Accuracy: 0.9698, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9749, Random Forest Test F1 Score: 0.9746

Processing dataset with 1000 clauses and 1000 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'log2', 'max_depth': 5}, Validation Accuracy: 0.9950, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9910, Random Forest Test F1 Score: 0.9910

Processing dataset with 1000 clauses and 5000 data points...
Best hyperparameters for random forest: {'n_estimators': 100, 'max_features': 'log2', 'max_depth': None}, Validation Accuracy: 0.9961, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9949, Random Forest Test F1 Score: 0.9949

Processing dataset with 1500 clauses and 100 data points...
Best hyperparameters for random forest: {'n_estimators': 50, 'max_features': 'sqrt', 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 36
Random Forest Test Accuracy: 1.0000, Random Forest Test F1 Score: 1.0000

Processing dataset with 1500 clauses and 1000 data points...
Best hyperparameters for random forest: {'n_estimators': 50, 'max_features': 'log2', 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 36
Random Forest Test Accuracy: 1.0000, Random Forest Test F1 Score: 1.0000

Processing dataset with 1500 clauses and 5000 data points...
Best hyperparameters for random forest: {'n_estimators': 50, 'max_features': 'log2', 'max_depth': 10}, Validation Accuracy: 0.9999, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9999, Random Forest Test F1 Score: 0.9999

Processing dataset with 1800 clauses and 100 data points...
Best hyperparameters for random forest: {'n_estimators': 20, 'max_features': 'log2', 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9950, Random Forest Test F1 Score: 0.9949

Processing dataset with 1800 clauses and 1000 data points...
Best hyperparameters for random forest: {'n_estimators': 20, 'max_features': 'sqrt', 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9995, Random Forest Test F1 Score: 0.9995

Processing dataset with 1800 clauses and 5000 data points...
Best hyperparameters for random forest: {'n_estimators': 20, 'max_features': 'log2', 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 36
Random Forest Test Accuracy: 0.9999, Random Forest Test F1 Score: 0.9999

# Boosted Classifiers
C:\Users\DragonFire\Desktop\Coding\Machine Learning\proj2>python main.py
Processing dataset with 300 clauses and 100 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 1}, Validation Accuracy: 0.7437, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.8241, AdaBoost Test F1 Score: 0.8309

Processing dataset with 300 clauses and 1000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.9035, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9380, AdaBoost Test F1 Score: 0.9386

Processing dataset with 300 clauses and 5000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.9745, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9845, AdaBoost Test F1 Score: 0.9846

Processing dataset with 500 clauses and 100 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 1}, Validation Accuracy: 0.7889, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.8693, AdaBoost Test F1 Score: 0.8725

Processing dataset with 500 clauses and 1000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.9430, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9770, AdaBoost Test F1 Score: 0.9770

Processing dataset with 500 clauses and 5000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.9830, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9906, AdaBoost Test F1 Score: 0.9906

Processing dataset with 1000 clauses and 100 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 1}, Validation Accuracy: 0.9447, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9548, AdaBoost Test F1 Score: 0.9538

Processing dataset with 1000 clauses and 1000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.9905, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9910, AdaBoost Test F1 Score: 0.9910

Processing dataset with 1000 clauses and 5000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.9981, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9987, AdaBoost Test F1 Score: 0.9987

Processing dataset with 1500 clauses and 100 data points...
Best hyperparameters for adaboost: {'n_estimators': 50, 'max_depth': 1}, Validation Accuracy: 0.9849, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9899, AdaBoost Test F1 Score: 0.9900

Processing dataset with 1500 clauses and 1000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 1.0000, Total Models Evaluated: 12
AdaBoost Test Accuracy: 1.0000, AdaBoost Test F1 Score: 1.0000

Processing dataset with 1500 clauses and 5000 data points...
Best hyperparameters for adaboost: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 1.0000, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9999, AdaBoost Test F1 Score: 0.9999

Processing dataset with 1800 clauses and 100 data points...
Best hyperparameters for adaboost: {'n_estimators': 50, 'max_depth': 1}, Validation Accuracy: 0.9899, Total Models Evaluated: 12
AdaBoost Test Accuracy: 1.0000, AdaBoost Test F1 Score: 1.0000

Processing dataset with 1800 clauses and 1000 data points...
Best hyperparameters for adaboost: {'n_estimators': 20, 'max_depth': 5}, Validation Accuracy: 1.0000, Total Models Evaluated: 12
AdaBoost Test Accuracy: 0.9995, AdaBoost Test F1 Score: 0.9995

Processing dataset with 1800 clauses and 5000 data points...
Best hyperparameters for adaboost: {'n_estimators': 50, 'max_depth': 5}, Validation Accuracy: 1.0000, Total Models Evaluated: 12
AdaBoost Test Accuracy: 1.0000, AdaBoost Test F1 Score: 1.0000

# Table 1 for Validation Accuracy of Bagged Decision Trees for Different Values

C:\Users\DragonFire\Desktop\Coding\Machine Learning\proj2>python main.py
Processing dataset with 300 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 50, 'max_depth': 5}, Validation Accuracy: 0.7286, Total Models Evaluated: 12
Bagging Test Accuracy: 0.7739, Bagging Test F1 Score: 0.7867

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.6231
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.6382
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.6231
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.6633
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.6834
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.6633
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.7035
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.7286
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.7035
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.6834
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.6884
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.6834


Processing dataset with 300 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.8689, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8904, Bagging Test F1 Score: 0.8922

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.7589
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.8074
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.7589
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.7819
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.8289
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.7849
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.8449
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.8519
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.8464
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.8689
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.8534
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.8674


Processing dataset with 300 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.8985, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9185, Bagging Test F1 Score: 0.9230

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.8161
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.7877
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.8408
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.8564
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.7823
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.8595
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.8864
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.7938
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.8849
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.8985
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.7920
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.8943


Processing dataset with 500 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': 5}, Validation Accuracy: 0.7990, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8543, Bagging Test F1 Score: 0.8599

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.5980
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.6281
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.5980
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.6583
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.6734
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.6583
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.7186
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.7387
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.7186
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.7889
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.7990
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.7889


Processing dataset with 500 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': 10}, Validation Accuracy: 0.8784, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8714, Bagging Test F1 Score: 0.8705

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.7739
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.7989
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.7739
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.8199
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.8154
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.8249
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.8704
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.8309
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.8739
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.8749
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.8219
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.8784


Processing dataset with 500 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9129, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9432, Bagging Test F1 Score: 0.9437

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.8433
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.7596
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.8489
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.8819
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.7750
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.8817
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9023
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.7811
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9008
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9129
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.7823
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9089


Processing dataset with 1000 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.8593, Total Models Evaluated: 12
Bagging Test Accuracy: 0.8894, Bagging Test F1 Score: 0.8866

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.8342
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.8241
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.8342
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.8342
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.8241
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.8342
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.8442
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.8241
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.8442
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.8593
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.8492
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.8593


Processing dataset with 1000 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9460, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9430, Bagging Test F1 Score: 0.9436

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.8934
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.8754
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.8939
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9265
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.8909
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9275
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9335
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.8849
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9340
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9460
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.8984
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9455


Processing dataset with 1000 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9523, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9636, Bagging Test F1 Score: 0.9635

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.9195
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.8338
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.9179
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9336
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.8314
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9285
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9460
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.8397
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9434
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9523
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.8388
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9472


Processing dataset with 1500 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 50, 'max_depth': 5}, Validation Accuracy: 0.9799, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9698, Bagging Test F1 Score: 0.9706

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.9347
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.9347
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.9347
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9548
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.9548
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9548
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9749
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.9799
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9749
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9799
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.9799
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9799


Processing dataset with 1500 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9795, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9825, Bagging Test F1 Score: 0.9824

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.9720
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.9520
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.9720
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9760
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.9575
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9760
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9770
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.9565
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9770
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9795
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.9605
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9795


Processing dataset with 1500 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9889, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9904, Bagging Test F1 Score: 0.9904

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.9797
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.9364
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.9789
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9853
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.9431
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9839
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9881
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.9450
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9867
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9889
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.9455
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9880


Processing dataset with 1800 clauses and 100 data points...
Best hyperparameters for bagging: {'n_estimators': 10, 'max_depth': None}, Validation Accuracy: 1.0000, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9698, Bagging Test F1 Score: 0.9700

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 1.0000
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 1.0000
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 1.0000
n_estimators: 20, max_depth: None -> Validation Accuracy: 1.0000
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 1.0000
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 1.0000
n_estimators: 50, max_depth: None -> Validation Accuracy: 1.0000
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 1.0000
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 1.0000
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9950
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.9950
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9950


Processing dataset with 1800 clauses and 1000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9910, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9925, Bagging Test F1 Score: 0.9925

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.9820
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.9745
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.9820
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9900
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.9805
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9900
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9905
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.9810
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9905
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9910
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.9820
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9910


Processing dataset with 1800 clauses and 5000 data points...
Best hyperparameters for bagging: {'n_estimators': 100, 'max_depth': None}, Validation Accuracy: 0.9966, Total Models Evaluated: 12
Bagging Test Accuracy: 0.9976, Bagging Test F1 Score: 0.9976

Validation Accuracy for Bagging Classifier (n_estimators, max_depth):
n_estimators: 10, max_depth: None -> Validation Accuracy: 0.9945
n_estimators: 10, max_depth: 5 -> Validation Accuracy: 0.9739
n_estimators: 10, max_depth: 10 -> Validation Accuracy: 0.9942
n_estimators: 20, max_depth: None -> Validation Accuracy: 0.9954
n_estimators: 20, max_depth: 5 -> Validation Accuracy: 0.9806
n_estimators: 20, max_depth: 10 -> Validation Accuracy: 0.9952
n_estimators: 50, max_depth: None -> Validation Accuracy: 0.9962
n_estimators: 50, max_depth: 5 -> Validation Accuracy: 0.9802
n_estimators: 50, max_depth: 10 -> Validation Accuracy: 0.9963
n_estimators: 100, max_depth: None -> Validation Accuracy: 0.9966
n_estimators: 100, max_depth: 5 -> Validation Accuracy: 0.9801
n_estimators: 100, max_depth: 10 -> Validation Accuracy: 0.9963
