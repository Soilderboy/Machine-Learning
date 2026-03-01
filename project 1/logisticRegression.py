"""
Main goal: implement logistic regression without complexity of O(n^2 * d)
Run it optimally at O(n * d)
    - compute all predictions O(nd)
    - compute gradient O(nd)
        per iteration O(nd), for T iterations O(Tdn) with iteration cap at 500

D = dataset
    x_i = feature vector for email i
    y = label for email i (1 for spam)

w 
    bias = 1 for all emails
    w_j = weight for feature j (influence of feature j)

sigmoid
    z = w^T * x_I -> z[i] = dot(w, x[i]) 
    sigmoid(z) = 1 / (1 + exp(-z)) -> 
        output [0,1] and probability of being spam

log likelihood (score the model to improve weights)
    L(w) = sum(y[i] * logP(y[i]=1|x[i];w)) + (1-y[i]) * logP(y[i]=0|x[i];w))
    P(y[i]=1|x[i];w) = sigmoid(w^T * x[i])
    P(y[i]=0|x[i];w) = 1 - sigmoid(w^T * x[i])

    
    maximize L(w) -> minimize -L(w) -> minimize cost function l(w) = -L(w)

gradient descent update rule
    w = w - learning rate * gradient of cost function
    gradient = dL/dw = sum((sigmoid(w^T * x[i]) - y[i]) * x[i])
    g[j] += x[i][j] * (y[i] - ypred[i]) -> technically gradient ascent

Pseudocode given:
    Input: Data array X (size d x n), learning rate eta, max iterations T, regularization constant lambda
    Output: optimized weight array (size n + 1)

    begin
        augment X with a column of 1's for dummy feature X_0 //size X: d cross n+1
        initialize weight array w randomly // size(w) = n + 1
        for t = 1 to T do
            # create array ypred to store predictions
            P(Y=1 | x_i;w_t) for each sample at each iteration t
            Initialize prediction array ypred # size ypred : d
                # create array z to store w_0 + sum(w_j * x_j_i) for each sample at each iteration t

            intialize z #size z : d
            
            #compute predictions for all samples
            for i = 1 to d do
                #compute weighted sum for sample i
                z[i] = 0
                for j = 0 to n do
                    z[i] += w[j] cross X[i][j] #dot product
                end
                #apply sigmoid function to get probability
                ypred[i] = 1 / (1 + exp(-z[i]))
            end
            
            #initialize gradient vector g #size g(n) = n + 1
            for j = 0 to n do
                g[j] = 0
                for i = 1 to d do
                    #gradient of conditional log likelihood for w_j
                    g[j] += X[i][j] cross (y[i] - ypred[i]) #gradient ascent
                end

                #apply l2 regularization to all weights but w_0
                if j != 0 then
                    g[j] -= lambda cross w[j]
                end
                #update weight w_j using gradient ascent
                w[j] += eta cross g[j]
            end
        end
        return w
    end

batch: compute gradient using all samples in each iteration
    done in pseudocode above

mini-batch: compute gradient using small random subset of samples in each iteration
    small batches (50 or 100)
    faster than BGD, more stable than SGD
    
stochastic: update weights using gradient from single sample at a time
    fastest, noisiest, can converge quickly
        each sample triggers weight update, so more updates per iteration


#todo
    #consider tweaking learning rate and regularization lambda
    #regularization lambda = {.01, .1, 1, 10} 
    #learning rate eta = t_0/t, meaning in this case we'll just set up .01 and update accordingly
    #.01: very light - model can be complex, .1: mild, 1: moderate - good middle ground, 10: strong - model will be simpler
        #split training data into 70/30 train/validation

        for each lambda in regularization lambda:
            train on 70%
            evaluate on 30% validation set
        pick lambda with best validation accuracy
        train final model on full training set with best lambda
        test on test set

3 different versions of each classifier
    for each dataset (3)
    for each representation (2)
    = 18 classifiers to train and evaluate
"""
import numpy as np
import math

#New day, so starting fresh with logistic regression
class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, max_iterations=500, regularization_lambda=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization_lambda = regularization_lambda
        
    #z = element-wise dot product
    def sigmoid(self, z):
        z = np.clip(z, -500, 500) #clip to prevent overflow in exp
        return 1 / (1 + np.exp(-z))
    
    #fit -> train the model and return optimized weights
    def fit(self, X, y, gd_variant='batch', batch_size=50, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        #augment x with column of 1's for bias term
        #x.shape = (num_samples, num_features) since X is d x n+1 where sample means email and feature means word in vocab
        #np.ones creates colunmn m of 1's with parameter -> in this case, number of rows in x
        #np.column_stack stacks arrays in sequence horizontally (column wise) -> adds ones column to left of x
        x = np.column_stack((np.ones(X.shape[0]), X))
        num_samples, num_features = x.shape

        #initialize weights randomly
        self.weights = np.random.rand(num_features) 
        for iteration in range(self.max_iterations):
            #create array ypred to store predictions
            #P(Y=1 | x_i;w_t) for each sample at each iteration t
            #compute predictions for all samples
            z = np.dot(x, self.weights)
            y_pred = self.sigmoid(z)
            
            if gd_variant == 'batch':
                #compute gradient using vectorized dot product: X^T @ (y - y_pred)
                gradient = np.dot(x.T, y - y_pred)
                #apply L2 regularization to all weights but w_0 (exclude bias term w_0)
                gradient[1:] -= self.regularization_lambda * self.weights[1:]
                #update all weights using gradient ascent
                self.weights += self.learning_rate * gradient
            elif gd_variant == 'mini-batch':
                #loop through mini-batches
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    x_batch = x[start:end]
                    y_batch = y[start:end]
                    z_batch = np.dot(x_batch, self.weights)
                    y_pred_batch = self.sigmoid(z_batch)
                    #vectorized gradient computation for batch
                    gradient = np.dot(x_batch.T, y_batch - y_pred_batch)
                    #apply L2 regularization (exclude bias term)
                    gradient[1:] -= self.regularization_lambda * self.weights[1:]
                    #update weights
                    self.weights += self.learning_rate * gradient
            elif gd_variant == 'stochastic':
                #update weights using single sample at a time
                for i in range(num_samples):
                    z_i = np.dot(x[i], self.weights)
                    y_pred_i = self.sigmoid(z_i)
                    #vectorized gradient for single sample
                    gradient = x[i] * (y[i] - y_pred_i)
                    #apply L2 regularization (exclude bias term)
                    gradient[1:] -= self.regularization_lambda * self.weights[1:]
                    #update weights
                    self.weights += self.learning_rate * gradient
                
            else: 
                raise ValueError("Invalid GD variant, choose within {'batch', 'mini-batch', 'stochastic'}")
        return self.weights

    #predict -> use optimized weights to make predictions on new data
    def predict(self, X):
        x = np.column_stack((np.ones(X.shape[0]), X))
        z = np.dot(x, self.weights)
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int) #threshold at .5 to classify spam vs ham
    
    #evaluate: compute accuracy, precision, recall, F1 of model on test set
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y) #proportion of correct predictions
        if np.sum(y_pred) == 0: #if no predicted positives (spam in this case), precision is undefined, set to 0
            precision = 0
        else:
            precision = np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1) #sum of true positives/sum of predicted positives
        if np.sum(y) == 0: #if no actual positives, recall is undefined
            recall = 0
        else:
            #sum of true positives/sum of actual positives (sum of true positives + sum of false negatives)
            recall = np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1) 
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall) #harmonic mean
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    
    

