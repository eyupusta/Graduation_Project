import pandas as pd 
import numpy as np 
from parsearff import parse_arff
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.utils import shuffle 


class SVM:
    def __init__(self, reg_parameter = 10000, learning_rate = 0.0001, max_epochs = 20000):
        self.learning_rate = learning_rate
        self.reg_parameter = reg_parameter
        self.max_epochs = max_epochs

    def compute_cost(self, W, X, Y):
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0 
        hinge_loss = self.reg_parameter * (np.sum(distances) / N)

        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost
    
    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])    
        
        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))    

        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.reg_parameter * Y_batch[ind] * X_batch[ind])
            dw += di    
        dw = dw/len(Y_batch)
        return dw

    def fit(self, features, outputs):
        weights = np.zeros(features.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01

        for epoch in range(1, self.max_epochs):
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                weights = weights - (self.learning_rate * ascent)        
            if epoch == 2 ** nth or epoch == self.max_epochs - 1:
                cost = self.compute_cost(weights, features, outputs)
                print("Epoch is:{} and Cost is: {}".format(epoch, cost))
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
                nth += 1
                
        self.weights = weights

    def get_w(self):
        return self.weights

    def set_weights(self, w):
        self.weights = w

    def predict(self, X_test, y_test):
        y_test_predicted = np.array([])
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(self.weights, X_test[i]))
            y_test_predicted = np.append(y_test_predicted, yp)

        print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))

        return y_test_predicted


data, columns = parse_arff('Training Dataset.arff')
df = pd.DataFrame(data, columns = columns, dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]],df[df.columns[-1]])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

svm = SVM()

svm.fit(x_train, y_train)

ypred = svm.predict(x_test, y_test)
print(y_test)
print(ypred)