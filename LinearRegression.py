import numpy as np
class LinRes():
    def __init__(self, x_train, y_train):

        self.n_examples, self.n_features = x_train.shape
        self.W = np.zeros(self.n_features)
        self.b = 0

        self.x_train = x_train
        self.y_train = y_train

    def test(self, X):

        return np.dot(X, self.W) + self.b

    def train(self, lr=0.00001, n_epochs=50,verbose=False):
        epoch = 0
        self.MSE=[]
        self.MAE=[]
        self.b_list=[]
        self.W_list=[]
        while epoch < n_epochs:
            y_pred = self.test(self.x_train)
            if verbose:
                mae = np.sum(abs(y_pred - self.y_train))/self.n_examples
                mse = np.sum((y_pred - self.y_train) ** 2)/(self.n_examples)

                self.MSE.append(mse)
                self.MAE.append(mae)

            dW = (1/self.n_examples) * np.dot(self.x_train.T, (y_pred - self.y_train))
            db = (1/self.n_examples) * np.sum((y_pred - self.y_train))

            self.W -= lr * dW
            self.b -= lr * db

            self.b_list.append(self.b)
            self.W_list.append(self.W)

            epoch += 1

        return y_pred,self.b_list,self.W_list,self.MSE,self.MAE


