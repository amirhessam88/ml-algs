import numpy as np


class LogisticRegression:
    """
    Logistic Regression based on Gradient Descent
    z = wx + b
    y_hat = sigmoid(z) = sigmoid(wx + b) = 1/(1 + exp[-(wx+b)])
    update rules: w = w - lr * dw
                  b = b - lr * db
    where dw = 1/N sum(2x(y_hat - y))
          db = 1/N sum(2(y_hat - y))
    Parameters
    ----------
    lr: float, optional (default=0.001)
        Learning rate used of updating weights/bias
    n_iters: int, optional (default=1000)
        Maximum number of iterations to update weights/bias
    weights: numpy.array, optional (default=None)
        Weights array of shape (n_features, ) where
        will be initialized as zeros
    bias: float, optional (default=None)
        Bias value where will be initialized as zero
    """

    def __init__(self, learning_rate=0.001, n_iters=1000, weights=None, bias=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        """
        Train model using iterative gradient descent
        Parameters
        ----------
        X_train: numpy.array
            Training feature matrix
        y_train: 1D numpy.array or list
            Training binary class labels [0, 1]
        """
        # unpack the shape of X_train
        n_samples, n_features = X_train.shape

        # initialize weights and bias with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # main loop
        # self.loss = []
        for _ in range(self.n_iters):
            z = np.dot(X_train, self.weights) + self.bias
            y_hat = self._sigmoid(z)

            # update weights + bias
            dw = (1.0 / n_samples) * 2 * np.dot(X_train.T, (y_hat - y_train))
            db = (1.0 / n_samples) * 2 * np.sum(y_hat - y_train)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # TODO : LOSS FUNCTION
            # loss_ = -(y_train * np.log(y_hat) + (1 - y_train)*np.log(1 - y_hat))
            # average cost
            # loss_ = np.nansum(loss_)/n_samples
            # self.loss.append(loss_)

        return None

    def predict_proba(self, X_test):
        """
        Prediction probability of the test samples
        Parameters
        ----------
        X_test: numpy.array
            Testing feature matrix
        """
        z = np.dot(X_test, self.weights) + self.bias
        y_pred_proba = self._sigmoid(z)

        return y_pred_proba

    def predict(self, X_test, threshold=0.5):
        """
        Prediction class of the test samples
        Parameters
        ----------
        X_test: numpy.array
            Testing feature matrix
        threshold: float, optional (default=0.5)
            Threshold to define predicted classes
        """
        z = np.dot(X_test, self.weights) + self.bias
        y_pred_proba = self._sigmoid(z)
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred_proba]

        return y_pred

    def _sigmoid(self, z):
        """
        Sigmoid Function
        f(z) = 1/(1 + exp(-z))
        """

        return 1.0 / (1 + np.exp(-z))
