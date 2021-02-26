import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from IPython.core.display import display, HTML
import warnings

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))


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
        self.weights = weights
        self.bias = bias

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
        y_pred_proba = self.predict_proba(X_test)
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred_proba]

        return y_pred

    def plot_decision_boundary(self, X_test, y_test, threshold=0.5, figsize=(8, 6)):
        """
        Plot the decision boundary
        Parameters
        ----------
        X_test: numpy.array
            Testing feature matrix
        y_test: 1D numpy.array or list
            Testing targets
        threshold: float, optional (default=0.5)
            Threshold to define predicted classes
        figsize: tuple, optional (default=(8,6))
            Figure size
        """
        # calc pred_proba
        y_pred_proba = self.predict_proba(X_test)

        # define positive and negative classes
        pos_class = []
        neg_class = []
        for i in range(len(y_pred_proba)):
            if y_test[i] == 1:
                pos_class.append(y_pred_proba[i])
            else:
                neg_class.append(y_pred_proba[i])

        # plotting
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            range(len(pos_class)),
            pos_class,
            s=25,
            color="navy",
            marker="o",
            label="Positive Class",
        )
        ax.scatter(
            range(len(neg_class)),
            neg_class,
            s=25,
            color="red",
            marker="s",
            label="Negative Class",
        )
        ax.axhline(
            threshold,
            lw=2,
            ls="--",
            color="black",
            label=f"Decision Bound = {threshold}",
        )
        ax.set(
            title="Decision Boundary",
            xlabel="# of Samples",
            ylabel="Predicted Probability",
        )
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1, framealpha=0.0)
        plt.show()

    def _sigmoid(self, z):
        """
        Sigmoid Function
        f(z) = 1/(1 + exp(-z))
        """

        return 1.0 / (1 + np.exp(-z))
