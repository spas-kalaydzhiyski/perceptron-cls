import numpy as np


class Perceptron:
    """
    Perceptron clasifier:
    @random_state is needed to generate the random state of the weights vector
    """
    def __init__(self, learning_rate=0.01, n_iterations=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.errors = []

    def fit(self, X, y):
        """
        Fit the training data for classification.
        @X - features matrix (n_samples X n_features)
        @y - correct labels for each sample (n_samples)
        """
        rand_gen = np.random.RandomState(self.random_state)
        # the weights have 1 more element than n_features to accomodate
        # for the bias unit w0*x0 where x0 = 1 and the threshold is 0
        w_vec_size = 1 + X.shape[1]
        self.w = rand_gen.normal(loc=0.0, scale=0.01, size=w_vec_size)
        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip (X, y):
                # each sample xi is a vector of shape (n_features, )
                update = self.learning_rate * (target - self.predict(xi))
                self.w[0] += update     # since xi in this case is x0=1
                self.w[1:] += update * xi
                errors += int(update != 0.0)
            self.errors.append(errors)

        return self

    def net_input(self, sample):
        return np.dot(sample, self.w[1:]) + self.w[0]

    def predict(self, sample):
        """
        Returns the predicted class label after the unit-step function
        """
        return np.where(self.net_input(sample) >= 0, 1, -1)

