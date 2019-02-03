import numpy as np


class Perceptron:
  """
  Class to represent one simple learning mechanism similar to
  real neurons in the human brain, the perceptron takes inputs
  and decides based on them whether to fire or not.
  """
  def __init__(self, learning_rate=0.01, n_epochs=64, random_state=1):
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.random_state = random_state
    self.errors = []    #used to store the errors per training epoch

  def fit(self, X, y):
    """
    Function to fit the model on the all the training samples.
    X = fetures matrix -               matrix (n_samples, n_features)
    y = target label for each sample - vector (n_samples)
    """
    rand_gen = np.random.RandomState(self.random_state)
    weights_vec_size = X.shape[1] + 1
    self.weights = rand_gen.normal(loc=0.0, scale=0.01, size=weights_vec_size)
    for _ in range(self.n_epochs):
      error = 0
      for xi, target in zip(X, y):
        update = self.learning_rate * (self.predict(xi) - target)
        self.weights[1:] += update * xi 
        self.weights[0] += update
        error += int(update != 0)
      self.errors.append(error)

  def get_errors(self):
    return self.errors

  def get_all_weight_update(self):
    return self.weight_updates
  
  def net_input(self, xi):
    """
    return the net_input of the weights and the current
    values of the sample being processed during training.
    """
    return np.dot(xi, self.weights[1:]) + self.weights[0]

  def predict(self, xi):
    """
    utilises a threshold (0 in this case) to classify the label
    for the inputs and the current state of the weights and biases.
    """
    return np.where(self.net_input(xi) >= 0, 1, -1)  


