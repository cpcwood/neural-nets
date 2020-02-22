import numpy as np
import math 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_circles

# Function that generates nonlinear data
np.random.seed(0)

def create_data():
  X, y = make_circles(n_samples=1000, factor=.3, noise=.10)
  return X, y


def create_data2(points, classes):
  X = np.zeros((points*classes,2))
  Y = np.zeros(points*classes, dtype = 'uint8')
  for class_number in range(classes):
    ix = range(points*class_number, points*(class_number+1))
    t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
    r = np.linspace(0.0, 1, points)  # radius
    X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
    Y[ix] = class_number
  return X, Y

# X is an array of samples or coordinate pairs, Y is the label for where the data falls


def plot_train(X,y):
  plt.scatter(X[:,0],X[:,1], c = y)
  plt.show()


class Layer_Dense:

  def __init__(self, inputs, neurons):
      self.weights = 0.01 * np.random.randn(inputs, neurons)
      self.biases = np.zeros((1, neurons))

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

  def forward(self, inputs):
    self.output = np.maximum(0.0,inputs)

  def derivative(self, dZ):
    dZ[dZ > 0 ] = 1
    dZ[dZ <= 0] = 0 
    return dZ

class Activation_Softmax():
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities
    
class Loss_CategoricalCrossEntropy():
  def forward(self, y_pred, y_true):
    y_pred = y_pred[range(len(y_pred)), y_true]
    negative_log_liklihoods = -np.log(y_pred)
    prediction_loss = np.mean(negative_log_liklihoods)
    return prediction_loss
  

def run_net(X, y, n):
  dense1 = Layer_Dense(2,3)
  activation1 = Activation_ReLU()
  dense2 = Layer_Dense(3,3)
  activation2 = Activation_Softmax()
  loss_function = Loss_CategoricalCrossEntropy()
  n = len(y)

  # lowest_loss = 9999999 # Initial Values
  # best_dense1_weights = dense1.weights
  # best_dense1_biases = dense1.biases
  # best_dense2_weights = dense2.weights
  # best_dense2_biases = dense2.biases 

  for i in range(n):

    # Update weights w/ small random values 
    # dense1.weights += 0.05 * np.random.randn(2,3)
    # dense1.biases += 0.05 * np.random.randn(1,3)
    # dense2.weights += 0.05 * np.random.randn(3,3)
    # dense2.biases += 0.05 * np.random.randn(1,3)

    # Move forward through network
    forward_values, probabilities = forward_propagation(dense1, activation1, dense2, activation2)

    # dense1.forward(X)
    # activation1.forward(dense1.output)
    # dense2.forward(activation1.output)
    # activation2.forward(dense2.output)
    
    # Loss calculation
    loss, predictions, acc = loss_and_accuracy(probabilities, y, loss_function)
    # loss = loss_function.forward(activation2.output, y)
    # predicitons = np.argmax(activation2.output, axis=1)
    # acc = np.mean(predicitons == y)

    # Propogate backwards 
    backprop_output = backward_propagation(forward_values, y, n, activation1)


    print('new set of weights found, iteration ', i,
    'loss: ', loss, 'acc: ', acc)

    




''' Functions To be applied to the loop above ''' 

def forward_propagation(X, dense1, activation1, dense2, activation2):
  dense1.forward(X)
  Z1 = dense1.output
  activation1.forward(Z1)
  A1 = activation1.output
  dense2.forward(A1)
  Z2 = dense2.output
  activation2.forward(Z2)
  probabilities = activation2.output
  forward_values = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": probabilities}
  return forward_values, probabilities

def loss_and_accuracy(probabilities, true_y, entropy_class):
  loss = entropy_class.forward(probabilities, true_y)
  predicitions = np.argmax(probabilities, axis=1)
  acc = np.mean(predicitions == true_y)
  return loss, predicitions, acc 

def backward_propagation(X, forward_values, true_y, sample_size, relu_class):
  dZ2 = np.argmax(forward_values["A2"]) - true_y
  dW2 = (1/sample_size) * np.dot(dZ2, forward_values["A1"].T) #Breaks here bc dims but remove transpose to see the shitty dimension i
  dB2 = (1/sample_size) * np.sum(dZ2, keepdims=True)
  dZ1 = relu_class.derivative(dZ2)
  dW1 = (1 / sample_size) * np.dot(dZ1, X)
  dB1 = (1 / sample_size) * np.sum(dZ1, keepdims=True)
  backprop_output = {"dZ2": dZ2, "dW2": dW2, "dB2": dB2, "dZ1": dZ1, "dB1": dB1, "dW1": dW1}
  return backprop_output


''' Below is me just trying to get some visibility '''
X,y = create_data2(100,3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()
sample_size = len(y)

forward_values, probs = forward_propagation(X, dense1, activation1, dense2, activation2)
loss, prediction, accuracy = loss_and_accuracy(probs,y,loss_function)
backward_output = backward_propagation(X, forward_values, y, sample_size, activation1)