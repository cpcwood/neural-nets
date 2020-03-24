import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
np.random.seed(0)

# Inputs => layer_transfer => layer_activation => layer_transfer(output) => output_function

# ======================================================
# Create network with randomised weights - each node hash with array of weights

def generate_layer(n_nodes, n_weights):
  return [(np.random.randn(n_nodes, n_weights)*np.sqrt(2.0/(n_weights+n_nodes))), (np.random.randn(n_nodes)*np.sqrt(2.0/(n_weights+n_nodes)))]

def create_network(n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs): 
  network = []
  hidden_layer1 = generate_layer(n_hidden_nodes, n_inputs)
  network.append(hidden_layer1)
  for layer_n in range(n_hidden_layers-1):
    hidden_layer = generate_layer(n_hidden_nodes, n_hidden_nodes)
    network.append(hidden_layer)
  output_layer = generate_layer(n_outputs, n_hidden_nodes)
  network.append(output_layer)
  return network

# Test network generation

# net = create_network(3, 2, 2, 2)  
# for layer in net:
#   print(layer)
  
# ======================================================
# Create hidden layer transfer function
def layer_transfer(layer, inputs):
  dot_product = np.dot(layer[0], inputs)
  z_all = dot_product + layer[1]
  return z_all

# Test transfer function
# layer = [np.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.5]]), np.array([0.2, 0.1, 0.2])]
# input = np.array([0.2, 0.5])
# print 'Expected value'
# print [(0.1*0.2 + 0.2*0.5 + 0.2), (0.2*0.2 + 0.5*0.3 + 0.2), (0.2* 0.2 + 0.5*0.5 + 0.2)]
# print 'Transfer function value'
# print layer_transfer(layer, input)

# ======================================================
# Create reLU activation function 
def reLU(weighted_sum):
  if weighted_sum < 0:
    return 0
  else:
    return weighted_sum

# Test activation function
# print reLU(-1)
# print reLU(0.5)

# ======================================================
# Create layer activation
def layer_activation(z_all):  # Takes hidden layer z_all and applies activation function
  a_all = np.array(map(reLU, z_all))
  return a_all

# Test layer activation
# layer = [np.array([[0.1, -0.2], [0.2, 0.3], [-0.2, -0.5]]), np.array([0.2, 0.1, 0.2])]
# print(layer)
# input = np.array([0.2, 0.5])
# z_all = layer_transfer(layer, input)
# print('Expected')
# print('[ 0.12  0.29  0.  ]')
# print('Layer Activation Value')
# print(layer_activation(z_all))


# ======================================================
# Create output function
def softmax(z_all):
  e_x = np.exp(z_all - np.max(z_all))
  return e_x / e_x.sum(axis=0)

# Create output function
# a = np.array([0.12, 0.01, 0.4])
# print(softmax(a))

# =======================================================
# Create feed forward results
def feed_forward(trained_network, inputs):
  in_values = inputs
  for layer in trained_network[:len(trained_network)-1]:
    z = layer_transfer(layer, in_values)
    a = layer_activation(z)
    in_values = a
  z = layer_transfer(trained_network[len(trained_network)-1], in_values)
  results = softmax(z)
  return results

# =======================================================
# Backpropagation Functionality
# =======================================================

# =======================================================
# Create forward propagation
def forward_propagate(network, inputs):
  a_in = inputs
  a_all = [a_in]
  for layer in network[:len(network)-1]:
    z = layer_transfer(layer, a_in)
    a = layer_activation(z)
    a_all.append(a)
    a_in = a
  z = layer_transfer(network[len(network)-1], a_in)
  results = softmax(z)
  a_all.append(results)
  return a_all 

# # Test forward propagation
# net = create_network(2, 2, 3, 2)
# print('net')
# print(net)
# inputs = np.array([0.4, 2.])
# a = forward_propagate(net, inputs)

# print('outputs')
# print('a_all')
# print(a)


# =======================================================
# dC/da for each node in the output array
def dC_da(a_output, y_expected):
  return 2*(y_expected - a_output)

# # Test dC/da
# a = np.array([0.4, 0.2, 0.2])
# y = np.array([1., 0., 0.,])
# print(dC_da(a,y))

# =======================================================
# Create da/dz
def softmax_da_dz(a_output):
  return a_output - (a_output**2)

# # Test softmax_da_dz
# a = np.array([0.4, 0.2, 0.2])
# print(softmax_da_dz(a))

# =======================================================
# Create backprop output
def backpropagation_output_layer(a_all, expected_output, layer_weights):
  dcost_dactivation = dC_da(a_all[-1], expected_output)
  dactivation_dz = softmax_da_dz(a_all[-1])
  dC_dz = dcost_dactivation * dactivation_dz
  # weights dervivative matrix is dcost_dz for each node * a^L-1 for each weight in node
  dC_dz_re = dC_dz.reshape(-1,1)
  dC_dw = dC_dz_re * a_all[-2]
  # a^L-1 derivatives is sum of (dC_dz_re for node * weight for input)
  dC_da_L_minus_1 = (dC_dz_re * layer_weights).sum(axis=0)
  # dC_dz is the same a derivative matrix for biases
  return [dC_dw, dC_dz, dC_da_L_minus_1]

  

# # Test backpropagation_output_layer
# net = create_network(2, 2, 4, 3)
# print('net')
# print(net)
# inputs = np.array([0.4, 2.])
# expected_output = np.array([1., 0., 0.])
# a_all = forward_propagate(net, inputs)

# print('outputs')
# print('a_all')
# print(a_all)

# print('output weight matrix new')
# output = backpropagation_output_layer(a_all, expected_output, net[-1][0])
# print(output[0])
# print('output bias matrix')
# print(output[1])
# print('dC_da_L_minus_1')
# print(output[2])



# # Test generating a^l-1 derivatives
# print('test test')
# dcdz = np.array([1,2,3])
# dcdz_re = dcdz.reshape(-1,1)
# w = np.array([[4,2], [1,2], [2,3]])
# print('3x nodes in output layer dc_dz')
# print(dcdz)
# print('each node having 2 weights')
# print(w)
# expected = np.array([[4,2], [2,4], [6,9]])
# print('expected')
# print(expected)
# print('output')
# print(dcdz_re * w)
# print((dcdz_re * w).sum(axis=0))


# =======================================================
# Create da/dz for relu
def reLU_da_dz(a_nodes):
  a_nodes[a_nodes > 0] = 1
  a_nodes[a_nodes <= 0] = 0
  return a_nodes

# # Test reLU_da_dz
# a_nodes = np.array([-3.34352538,  2.06987895,  3.19224339,  4.30583721])
# print(reLU_da_dz(a_nodes))


# =======================================================
# Create backpropagation full
# desired output [[weights derivatives], [bias derviatives]]
def backpropagation(net, a_all, expected_output):
  cost_derivatives = []
  output_backprop = backpropagation_output_layer(a_all, expected_output, net[-1][0])
  # append output_backprop derivatives to results matrix
  cost_derivatives.append([output_backprop[0], output_backprop[1]])
  dC_da = output_backprop[2]
  # perform backprop on all hidden layers
  for i in reversed(range(len(a_all)-1)):
    if i == 0:
      break
    # create dC_dz
    da_dz = reLU_da_dz(a_all[i])
    dC_dz = da_dz * dC_da
    # create weight derviatives
    dC_dz_re = dC_dz.reshape(-1,1)
    dC_dw = dC_dz_re * a_all[i-1]
    cost_derivatives.append([dC_dw, dC_dz])
    # create and ammend a^L-1 derivatives
    dC_da = (dC_dz_re * net[i-1][0]).sum(axis=0)
  #reverse appended lists to match net weights structure
  cost_derivatives = cost_derivatives[::-1]
  return cost_derivatives


# # Test backpropagation
# net = create_network(2, 1, 1, 2)
# print('net')
# print(net)
# inputs = np.array([0.4, 2.])
# expected_output = np.array([1.])
# a_all = forward_propagate(net, inputs)

# print('outputs')
# print('a_all')
# print(a_all)

# # print('output h layers')
# # output = backpropagation(net, a_all, expected_output)

# output = backpropagation(net, a_all, expected_output)
# print('output weight matrix new')
# print(output)

# backpropagation returns cost matrix same dimensions as the net matrix

# =======================================================
# Create mean squared error
def mse(outputs, expected_outputs):
  mean_squared_error = (((outputs - expected_outputs)**2).sum(axis=0)) / len(outputs)
  return mean_squared_error * 100

# # test mean squared error
# outputs = np.array([0.7, 0.2, 0.1])
# expected_outputs = np.array([1., 0., 0.])

# # 0.3, 0.2, 0.1 => 0.09, 0.04, 0.01 => 0.14 => 0.0466 => 4.66

# print(mse(outputs, expected_outputs))
    
# =======================================================
# Create 2 point dataset plotter

def plot_train(X,y):
  plt.scatter(X[:,0],X[:,1], c = y)
  plt.show()

# =======================================================
# Create hard dataset to train net on

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

# Test generation and print
# xy, c = create_data2(1000, 2)
# plot_train(xy, c)



# =======================================================
# Create easy dataset to train net on

# Function that generates nonlinear data
def create_circles_data(samples, factor, noise):
  X, y = make_circles(n_samples=samples, factor=factor, noise=noise)
  return X, y

# Test generation and print
# xy, c = create_circles_data(1000, 0.2, 0.1)
# plot_train(xy, c)

# =======================================================
# Create methods to shuffle and make classification data hot for training

# convert c into hot-array
def class_to_hot(c):
  c_hot = np.zeros((c.size, c.max()+1))
  c_hot[np.arange(c.size),c] = 1
  return c_hot

# shuffle arrays for batch data
def unison_shuffle_array(a, b):
  p = np.random.permutation(len(a))
  return a[p], b[p]


# =======================================================
# Create batch train

def batch_train(net, xy, c_hot, batch_size, learning_rate):
  assert len(c) % batch_size == 0
  loc = 0

  for i in range(len(c)/batch_size):
    backprop_matrix = []

    for n in range(batch_size):
      a_all = forward_propagate(net, xy[i])
      if n == 0:
        backprop_matrix = backpropagation(net, a_all, c_hot[i])
      else:
        next_backprop_matrix = backpropagation(net, a_all, c_hot[i])
        for j in range(len(backprop_matrix)):
          for n in range(len(backprop_matrix[j])):
            backprop_matrix[j][n] += next_backprop_matrix[j][n]
      loc += 1

    # normalise cost matrix
    for j in range(len(backprop_matrix)):
      for n in range(len(backprop_matrix[j])):
        backprop_matrix[j][n] / batch_size

    # step network weight and biases in direction of cost function
    for j in range(len(net)):
      for n in range(len(net[j])):
        net[j][n] += (backprop_matrix[j][n] * learning_rate)

    outputs = feed_forward(net, xy[i-1])
    # print(mse(outputs, c_hot[i-1]))
    
  return net



# create classification dataset
xs = np.linspace(-1,1,50)
ys = np.linspace(-1,1,50)
data_set = []
for x in xs:
  x
  for y in ys:
    data_set.append([x, y])
classify_net_data = np.array(data_set)

def classify_net(net):
  outputs = []
  for xy in classify_net_data:
    outputs.append(feed_forward(net, xy))
  c_data = np.argmax(np.array(outputs), axis=1)
  Z = np.split(c_data, 50)
  # print(Z)
  plt.ion()
  plt.imshow(Z, cmap=plt.cm.RdBu, extent=(0, 1, 0, 1), interpolation='bilinear')
  plt.title('net')
  plt.show()
  plt.pause(0.0001)




    
# Train net with dataset
# ============================
# create_network(n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs): 

# Simple 2D dataset test
net = create_network(2, 2, 5, 2)
xy, c = create_circles_data(1000, 0.2, 0.1)
batch_size = 10
step_size = 0.01

# Complex 2D dataset test
# net = create_network(2, 8, 8, 2)
# xy, c = create_data2(1000, 2)
# batch_size = 10
# step_size = 0.001


# Run training and print output
c_hot = class_to_hot(c)
xy_shuffle, c_hot_shuffle = unison_shuffle_array(xy, c_hot)

for i in range(100):
  net = batch_train(net, xy_shuffle, c_hot_shuffle, batch_size, step_size)
  # check and print current error
  outputs = feed_forward(net, xy[0])
  # print(mse(outputs, c_hot[0]))
  outputs = feed_forward(net, xy[-1])
  # print(mse(outputs, c_hot[-1]))
  classify_net(net)