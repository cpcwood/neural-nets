import numpy as np
from random import seed
from random import random

# Inputs => layer_transfer => layer_activation => layer_transfer(output) => output_function

# ======================================================
# Create network with randomised weights - each node hash with array of weights
def create_network(n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs): 
  network = []
  hidden_layer1 = [np.random.random((n_hidden_nodes, n_inputs)), np.random.rand(n_hidden_nodes)]
  network.append(hidden_layer1)
  for layer_n in range(n_hidden_layers-1): # Create each layer
    hidden_layer = [np.random.random((n_hidden_nodes, n_hidden_nodes)), np.random.rand(n_hidden_nodes)]
    network.append(hidden_layer)
  output_layer = [np.random.random((n_outputs, n_hidden_nodes)), np.random.rand(n_outputs)]
  network.append(output_layer)
  return network

# Test network generation
seed(1)
# print create_network(3, 2, 4, 2)  
  
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
  a_nodes[a_nodes < 0] = 0
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
# net = create_network(2, 2, 4, 3)
# print('net')
# print(net)
# inputs = np.array([0.4, 2.])
# expected_output = np.array([1., 0., 0.])
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
# Calcualte dC_dz from relu da_dz and previous layer dC_da
# a = np.array([2,3,1])
# a = a.reshape(-1,1)
# b = np.array([1,4,0])
# print(a)
# print(b)
# print(a * b)
# print(sum(a * b))


#dC_dz_derivatives is dC_da * da_dz for each node... equal to dC_db

# =======================================================
# Create mean squared error
def mean_square_error(results, expected):
  error = (expected - results) ** 2
  return error

# # Test mean squared error
# results = np.array([0.25, 0.5, 0.15, 0.1])
# print('results')
# print(results)
# expected = np.array([0,1,0,0])
# print('expected')
# print(expected)
# print('equation')
# print(mean_square_error(results, expected))

# =======================================================
# Create cost total
def cost(results, expected):
  cost = np.sum(mean_square_error(results, expected))
  return cost

# # Test cost function
# results = np.array([0.25, 0.5, 0.15, 0.1])
# expected = np.array([0,1,0,0])
# print('Expected result')
# print(0.345)
# print('result')
# print(cost(results, expected))



# =======================================================
# Create derviate vector of softmax function with respect to change in z
# da/dz for each node in output array
def softmax_derivative(a_output):
  s = a_output.reshape(-1,1)
  s_aj = np.diagflat(s)
  s_mat = np.dot(s, s.T)
  derivate_matrix = s_aj - s_mat
  per_node_derivative = np.prod(derivate_matrix, axis=0)
  return per_node_derivative
