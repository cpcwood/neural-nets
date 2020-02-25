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
  in_values = inputs
  z_all = []
  a_all = []
  for layer in network[:len(network)-1]:
    z = layer_transfer(layer, in_values)
    z_all.append(z)
    a = layer_activation(z)
    a_all.append(a)
    in_values = a
  z = layer_transfer(network[len(network)-1], in_values)
  z_all.append(z)
  results = softmax(z)
  a_all.append(results)
  return [z_all, a_all] 

# # Test forward propagation
# net = create_network(2, 2, 3, 2)
# print('net')
# print(net)
# inputs = np.array([0.4, 2.])
# a = forward_propagate(net, inputs)

# print('outputs')
# print('z_all')
# print(a[0])
# print('a_all')
# print(a[1])

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
def backpropagation_output_layer(a_all, expected_output):
  dcost_dactivation = dC_da(a_all[-1], expected_output)
  dactivation_dz = softmax_da_dz(a_all[-1])
  # weights matrix
  dC_dw = []
  for i in range(len(a_all[-1])):
    output_node = []
    for input_value in a_all[-2]:
      weight_derivative = input_value * dactivation_dz[i] * dcost_dactivation[i]
      output_node.append(weight_derivative)
    dC_dw.append(output_node)
  dC_dw = np.array(dC_dw)
  # bias matrix
  dC_db = dcost_dactivation * dactivation_dz
  return [dC_dw, dC_db]


# # Test backpropagation_output_layer
# net = create_network(2, 2, 4, 3)
# print('net')
# print(net)
# inputs = np.array([0.4, 2.])
# expected_output = np.array([1., 0., 0.])
# a = forward_propagate(net, inputs)

# print('outputs')
# print('z_all')
# print(a[0])
# print('a_all')
# print(a[1])
# print('output weight matrix')
# output = backpropagation_output_layer(a[1], expected_output)
# print(output[0])
# print('output bias matrix')
# print(output[1])

# =======================================================
# Create da/dz for relu
def reLU_da_dz(a_nodes):
  a_nodes[a_nodes > 0] = 1
  a_nodes[a_nodes < 0] = 0
  return a_nodes

# Test reLU_da_dz
a_nodes = np.array([-3.34352538,  2.06987895,  3.19224339,  4.30583721])
print(reLU_da_dz(a_nodes))


# =======================================================
# Create backprop hidden layers
def backpropagation_hidden_layers(a_all, output_dC_dz_derivatives):
  return


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


# =======================================================
# Create da/dz
def da_dz(a_all, z_all):

  return


# Test da/dz


# =======================================================
# Create dz/dw
def dz_dw(z_all, layer):

  return





# =======================================================
# Create batch training data
# def batch_training(inputs_batch, network):
#   return 

# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
# 	return output * (1.0 - output)

# def back_prop_test(net, expected):
#   for i in reversed(range(len(net))):
#     layer = net[i]
#     print(i)
#     print(layer)
#     errors = list()
# 		if i != len(network)-1:
#       for j in range(len(layer)):
#         print(j)
# 				error = 0.0
# 				for neuron in network[i + 1]:
# 					error += (neuron['weights'][j] * neuron['delta'])
# 				errors.append(error)
# 		else:
# 			for j in range(len(layer)):
# 				neuron = layer[j]
# 				errors.append(expected[j] - neuron['output'])
# 		for j in range(len(layer)):
# 			neuron = layer[j]
# 			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])




# # Backpropagate error and store in neurons
# def backward_propagate_error(network, expected):
#   for i in reversed(range(len(network))):
#     layer = network[i]
#     print(i)
 
# # test backpropagation of error
# network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# 		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# expected = [0, 1]
# back_prop_test(network, expected)
# # for layer in network:
	# print(layer)