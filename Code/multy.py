import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd



data = "../data/iris.csv"
df = pd.read_csv(data, usecols= ["Id","SepalLengthCm","SepalWidthCm","PetalLengthCm",
                                "PetalWidthCm","Species"], sep=',')

df = df.sample(frac=1)
df = df.drop(['Id'], axis=1)
y = df.pop('Species')

x_train = df.iloc[0:int(0.75*len(df)), :]
x_train = np.array(x_train.reset_index(drop=True))
x_test = df.iloc[int(0.75*len(df))+1:len(df), :]
x_test = np.array(x_test.reset_index(drop=True))
labels_train = y.iloc[0:int(0.75*len(df))]
labels_train = labels_train.reset_index(drop=True)
labels_test = y.iloc[int(0.75*len(df))+1:len(df)]
labels_test = labels_test.reset_index(drop=True)

x_train = x_train.T
x_test = x_test.T

y_train = np.zeros((3, x_train.shape[1]))
count = 0
for label in labels_train:
  if label == 'Iris-virginica':
    y_train[0][count] = 1.0
    y_train[1][count] = 0.0
    y_train[2][count] = 0.0

  elif label == 'Iris-setosa':
    y_train[0][count] = 0.0
    y_train[1][count] = 1.0
    y_train[2][count] = 0.0
  elif label == 'Iris-versicolor':
    y_train[0][count] = 0.0
    y_train[1][count] = 0.0
    y_train[2][count] = 1.0
  count += 1

y_test = np.zeros((3, x_test.shape[1]))
count = 0
for label in labels_test:
  if label == 'Iris-virginica':
    y_test[0][count] = 1.0
    y_test[1][count] = 0.0
    y_test[2][count] = 0.0

  elif label == 'Iris-setosa':
    y_test[0][count] = 0.0
    y_test[1][count] = 1.0
    y_test[2][count] = 0.0
  elif label == 'Iris-versicolor':
    y_test[0][count] = 0.0
    y_test[1][count] = 0.0
    y_test[2][count] = 1.0
  count += 1


Xt_train = tf.Variable(x_train, name='Xt_train')
Xt_test = tf.Variable(x_test, name='Xt_test')
Yt_train = tf.Variable(y_train, name='Yt_train')
Yt_test = tf.Variable(y_test, name='Yt_test')

def relu(z):
  return(tf.math.maximum(0, z))

def softmax(z):
  e_x = tf.math.exp(z - tf.math.reduce_max(z, axis=0))
  return e_x / tf.math.reduce_sum(e_x, axis=0)

z = tf.convert_to_tensor([[1.0,2.0,-1],[4.0,5.0,6.0]])
print(tf.transpose(softmax(z)))
print(tf.nn.softmax(tf.transpose(z)))

print(relu(z))
print(tf.nn.relu(z))


def layer_sizes(X, Y):
  """
  Arguments:
  X -- input dataset of shape (input size, number of examples)
  Y -- labels of shape (output size, number of examples)

  Returns:
  n_x -- the size of the input layer
  n_h -- the size of the hidden layer
  n_y -- the size of the output layer
  """
  ### START CODE HERE ### (≈ 3 lines of code)
  n_x = X.shape[0]  # size of input layer
  n_h1 = 5
  n_h2 = 5
  n_y = Y.shape[0]  # size of output layer
  ### END CODE HERE ###
  return (n_x, n_h1, n_h2, n_y)

def initialize_parameters(n_x, n_h1, n_h2, n_y):


  np.random.seed(2)  # we set up a seed so that the result become same in every execution.

  ### START CODE HERE ### (≈ 4 lines of code)
  W1 = np.random.randn(n_h1, n_x) * 0.01
  b1 = np.zeros((n_h1, 1))
  W2 = np.random.randn(n_h2, n_h1)
  b2 = np.zeros((n_h2, 1))
  W3 = np.random.randn(n_y, n_h2)
  b3 = np.zeros((n_y, 1))

  W1 = tf.Variable(W1, dtype='float64', name='W1')
  b1 = tf.Variable(b1, dtype='float64', name='b2')
  W2 = tf.Variable(W2, dtype='float64', name='W2')
  b2 = tf.Variable(b2, dtype='float64', name='b1')
  W3 = tf.Variable(W3, dtype='float64', name='W3')
  b3 = tf.Variable(b3, dtype='float64', name='b3')

  parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3}

  return parameters

def forward_propagation(X, parameters):
  """
  Argument:
  X -- input data of size (n_x, m)
  parameters -- python dictionary containing your parameters (output of initialization function)

  Returns:
  A2 -- The sigmoid output of the second activation
  cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
  """
  # Retrieve each parameter from the dictionary "parameters"
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
  W3 = parameters["W3"]
  b3 = parameters["b3"]

  Z1 = tf.linalg.matmul(W1, X) + b1
  A1 = relu(Z1)
  Z2 = tf.linalg.matmul(W2, A1) + b2
  A2 = relu(Z2)
  Z3 = tf.linalg.matmul(W3, A2) + b3
  A3 = softmax(Z3)

  return A3

def C(A3, Y):
  """
  Computes the cross-entropy cost given in equation (13)

  Arguments:
  A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
  Y -- "true" labels vector of shape (1, number of examples)
  parameters -- python dictionary containing your parameters W1, b1, W2 and b2
  [Note that the parameters argument is not used in this function,
  but the auto-grader currently expects this parameter.
  Future version of this notebook will fix both the notebook
  and the auto-grader so that `parameters` is not needed.
  For now, please include `parameters` in the function signature,
  and also when invoking this function.]

  Returns:
  cost -- cross-entropy cost given equation (13)

  """

  m = Y.shape[1]  # number of example

  # Compute the cross-entropy cost
  ### START CODE HERE ### (≈ 2 lines of code)
  logprobs = tf.math.multiply(tf.math.log(A3), Y)
  cost = -tf.math.reduce_sum(logprobs)
  ### END CODE HERE ###

  cost = tf.squeeze(cost)  # makes sure cost is the dimension we expect.
  # E.g., turns [[17]] into 17

  return cost

lamda = tf.convert_to_tensor(0.009, dtype='float64')
thrsh = 0.1

n_x, n_h1, n_h2, n_y = layer_sizes(Xt_train, Yt_train)
parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
# print(parameters)

i = 0

W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
W3 = parameters["W3"]
b3 = parameters["b3"]

while i < 100:
  with tf.GradientTape(persistent=True) as tape:
    A3 = forward_propagation(Xt_train, parameters)
    cost = C(A3, Yt_train)
    # print(cost)

  [dW1, dW2, dW3, db1, db2, db3] = tape.gradient(cost, [W1, W2, W3, b1, b2, b3])

  # if (tf.norm(dC_dw) < thrsh and tf.norm(dC_db) < thrsh):
  #     break
  # print(type(lamda))
  # print(dW1)
  # print(lamda*dW1)
  W1 = tf.Variable((W1 - (lamda * dW1)), name='W1', dtype='float64')
  b1 = tf.Variable((b1 - (lamda * db1)), name='b1', dtype='float64')
  W2 = tf.Variable((W2 - (lamda * dW2)), name='W2', dtype='float64')
  b2 = tf.Variable((b2 - (lamda * db2)), name='b2', dtype='float64')
  W3 = tf.Variable((W3 - (lamda * dW3)), name='W3', dtype='float64')
  b3 = tf.Variable((b3 - (lamda * db3)), name='b3', dtype='float64')
  # print(W1)
  i += 1


