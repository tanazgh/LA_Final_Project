import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# visualizing 5d data
raw_data = np.load('../Data/data5d.npz')
X = raw_data['X']
Y = np.array(raw_data['y'])

print("Pre:")
print(X.shape)
print(Y.shape)
#Reshape Data:
X = X.T
Y = Y.reshape(1, Y.shape[0])
print("Reshaped:")
print(X.shape)
print(Y.shape)

Xt = tf.Variable(X, name='Xt')
Yt = tf.Variable(Y, name='Yt')


def sigmoid(z):
    return 1 / (1 + tf.math.exp(-z))


def Phi(X, w, b):
    return sigmoid(tf.linalg.matmul(tf.transpose(w), X) + b)


def C(w, b, data):
    X, Y = data
    A = Phi(X, w, b)
    #     print(A)
    return -1 * tf.math.reduce_sum(Y * tf.math.log(A) + (1 - Y) * (tf.math.log(1 - A)))


lamda = 0.009
thrsh = 0.1

w = tf.Variable(tf.zeros([X.shape[0], 1], dtype='float64'), name='w')
b = tf.Variable(0.0, dtype='float64')

data = (Xt, Yt)

while True:
    with tf.GradientTape(persistent=True) as tape:
        cost = C(w, b, data)
        print(cost)

    [dC_dw, dC_db] = tape.gradient(cost, [w, b])

    if (tf.norm(dC_dw) < thrsh and tf.norm(dC_db) < thrsh):
        break
    w = tf.Variable(w - lamda * dC_dw)
    b = tf.Variable(b - lamda * dC_db)

Yt_prediction = np.zeros([1,Xt.shape[1]], dtype='float64')
A = Phi(Xt, w, b)
print(Yt_prediction)
for i in range(A.shape[1]):
    if A[0,i]>0.5:
        Yt_prediction[0,i] = 1
    else:
        Yt_prediction[0,i] = 0

Yt_prediction = tf.convert_to_tensor(Yt_prediction)
print(Yt)
print(Yt_prediction)
print("train accuracy: {} %".format(100 - tf.math.reduce_mean(tf.math.abs(Yt_prediction - Yt)) * 100))

x_list = tf.linspace(-4, 4, 100)
y_list = (w[0][0]*x_list + b)/(-w[1][0])
Y_prim = tf.reshape(Yt, Xt.shape[1])
X_prim = tf.transpose(Xt)
z = X_prim[Y_prim == 1]
plt.plot(z[:, 0], z[:, 1], 'bo')
z = X_prim[Y_prim == 0]
plt.plot(z[:, 0], z[:, 1], 'go')
# z = X_prim[np.abs(Y_prediction - Y).reshape(70) == 1]
# plt.plot(z[:, 0], z[:, 1], 'ro')
plt.plot(x_list, y_list)
plt.show()