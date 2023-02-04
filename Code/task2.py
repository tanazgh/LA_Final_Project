import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# visualizing 5d data
raw_data = np.load('../Data/data2d.npz')
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

fig = plt.figure()

def visualize():
    Xt_total = tf.transpose(Xt)
    Yt_total = Yt.numpy().reshape((Xt.shape[1]))
    Yt_prediction_total = Yt_prediction.reshape((X.shape[1]))
    Xt_prim = Xt_total[Yt_prediction_total - Yt_total == 0]
    Yt_prim = Yt_total[Yt_prediction_total - Yt_total == 0]
    c1 = Xt_prim[Yt_prim == 0]
    plt.scatter(c1[:, 0], c1[:, 1], facecolor='b')
    c2 = Xt_prim[Yt_prim == 1]
    plt.scatter(c2[:, 0], c2[:, 1], facecolor='g')
    z = Xt_total[Yt_prediction_total - Yt_total != 0]
    plt.scatter(z[:, 0], z[:, 1], facecolor='none', edgecolor='black')


lamda = 0.009
thrsh = 0.1

w = tf.Variable(tf.zeros([X.shape[0], 1], dtype='float64'), name='w')
b = tf.Variable(0.0, dtype='float64')

data = (Xt, Yt)
Yt_prediction = np.zeros([1, Xt.shape[1]], dtype='float64')

while True:
    with tf.GradientTape(persistent=True) as tape:
        cost = C(w, b, data)
        print(cost)

    A = Phi(Xt, w, b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Yt_prediction[0, i] = 1
        else:
            Yt_prediction[0, i] = 0
    x_list = np.linspace(-4, 4, 100)
    y_list = (w[0][0] * x_list + b) / (-w[1][0])
    visualize()
    plt.plot(x_list, y_list, '-r')
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('plot', img)

    fig.clear()
    # press any key to exit
    key = cv2.waitKey(33)
    if key == 27:
        break
    [dC_dw, dC_db] = tape.gradient(cost, [w, b])
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
