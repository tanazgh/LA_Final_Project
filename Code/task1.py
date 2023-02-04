import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import cv2

# # # visualizing 2d data
raw_data = np.load('../Data/data2d.npz')
X = raw_data['X']
Y = np.array(raw_data['y'])

# z = X[Y == 1]
# plt.plot(z[:, 0], z[:, 1], 'bo')
# z = X[Y == 0]
# plt.plot(z[:, 0], z[:, 1], 'go')
# plt.show()

# visualizing 5d data
# raw_data = np.load('data5d.npz')
# X = raw_data['X']
# Y = np.array(raw_data['y'])

# z = X[Y == 1]
# plt.plot(z[:, 0], z[:, 1], 'bo')
# z = X[Y == 0]
# plt.plot(z[:, 0], z[:, 1], 'go')
# plt.show()

print("Pre:")
print(X.shape)
print(Y.shape)
#Reshape Data:
X = X.T
Y = Y.reshape(1, Y.shape[0])
print("Reshaped:")
print(X.shape)
print(Y.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Phi(X, w, b):
    return sigmoid(np.dot(w.T, X) + b)


def C(w, b, data):
    X, Y = data
    A = Phi(X, w, b)
    return -1 * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))


def compute_dC_dw(w, b, data):
    X, Y = data
    A = Phi(X, w, b)
    return np.dot(X, (A - Y).T)


def compute_dC_db(w, b, data):
    X, Y = data
    A = Phi(X, w, b)
    return np.sum(A - Y)

def compute_dC_dw_numeric(w,b, data, epsilon=1e-7):
    C_plus = np.zeros((w.shape[0], 1))
    C_minus = np.zeros((w.shape[0], 1))
    for i in range (w.shape[0]):
        theta_plus = np.copy(w)
        theta_plus[i] = theta_plus[i] + epsilon
        C_plus[i] = C(theta_plus, b, data)
        theta_minus = np.copy(w)
        theta_minus[i] = theta_minus[i] - epsilon
        C_minus[i] = C(theta_minus, b, data)
    return (C_plus-C_minus)/(2*epsilon)

def compute_dC_db_numeric(w,b, data, epsilon=1e-7):
    theta_plus = b + epsilon
    C_plus = C(w, theta_plus, data)
    theta_minus = b - epsilon
    C_minus = C(w, theta_minus, data)
    return (C_plus-C_minus)/(2*epsilon)

data = (X,Y)

fig = plt.figure()

lamda = 0.009
thrsh = 0.1

w = np.zeros((X.shape[0], 1))
b = 0
i=0

def visualize():
    X_total = X.T
    Y_total = Y.reshape(X.shape[1])
    Y_prediction_total = Y_prediction.reshape(X.shape[1])
    X_prim = X_total[Y_prediction_total - Y_total == 0]
    Y_prim = Y_total[Y_prediction_total - Y_total == 0]
    c1 = X_prim[Y_prim == 0]
    plt.scatter(c1[:, 0], c1[:, 1], facecolor='b')
    c2 = X_prim[Y_prim == 1]
    plt.scatter(c2[:, 0], c2[:, 1], facecolor='g')
    z = X_total[Y_prediction_total - Y_total != 0]
    plt.scatter(z[:, 0], z[:, 1], facecolor='none', edgecolor='black')

while True:
    cost = C(w, b, data)
    A = Phi(X, w, b)
    print(cost)
    Y_prediction = np.zeros((1,X.shape[1]))
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
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
    dC_dw = compute_dC_dw(w,b, data)
    dC_db = compute_dC_db(w,b, data)
    if (np.linalg.norm(dC_dw) < thrsh and np.linalg.norm(dC_db) < thrsh):
        break
    w = w - lamda * dC_dw
    b = b - lamda * dC_db