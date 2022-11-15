import numpy as np
import scipy.io
from pyDOE import lhs
import time
import tensorflow as tf
import heapq
import matplotlib.pyplot as plt


# np.set_printoptions(threshold=np.inf)


class PhysicsInformedNN:
    # Initialize the class
    # X_u:边界上的100个样本，u:对应100个样本的答案，X_f:10000个内点样本
    def __init__(self, X_u, u, X_f, layers, lb, ub, sep):
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:, 0:1]  # 100个边界样本的x值
        self.t_u = X_u[:, 1:2]  # 100个边界样本的t值
        self.x_f = X_f[:, 0:1]  # 所有样本的x值
        self.t_f = X_f[:, 1:2]  # 所有样本的t值
        self.sep = sep
        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf, self.sep)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, y, num):
        xincrack = x[0:num, :]
        yincrack = y[0:num, :]
        u1 = self.net_u(xincrack, yincrack)
        u_t = tf.gradients(u1, yincrack)[0]
        u_tt = tf.gradients(u_t, yincrack)[0]
        u_x = tf.gradients(u1, xincrack)[0]
        u_xx = tf.gradients(u_x, xincrack)[0]
        m = 500 * (u_tt + u_xx)
        otherx = x[num:, :]
        othery = y[num:, :]
        u2 = self.net_u(otherx, othery)
        u_t = tf.gradients(u2, othery)[0]
        u_tt = tf.gradients(u_t, othery)[0]
        u_x = tf.gradients(u2, otherx)[0]
        u_xx = tf.gradients(u_x, otherx)[0]
        n = u_tt + u_xx
        f = tf.concat([m, n], 0)
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})
        return u_star, f_star


def genBoundData(number):
    num = int(number / 4)
    arr1 = np.random.rand(num, 1)
    arr2 = np.zeros([num, 1])
    arrA = np.hstack((arr2, arr1))
    arr3 = np.random.rand(num, 1)
    arr4 = np.ones([num, 1])
    arrB = np.hstack((arr4, arr3))
    arr5 = np.random.rand(num, 1)
    arrC = np.hstack((arr5, arr2))
    arr6 = np.random.rand(num, 1)
    arrD = np.hstack((arr6, arr4))
    array = np.vstack((arrA, arrB, arrC, arrD))
    array_1 = arr1 * arr1 * (-1 / 2) + arr1 / 2
    array_2 = np.zeros([3 * num, 1])
    array_a = np.vstack((array_1, array_2))
    return array, array_a


N_u = 1000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('D:/MyMatProgram/PINNdataset/data3.mat')
X_star = data['p_inside']
u_star = data['permeability500']
lb = X_star.min(0)
ub = X_star.max(0)

X_u_train, u_train = genBoundData(N_u)

#  制作最初样本集
sample = lhs(2, 10000)
index_final = []
index = np.where(np.logical_and(sample[:, 0] >= 0.02, sample[:, 0] <= 0.82))
x1 = (index[0]).T
index2 = np.where(np.logical_and(sample[:, 1] >= 0.8, sample[:, 1] <= 0.81))
y1 = (index2[0]).T
for i in x1:
    if i in y1:
        index_final.append(i)
index = np.where(np.logical_and(sample[:, 0] >= 0.05, sample[:, 0] <= 0.06))
x2 = (index[0]).T
index2 = np.where(np.logical_and(sample[:, 1] >= 0.1, sample[:, 1] <= 0.9))
y2 = (index2[0]).T
for i in x2:
    if i in y2:
        index_final.append(i)
sample1 = []
for i in index_final:
    sample1.append(sample[i])
sample2 = np.delete(sample, index_final, axis=0)
sample = np.concatenate((sample1, sample2), axis=0)
###########################################################
# adaptive sampling
separation = len(index_final)
X_f_train = sample
# X_f_train_2 = sample
start_time = time.time()
for i in range(20):
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, separation)
    model.train()
    newLHS = lhs(2, 10000)
    u_pred, f_pred = model.predict(newLHS)
    difference = list(abs(f_pred))
    count = 0
    top1000 = map(difference.index, heapq.nsmallest(2000, difference))
    for j in top1000:
        # X_f_train_2 = np.concatenate((X_f_train_2, [newLHS[i]]), axis=0)
        if (0.02 <= newLHS[j][0] <= 0.82 and 0.8 <= newLHS[j][1] <= 0.81) or (
                0.05 <= newLHS[j][0] <= 0.06 and 0.1 <= newLHS[j][1] <= 0.9):
            count = count + 1
            X_f_train = np.concatenate(([newLHS[j]], X_f_train), axis=0)
        else:
            X_f_train = np.concatenate((X_f_train, [newLHS[j]]), axis=0)
    separation = separation + count

np.savetxt("adaptiveLHS500Sample_5w_SmallError.csv", X_f_train, delimiter=',')
######################################################################
# application
model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, separation)
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

u_pred, f_pred = model.predict(X_star)

np.savetxt("adaptiveLHS500_5w.csv", u_pred, delimiter=',')

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))

plt.title("50000 Training Data Distribution_adaptive")
plt.xlim(xmax=1, xmin=0)
plt.ylim(ymax=1, ymin=0)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_f_train[:, 0], X_f_train[:, 1], s=1, marker=".")
plt.show()

# 2w
# Training time: 2154.8150
# Error u: 4.098740e-01