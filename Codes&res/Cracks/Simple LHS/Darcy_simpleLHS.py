import numpy as np
import scipy.io
from pyDOE import lhs
import time
import tensorflow as tf


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
        m = 100 * (u_tt + u_xx)
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
N_f = 100000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('D:/MyMatProgram/PINNdataset/data3.mat')
X_star = data['p_inside']
u_star = data['permeability100']
lb = X_star.min(0)
ub = X_star.max(0)

X_u_train, u_train = genBoundData(N_u)
# X_f_train = np.vstack((X_f_train, X_u_train))

sample = lhs(2, N_f)
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
X_f_train = np.concatenate((sample1, sample2), axis=0)
separation = len(index_final)

model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, separation)
# X_u:边界上的100个样本，u:对应100个样本的答案，X_f:内点样本
start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

u_pred, f_pred = model.predict(X_star)

np.savetxt("simpleLHS100_1000_10w.csv", u_pred, delimiter=',')

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))

# k50
# Training time: 161.2327
# Error u: 9.292513e-02


# k100_1w
# Training time:241.0195
# Error u:1.428678e-01

# k100_1000_4w
# Training time:1694.1756
# Error u:1.276293e-01

# k100_1000_10w
# Training time:4119.0064
# Error u: 1.328698e-01


# k250
# Training time: 139.3364
# Error u: 2.205777e-01


# k500_4w
# Error u: 4.592168e-01

# k500_10w
# Error u: 2.968605e-01

