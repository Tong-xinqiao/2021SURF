import numpy as np
import scipy.io
from pyDOE import lhs
import time
import tensorflow as tf
#    np.set_printoptions(threshold=np.inf)


class PhysicsInformedNN:
    # Initialize the class
    # X_u:边界上的100个样本，u:对应100个样本的答案，X_f:10000个内点样本+所有边界样本
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:, 0:1]  # 100个边界样本的x值
        self.t_u = X_u[:, 1:2]  # 100个边界样本的t值
        self.x_f = X_f[:, 0:1]  # 所有样本的x值
        self.t_f = X_f[:, 1:2]  # 所有样本的t值

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
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

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

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_tt + u_xx
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
    num = int(number/4)
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
    array_2 = np.zeros([3*num, 1])
    array_a = np.vstack((array_1, array_2))
    return array, array_a


N_u = 100
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
data = scipy.io.loadmat('D:/MyMatProgram/PINN/data.mat')
X_star = data['p_inside']
u_star = data['permeability']
lb = X_star.min(0)
ub = X_star.max(0)
X_f_train = lb + (ub - lb) * lhs(2, N_f)    # 抽内点

X_u_train, u_train= genBoundData(N_u)
X_f_train = np.vstack((X_f_train, X_u_train))

model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
# X_u:边界上的100个样本，u:对应100个样本的答案，X_f:内点样本+所有边界样本
start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

u_pred, f_pred = model.predict(X_star)
#np.savetxt("Prediction_k1.csv", u_pred, delimiter=',')
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))


