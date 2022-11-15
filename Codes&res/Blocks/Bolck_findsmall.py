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

    # def net_f(self, x, t):
    #     u = self.net_u(x, t)
    #     u_t = tf.gradients(u, t)[0]
    #     u_tt = tf.gradients(u_t, t)[0]
    #     u_x = tf.gradients(u, x)[0]
    #     u_xx = tf.gradients(u_x, x)[0]
    #     f = u_tt + u_xx
    #     return f

    def net_f(self, x, t):
        f = [[0]]
        karr5 = [4.7975, 3.9610, 4.5787, 2.1088, 4.6700, 4.2456, 0.1786, 3.2787, 1.9611,
                 3.7156, 3.7887, 3.3936, 0.1592, 3.5302, 0.8559, 3.274]
        karr20 = [18.2675, 2.5397, 18.1158, 16.2945, 10.9376, 5.5700, 1.9508, 12.6472, 19.4119,
                  3.1523, 19.2978, 19.1501, 2.8377, 16.0056, 9.7075, 19.1433]
        karr1000 = [823.4578, 97.1318, 46.1714, 276.9230, 34.4461, 950.2220, 317.0995, 694.8286, 795.1999, 765.5168,
                    381.5585, 438.7444, 646.3130, 445.5862, 489.7644, 186.8726]
        karr500 = [168.5613, 5.9510, 234.6953, 284.4118, 264.2666, 155.6075, 397.1423, 81.0912, 327.0395, 131.4856,
                   300.9910, 82.8244, 41.9107, 225.2708, 374.0758, 344.6073]
        karr250 = [227.6619, 107.8535, 200.0171, 64.9676, 34.0171, 36.3847, 65.9507, 45.4618, 36.2387, 137.4651,
                   144.9261, 217.3231, 128.3124, 87.7381, 155.5138, 213.2578]
        karr100 = [12.3319, 23.9916, 7.5967, 40.1808, 4.9654, 41.7267, 23.9953, 18.3908, 48.9253, 49.0864, 94.4787,
                   90.2716, 11.1203, 36.9247, 90.0054, 33.7719]
        karr75 = [61.7593, 7.2849, 3.4629, 20.7692, 2.5835, 71.2667, 23.7825, 52.1121, 59.6400, 57.4138, 28.6169,
                  32.9058, 48.4735, 33.4190, 36.7323, 14.0154]
        karr50 = [45.6688, 6.3493, 45.2896, 40.7362, 27.3441, 13.9249, 4.8770, 31.6180, 48.5296, 7.8807, 48.2444,
                  47.8753, 7.0943, 40.0140, 24.2688, 47.8583]

        num = int(N_f / 16)
        for i in range(0, 16):
            xx = x[i * num:(i + 1) * num]
            tt = t[i * num:(i + 1) * num]
            u1 = self.net_u(xx, tt)
            u_t = tf.gradients(u1, tt)[0]
            u_tt = tf.gradients(u_t, tt)[0]
            u_x = tf.gradients(u1, xx)[0]
            u_xx = tf.gradients(u_x, xx)[0]
            m = karr500[i] * (u_tt + u_xx)
            f = tf.concat([f, m], 0)
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


def latinHypercubeSample(A, B, number):
    M = [[0, 0]]
    num = int(number / 16)
    for i in range(0, 16):
        lb = A[B[i][0]:B[i + 1][0], :].min(0)
        ub = A[B[i][0]:B[i + 1][0], :].max(0)
        m = lb + (ub - lb) * lhs(2, num)
        M = np.vstack((M, m))
    M = np.delete(M, 0, axis=0)
    return M


N_u = 100
N_f = 40000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('D:/MyMatProgram/PINNdataset/data2.mat')
partition = data['partitioned']
index = data['rowOfDivision']
X_star = data['p_inside']
lb = X_star.min(0)
ub = X_star.max(0)

X_f_train = latinHypercubeSample(partition, index, N_f)  # 抽内点
X_u_train, u_train = genBoundData(N_u)
# X_f_train = np.vstack((X_f_train, X_u_train))

model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
# X_u:边界上的100个样本，u:对应100个样本的答案，X_f:内点样本
start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

u_pred, f_pred = model.predict(X_f_train)
difference = list(abs(f_pred))
least1w = map(difference.index, heapq.nsmallest(5000, difference))
Sample = np.array([[0, 0]])
for i in least1w:
    Sample = np.concatenate((Sample, [X_star[i]]), axis=0)
Sample = np.delete(Sample, 0, axis=0)
plt.title(" Top 5000 Training Points with Smallest Error")
plt.xlim(xmax=1, xmin=0)
plt.ylim(ymax=1, ymin=0)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(Sample[:, 0], Sample[:, 1], s=1, marker=".")
plt.savefig('Block500算得好的5k点.jpg')
plt.show()
