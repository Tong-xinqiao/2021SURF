import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    return data


# points = load_csv('D:/Python/程序/PINN/Cracks/自适应加点/打出误差小的点/adaptiveLHS500Sample_1.5w_LargeErrorInSteps.csv')
# plt.title("10000 Training Data Distribution_adaptive")
# plt.xlim(xmax=1, xmin=0)
# plt.ylim(ymax=1, ymin=0)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.scatter(points[0:10000, 0], points[0:10000, 1], s=1, marker=".")
# plt.savefig('step1.jpg')
# for i in range(10):
#     plt.title("%d Training Data Distribution_adaptive" % (10000 + 500 * (i + 1)))
#     plt.scatter(points[10000 + 500 * i:10000 + 500 * (i + 1), 0], points[10000 + 500 * i:10000 + 500 * (i + 1), 1],
#                 s=1, marker=".")
#     plt.savefig("step%d.jpg" % (i + 2))
# plt.show()
#
points = load_csv('D:/Python/程序/PINN/Cracks/自适应加点/打出误差小的点/adaptiveLHS500Sample_5w_SmallError.csv')
plt.xlim(xmax=1, xmin=0)
plt.ylim(ymax=1, ymin=0)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(points[:, 0], points[:, 1], s=1, marker=".")
plt.savefig('Trainingdata_2.jpg')
plt.show()