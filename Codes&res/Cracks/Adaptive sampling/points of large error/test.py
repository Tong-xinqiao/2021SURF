import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    return data


points = load_csv('D:/Python/程序/PINN/Cracks/自适应加点/打出误差大的点/adaptiveLHS500Sample_4w_LargeErrorInSteps.csv')
plt.title("10000 Training Data Distribution_adaptive")
plt.xlim(xmax=1, xmin=0)
plt.ylim(ymax=1, ymin=0)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(points[:, 0], points[:, 1], s=1, marker=".")
plt.title("50000 Training Data Distribution_adaptive")
plt.savefig('Trainingdata.jpg')

plt.show()

