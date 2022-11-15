import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting import newfig, savefig


def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    return data


# name = 'adaptiveLHS100_5w'
# pred = load_csv(name + '.csv')

data = scipy.io.loadmat('haha.mat')
p = data['p']
pred = data['u']
grid_x, grid_y = np.mgrid[0:1:500j, 0:1:500j]

U_pred = griddata(p, pred.flatten(), xi=(grid_x, grid_y), method='cubic')

fig, ax = newfig(1.0, 1)
ax.axis('off')
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1, bottom=0, left=0, right=1, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[0, 1, 0, 1],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

# savefig('D:\Python\程序\PINN\Pic\\' + name)
savefig('D:\Python\程序\PINN\Pic\\' + 'DP_k1')
