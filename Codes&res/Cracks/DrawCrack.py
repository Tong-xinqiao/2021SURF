import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

fig, ax = plt.subplots()
xy = np.array([0.05, 0.1])
xy1 = np.array([0.02, 0.8])
# 圆形
# circle = mpathes.Circle(xy1, 0.05)
# ax.add_patch(circle)
# 长方形
rect = mpathes.Rectangle(xy, 0.01, 0.8)
rect2 = mpathes.Rectangle(xy1, 0.8, 0.01)
ax.add_patch(rect)
ax.add_patch(rect2)
# 多边形
# polygon = mpathes.RegularPolygon(xy3, 5, 0.1, color='g')
# ax.add_patch(polygon)
# 椭圆形
# ellipse = mpathes.Ellipse(xy4, 0.4, 0.2, color='y')
# ax.add_patch(ellipse)

# plt.axis('equal')
plt.xlim(xmax=1, xmin=0)
plt.ylim(ymax=1, ymin=0)
plt.savefig('CrackShape.jpg')
plt.show()
