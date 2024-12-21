import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev, splrep
import pandas as pd
import os

def read_point(pred_path, gt_path, value):
    points_pred = pd.read_csv(pred_path).values[:, 1:]
    points_gt = pd.read_csv(gt_path).values[:, 1:]

    gt_max = np.array([-np.inf, -np.inf, -np.inf])
    gt_min = np.array([np.inf, np.inf, np.inf])
    for idx in range(len(points_pred)):
        gt_max = np.maximum(gt_max, points_gt[idx])  # 5.52
        gt_min = np.minimum(gt_min, points_gt[idx])  # -0.644

    x1 = points_pred[:, 0]
    y1 = points_pred[:, 1]
    z1 = points_pred[:, 2]


    x = points_gt[:, 0]
    y = points_gt[:, 1]
    z = points_gt[:, 2]

    ground_truth_timestamps = np.arange(len(points_pred))
    spl_x = splrep(ground_truth_timestamps, x, s=0)
    spl_y = splrep(ground_truth_timestamps, y, s=0)
    spl_z = splrep(ground_truth_timestamps, z, s=0)
    x_new = []
    y_new = []
    z_new = []
    gt_time = [ground_truth_timestamps[i] for i in range(len(ground_truth_timestamps)) if i % 3 == 0]
    for ts in gt_time:
        x_new.append(splev(ts, spl_x))
        y_new.append(splev(ts, spl_y))
        z_new.append(splev(ts, spl_z))
    spl_x1 = splrep(ground_truth_timestamps, x1, s=0)
    spl_y1 = splrep(ground_truth_timestamps, y1, s=0)
    spl_z1 = splrep(ground_truth_timestamps, z1, s=0)

    x_new1 = []
    y_new1 = []
    z_new1 = []
    pred_time = [ground_truth_timestamps[i] for i in range(len(ground_truth_timestamps)) if
                 i % value == 0]
    for ts1 in pred_time:
        x_new1.append(splev(ts1, spl_x1))
        y_new1.append(splev(ts1, spl_y1))
        z_new1.append(splev(ts1, spl_z1))


    return (x_new, y_new, z_new, x_new1, y_new1, z_new1)

# the paper img. value is smooth factor
# x1_gt, y1_gt, z1_gt, x1, y1, z1 = read_point("result_csv/Mavic2_pred.csv","result_csv/Mavic2_gt.csv", value=39)
# x2_gt, y2_gt, z2_gt, x2, y2, z2 = read_point("result_csv/Mavic3_pred.csv","result_csv/Mavic3_gt.csv", value=36)
# x3_gt, y3_gt, z3_gt, x3, y3, z3 = read_point("result_csv/Phantom4_pred.csv","result_csv/Phantom4_gt.csv", value=37)
# x4_gt, y4_gt, z4_gt, x4, y4, z4 = read_point("result_csv/M300_pred.csv","result_csv/M300_gt.csv", value=51)
# x4_gt, y4_gt, z4_gt, x4, y4, z4 = read_point("result_csv/Avata_pred.csv","result_csv/Avata_gt.csv")

x1_gt, y1_gt, z1_gt, x1, y1, z1 = read_point("result_csv/Mavic2_pred.csv","result_csv/Mavic2_gt.csv", value=45)
x2_gt, y2_gt, z2_gt, x2, y2, z2 = read_point("result_csv/Mavic3_pred.csv","result_csv/Mavic3_gt.csv", value=44)
x3_gt, y3_gt, z3_gt, x3, y3, z3 = read_point("result_csv/Phantom4_pred.csv","result_csv/Phantom4_gt.csv", value=44)
x4_gt, y4_gt, z4_gt, x4, y4, z4 = read_point("result_csv/M300_pred.csv","result_csv/M300_gt.csv", value=49)

fig = plt.figure(figsize=(12, 12))

ax2 = fig.add_subplot(221, projection='3d')
ax2.set_facecolor('white')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

# ax2.plot(x1, y1, z1, color='r')
ax2.scatter(x1_gt, y1_gt, z1_gt, color='crimson', label='Ground Truth', s=2)
ax2.plot(x1, y1, z1 , color='deepskyblue', label='Estimation', linewidth=1.5)
ax2.set_title('Mavic2')
ax2.grid(False)  #

ax3 = fig.add_subplot(222, projection='3d')
ax3.set_facecolor('white')
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
# ax3.plot(x2, y2, z2, color='g')
ax3.scatter(x2_gt, y2_gt, z2_gt, color='crimson', label='Ground Truth', s=2)
ax3.plot(x2, y2, z2, color='deepskyblue', label='Estimation', linewidth=1.5)
ax3.set_title('Mavic3')
ax3.grid(False)

ax4 = fig.add_subplot(223, projection='3d')
ax4.set_facecolor('white')
# ax4.plot(x3, y3, z3, color='b')
ax4.scatter(x3_gt, y3_gt, z3_gt, color='crimson', label='Ground Truth', s=2)
ax4.plot(x3, y3, z3, color='deepskyblue', label='Estimation', linewidth=1.5)
ax4.set_title('Phantom4')
ax4.grid(False)  #
ax4.xaxis.pane.fill = False  # remove x axis background
ax4.yaxis.pane.fill = False  # remove y axis background
ax4.zaxis.pane.fill = False  # remove z axis background
ax1 = fig.add_subplot(224, projection='3d')
ax1.set_facecolor('white')

# ax1.plot(x4, y4, z4, color='m')
ax1.scatter(x4_gt, y4_gt, z4_gt, color='crimson', label='Ground Truth', s=2)
ax1.plot(x4, y4, z4, color='deepskyblue', label='Estimation', linewidth=1.5)
ax1.set_title('M300')
ax1.grid(False)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig('4uav.svg')
plt.savefig('4uav.png')

# plt.show()