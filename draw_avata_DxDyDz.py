import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev, splrep
import pandas as pd
import os

t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
z = t

def extract_timestamp(filename):
    return float(filename.rsplit('.', 1)[0])
def draw_two_line(pred_path, gt_path, cls_name, image_folder):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    t = [extract_timestamp(f) for f in image_files]
    t = [t[i] for i in range(len(t)) if i%3 == 0]

    points_pred = pd.read_csv(pred_path).values[:, 1:]
    points_gt = pd.read_csv(gt_path).values[:, 1:]

    gt_max = np.array([-np.inf, -np.inf, -np.inf])
    gt_min = np.array([np.inf, np.inf, np.inf])
    for idx in range(len(points_pred)):
        gt_max = np.maximum(gt_max, points_gt[idx])  # 5.52
        gt_min = np.minimum(gt_min, points_gt[idx])  # -0.644

    ground_truth_timestamps = np.arange(len(points_pred))
    x1 = points_pred[:, 0]
    y1 = points_pred[:, 1]
    z1 = points_pred[:, 2]

    x = points_gt[:, 0]
    y = points_gt[:, 1]
    z = points_gt[:, 2]

    spl_x = splrep(ground_truth_timestamps, x, s=0)
    spl_y = splrep(ground_truth_timestamps, y, s=0)
    spl_z = splrep(ground_truth_timestamps, z, s=0)
    interpolated_poses = []
    x_new = []
    y_new = []
    z_new = []
    gt_time = [ground_truth_timestamps[i] for i in range(len(ground_truth_timestamps)) if i % 3 == 0]
    for ts in gt_time:
        x_new.append(splev(ts, spl_x))
        y_new.append(splev(ts, spl_y))
        z_new.append(splev(ts, spl_z))

    x_new_l = []
    y_new_l = []
    z_new_l = []
    gt_time = [ground_truth_timestamps[i] for i in range(len(ground_truth_timestamps)) if i % 35 == 0]
    for ts in gt_time:
        x_new_l.append(splev(ts, spl_x))
        y_new_l.append(splev(ts, spl_y))
        z_new_l.append(splev(ts, spl_z))

    spl_x1 = splrep(ground_truth_timestamps, x1, s=0)
    spl_y1 = splrep(ground_truth_timestamps, y1, s=0)
    spl_z1 = splrep(ground_truth_timestamps, z1, s=0)
    interpolated_poses = []
    x_new1 = []
    y_new1 = []
    z_new1 = []
    pred_time = [ground_truth_timestamps[i] for i in range(len(ground_truth_timestamps)) if
                 i % 35 == 0]
    for ts1 in pred_time:
        x_new1.append(splev(ts1, spl_x1))
        y_new1.append(splev(ts1, spl_y1))
        z_new1.append(splev(ts1, spl_z1))

    for i in range(len(x_new_l)):
        x_new_l[i] = np.abs(x_new1[i] - x_new_l[i])
        y_new_l[i] = np.abs(y_new1[i] - y_new_l[i])
        z_new_l[i] = np.abs(z_new1[i] - z_new_l[i])

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x_new, y_new, z_new, color='crimson', label='Ground Truth', s=2)
    ax.plot(x_new1, y_new1, z_new1, color='deepskyblue', label='Estimation', linewidth=1.5)
    # ax.scatter(x2, y2, z2, color='limegreen', label='Estimation', s=1)
    ax.set_title('Avata', pad=-80)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.grid(False)

    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax2 = fig.add_subplot(322)
    ax2.plot(pred_time, x_new_l, color='mediumseagreen', label=r'$\mathit{D}_{x}$', linewidth=1.5)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('X (m)')
    ax2.set_facecolor('white')

    ax3 = fig.add_subplot(324)
    ax3.plot(pred_time, y_new_l, color='mediumseagreen', label=r'$\mathit{D}_{y}$', linewidth=1.5)
    ax3.legend(loc='upper right')
    ax3.set_ylabel('Y (m)')
    ax3.set_facecolor('white')

    ax4 = fig.add_subplot(326)
    ax4.plot(pred_time,  z_new_l, color='mediumseagreen', label=r'$\mathit{D}_{z}$', linewidth=1.5)
    ax4.legend(loc='upper right')
    ax4.set_ylabel('Z (m)')
    ax4.set_xlabel('Time (s)')
    ax4.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('avata2.svg'.format(cls_name))
    # plt.show()

draw_two_line("result_csv/Avata_pred.csv", "result_csv/Avata_gt.csv", "Avata", '/media/xiao/HIKSEMI/datasets/MMAUD/raw_data/audio/Avata/Avata/image')
