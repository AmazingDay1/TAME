from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

colors = [(1, 1, 1), (0.53, 0.81, 0.92)]
cmap = LinearSegmentedColormap.from_list('white_to_skyblue', colors, N=100)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.title('Confusion Matrix')
    # plt.show()
    plt.savefig('visaulization/figure/{}.svg'.format("confusionmatrix"))
    plt.show()
    plt.close()

cm_normalized = np.array([[0.97231834, 0.01038062, 0.00692042, 0.,         0.01038062],
 [0.01355333, 0.96994697, 0.00766058, 0.,         0.00883913],
 [0.00688559, 0.00476695, 0.98358051, 0.,         0.00476695],
 [0.,         0.,         0.,         1.,         0.,        ],
 [0.01461988, 0.00877193, 0.00633528, 0.,         0.9702729 ]])

cm_normalized = np.round(cm_normalized, 3)
print("Normalized Confusion Matrix:\n", cm_normalized)

fig, ax = plt.subplots()
cax = ax.matshow(cm_normalized, cmap=cmap)
# fig.colorbar(cax)

classes = ["Mavic2", "Mavic3", "Phantom4", "Avata", "M300"]
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.spines[:].set_visible(False)
cbar = fig.colorbar(cax)
cbar.outline.set_visible(False)

for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        ax.text(j, i, format(cm_normalized[i, j], '.3f'),
                ha="center", va="center", color="black")

ax.set_ylabel('Ground Truth')
ax.set_xlabel('Prediction')
plt.savefig('conf.svg')
plt.show()
