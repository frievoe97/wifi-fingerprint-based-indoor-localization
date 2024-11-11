import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def sorensen_distance(x, y):
    return np.sum(np.abs(x - y)) / np.sum(np.abs(x) + np.abs(y) + 1e-10)

axis_values = np.linspace(0, 100, 100)

results = {
    "euclidean": np.zeros((100, 100)),
    "sorensen": np.zeros((100, 100))
}

for i, x_val in enumerate(axis_values):
    for j, y_val in enumerate(axis_values):
        x = np.array([x_val])
        y = np.array([y_val])
        results["euclidean"][i, j] = euclidean_distance(x, y)
        results["sorensen"][i, j] = sorensen_distance(x, y)

axis_ticks = np.arange(0, 101, 10)
axis_labels = np.round(np.linspace(0, -100, len(axis_ticks)), 1)

plt.rcParams.update({'font.size': 7})

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, sharex=True)

col_titles = ["Euclidean Distance", "Sorensen Distance"]

for col, distance in enumerate(col_titles):
    ax = axes[col]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    sns.heatmap(np.flipud(results[distance.split()[0].lower()]), ax=ax, xticklabels=axis_ticks, yticklabels=axis_ticks, cmap="viridis", square=True, cbar_ax=cax)
    ax.set_title(col_titles[col])
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    ax.set_xticks(axis_ticks)
    ax.set_xticklabels(axis_labels)
    ax.set_yticks(axis_ticks)
    ax.set_yticklabels(axis_labels[::-1])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.tight_layout(rect=(0, 0, 1, 1 ))

plt.savefig('plot_euclidean_vs_sorensen.png', dpi=300)

plt.show()
