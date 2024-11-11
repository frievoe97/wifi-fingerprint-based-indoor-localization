import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 7})

alpha = 24
beta = np.e

min_rssi = -101

def positive_values_representation(rssi_values, min_rssi):
    return rssi_values - min_rssi

def exponential_representation(rssi_values, min_rssi):
    positive_values = positive_values_representation(rssi_values, min_rssi)
    return np.exp(positive_values / alpha) / np.exp(min_rssi / alpha)

def powed_representation(rssi_values, min_rssi):
    positive_values = positive_values_representation(rssi_values, min_rssi)
    return (positive_values ** beta) / (abs(min_rssi) ** beta)

rssi_values = np.linspace(-100, 0, 100)

positive_values_correct = positive_values_representation(rssi_values, min_rssi)
exponential_values_correct = exponential_representation(rssi_values, min_rssi)
powed_values_correct = powed_representation(rssi_values, min_rssi)

positive_values_norm = positive_values_correct / np.max(positive_values_correct)
exponential_values_correct_norm = exponential_values_correct / np.max(exponential_values_correct)
powed_values_correct_norm = powed_values_correct / np.max(powed_values_correct)

palette = sns.color_palette("viridis", 3)

plt.figure(figsize=(12, 4))

sns.lineplot(x=rssi_values, y=positive_values_norm, label='Positive Scaled', linestyle='solid', color=palette[0])
sns.lineplot(x=rssi_values, y=exponential_values_correct_norm, label='Exponential Scaled', linestyle='solid', color=palette[1])
sns.lineplot(x=rssi_values, y=powed_values_correct_norm, label='Powed Scaled', linestyle='solid', color=palette[2])

plt.xlabel('Original RSSI values  (dBm)')
plt.ylabel('Scaled RSSI values')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plot_scaling_strategies.png', dpi=300)
plt.show()
