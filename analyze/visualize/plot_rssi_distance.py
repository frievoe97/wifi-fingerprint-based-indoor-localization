import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import seaborn as sns

plt.rcParams.update({'font.size': 7})

sns.set(style="whitegrid")

n = 2.013
C = -49.99

def calculate_rssi(d, n, C):
    return -10 * n * np.log10(d) + C

distances = np.linspace(0.1, 140, 200)
rssi_values = calculate_rssi(distances, n, C)

viridis = cm.viridis
color = viridis(0)

plt.figure(figsize=(12, 4))
sns.lineplot(x=distances, y=rssi_values, color=color)
plt.xlabel('Distanz (m)')
plt.ylabel('RSSI (dBm)')
plt.title('Pfadverlustmodell')
# plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig('plot_rssi_distance.png', dpi=300)

# Plot anzeigen
plt.show()
