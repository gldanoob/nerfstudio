import matplotlib.pyplot as plt
import numpy as np

money_data = np.load('results/money_no_vase_heuristic.npz')
tree_data = np.load('results/tree_no_vase_heuristic.npz')

distances = np.concatenate((money_data['distances'], tree_data['distances']))
predicted = np.concatenate((money_data['predicted'], tree_data['predicted']))

fig, ax = plt.subplots(figsize=(16, 8), dpi=400)

# use log scale on x axis
# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
max_dist = np.max(distances)

ax.hist(distances[predicted], range=(0, max_dist), bins=500)
ax.hist(distances[~predicted], range=(0, max_dist), bins=500)

ax.legend(['Leaf', 'Non-leaf'])
ax.set_xlabel('Mesh-to-GT CD (10^-4)')

plt.savefig(f'results/leaf_heuristic.png', bbox_inches='tight', dpi=200)
plt.close(fig)