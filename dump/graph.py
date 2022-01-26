

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import numpy as np
import networkx as nx

N = 30
M = 2

G = nx.barabasi_albert_graph(n=N, m=M)
opinions = np.random.uniform(0,10,N)

color_map = []
colors = LinearSegmentedColormap.from_list('name', ['blue','red'])
cmappable = ScalarMappable(norm=Normalize(0,1), cmap=colors)

for node in G: # here we can just pass the final model Graph
    rgba = colors(opinions[node]/10)
    color_map.append(rgba)

fig, ax = plt.subplots(1)
nx.draw(G, ax=ax, node_size=300, node_color=color_map, width=0.5)

cbar = fig.colorbar(cmappable, orientation='horizontal',label="Opinion", ticks=[0,1])
cbar.ax.set_xticklabels(['Far left', 'Far right'])
#cb.set_label('Color Scale')
plt.show()

