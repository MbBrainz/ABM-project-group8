#%%
import os
import pandas as pd
import numpy as np
import networkx as nx

from random import uniform
from sqlite3 import Row

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# load data from pickle-files
path = './data' # this might need to change!
files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
         '_modeldf' in i or '_agentdf' in i]
print(files)

model_file = next((s for s in files if '_modeldf' in s), None)
agent_file = next((s for s in files if '_agentdf' in s), None)

agents = pd.read_pickle(f"./data/{agent_file}")
model = pd.read_pickle(f"./data/{model_file}")

# extract first and last step data
first_run_dict = model.loc[model.index[0],"edges"]
G_init = nx.from_dict_of_dicts(first_run_dict)

last_run_dict = model.loc[model.index[-1], "edges"]
G_last = nx.from_dict_of_dicts(last_run_dict)

# permute data
agents = agents.reset_index([0])
agents_firststep = agents[agents['Step'] == 1] # this can stay, first step is always 1
agents_laststep = agents[agents['Step'] == 10] # this is hardcoded!!!

# define coloring
color_map_first = []
color_map_last = []
cmap = plt.get_cmap('bwr')

for node in G_last: # same IDs in both graphs
    opinion_first = agents_firststep['opinion'][node]
    rgba = cmap(opinion_first/10) # maps from 0 to 1, so need to normalize!
    color_map_first.append(rgba)

    opinion_last = agents_laststep['opinion'][node]
    rgba = cmap(opinion_last/10) # maps from 0 to 1, so need to normalize!
    color_map_last.append(rgba)

# plot
fig, axs = plt.subplots(1,2)
nx.draw(G_init, ax=axs[0], node_size=100, node_color=color_map_first, width=0.3, edgecolors='k')
nx.draw(G_last, ax=axs[1], node_size=100, node_color=color_map_last, width=0.3, edgecolors='k')

axs[0].set_title('Initialized')
axs[1].set_title('After model run')

cbar = fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap=cmap), orientation='horizontal',label="Opinion", ticks=[0,1])
cbar.ax.set_xticklabels(['Far left', 'Far right'])

plt.show()
#%%

def create_graph(agents, model, params):
    first_run_dict = model.loc[model.index[0],"edges"]
    G_init = nx.from_dict_of_dicts(first_run_dict)

    last_run_dict = model.loc[model.index[-1], "edges"]
    G_last = nx.from_dict_of_dicts(last_run_dict)

# permute data
    agents = agents.reset_index([0])
    agents_firststep = agents[agents['Step'] == 1] # this can stay, first step is always 1
    agents_laststep = agents[agents['Step'] == params.total_steps] # this is hardcoded!!!

# define coloring
    color_map_first = []
    color_map_last = []
    cmap = plt.get_cmap('bwr')

    for node in G_last: # same IDs in both graphs
        opinion_first = agents_firststep['opinion'][node]
        rgba = cmap(opinion_first/10) # maps from 0 to 1, so need to normalize!
        color_map_first.append(rgba)

        opinion_last = agents_laststep['opinion'][node]
        rgba = cmap(opinion_last/10) # maps from 0 to 1, so need to normalize!
        color_map_last.append(rgba)

# plot
    fig, axs = plt.subplots(1,2)
    nx.draw(G_init, ax=axs[0], node_size=100, node_color=color_map_first, width=0.3, edgecolors='k')
    nx.draw(G_last, ax=axs[1], node_size=100, node_color=color_map_last, width=0.3, edgecolors='k')

    axs[0].set_title('Initialized')
    axs[1].set_title('After model run')

    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap=cmap), orientation='horizontal',label="Opinion", ticks=[0,1])
    cbar.ax.set_xticklabels(['Far left', 'Far right'])

    plt.show()

    # %%

# create_graph(agents, model)
