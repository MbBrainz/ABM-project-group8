#%%
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from polarization.model import CityModel
import numpy as np
import matplotlib as plt
import random
random.seed(102923)
#%%
# i don't really want to have to run the model again, I just want data from the data collector
model = CityModel(50, 10, 10)
for i in range(20):
    model.step()

agent_opinion = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content,x,y = cell
    #not sure exactly how to call the opinion attribute
    agent_opinion = x.opinion
    grid_opinion[x][y] = agent_opinion

plt.set_cmap('bwr')
plt.imshow(grid_opinion, interpolation='nearest')
plt.colorbar()
plt.show()
#%%
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

#%%
# agent_opinion=[]
# for i in range(100):
#     opinion = random.random()
#     agent_opinion.append(opinion)

# Grid = [[random.random()for c in range(10)] for r in range(10)]

# color_map=[]
# colors=LinearSegmentedColormap.from_list('name', ['blue','red'])
# cmappable = ScalarMappable(norm=Normalize(0,1), cmap=colors)
#
#trying to same as Noah but it ain't gonna work I don't think
# for x in Grid:
#     rgba=colors(opinions[x])
#     color_map.appen(rgba)

# cbar=fig.colorbar(cmappable, orientation='horizontal', label='Opinion', ticks=[0,1])
# cbar.ax.set_xticklabels(['Far Left', 'Far Right'])
# plt.colorbar()
# plt.show()
#%%
#random generate of grid to see how it would look
opinions = np.random.rand(10, 10)
x = np.arange(0, 11, 1)  # len = 10
y = np.arange(0, 11, 1)  # len = 10

#colors = LinearSegmentedColormap.from_list('name', ['blue','red'])
#cmappable = ScalarMappable(norm=Normalize(0,1), cmap=colors)

fig= plt.figure()
plt.set_cmap('bwr')
plt.pcolormesh(x, y, opinions, vmin=0, vmax=1)
plt.colorbar()
# %%
