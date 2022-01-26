#%%
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from polarization.model import CityModel
#%%
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
random.seed(102923)
#%%
# i don't really want to have to run the model again, I just want data from the data collector
model = CityModel()
for i in range(20):
    model.step()

agent_opinion = np.zeros((10,10))
for cell in model.grid.coord_iter():
    cell_content,x,y = cell
    #not sure exactly how to call the opinion attribute
    agent_opinion = cell_content.opinion
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
# opinions = np.random.rand(10, 10)
# x = np.arange(0, 11, 1)  # len = 10
# y = np.arange(0, 11, 1)  # len = 10

#colors = LinearSegmentedColormap.from_list('name', ['blue','red'])
#cmappable = ScalarMappable(norm=Normalize(0,1), cmap=colors)

# plt.set_cmap('bwr')
# plt.pcolormesh(x, y, opinions, vmin=0, vmax=1)
# plt.colorbar()
# %%
#testing reading the file
df=pd.read_pickle("./test.pkl")
print(df["opinion"])
posdata=df["position"]
print(posdata)
# %%
df['position'].tolist()
pd.DataFrame(df['position'].tolist(),index=df.index)
df[['x','y']]=pd.DataFrame(df['position'].tolist(),index=df.index)
# %%
df.reset_index(level='AgentID')
print(df.tail())
# %%

# %%
#mesa way
#call these from the data file name
grid_width = 10
grid_height= 10
num_agents = 49
last_step=df[-49:]
print(last_step)
#%%
# last_run = df.loc[df.index[-1]]
#can make this cleaner but fine for now
x = df.loc[df.index[-49:], "x"]
y=df.loc[df.index[-49:], "y"]
opinion = df.loc[df.index[-49:], "opinion"]
print(opinion)
#%%
grid_opinion = np.zeros((grid_width,grid_height))
#for agent in AgentID:
for agent in range(num_agents):
    agent_opinion = opinion
    grid_opinion[x][y] = agent_opinion
print(grid_opinion)
#%%
plt.set_cmap('bwr')
plt.imshow(grid_opinion, interpolation='nearest')
plt.colorbar()
plt.show()
# %%
# %%
agent_id=df.index
print(agent_id)
# %%
