
#%%
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
random.seed(102923)
# %%
#testing reading the file
df=pd.read_pickle("./test.pkl")
print(df["opinion"])
posdata=df["position"]
print(posdata)
agent_df=df
# %%
# df['position'].tolist()
# pd.DataFrame(df['position'].tolist(),index=df.index)
# df[['x','y']]=pd.DataFrame(df['position'].tolist(),index=df.index)
# %%
# df.reset_index(level='AgentID')
# print(df.tail())
# %%

#%%
# last_run = df.loc[df.index[-1]]
#can make this cleaner but fine for now
# x = df.loc[df.index[-49:], "x"]
# y=df.loc[df.index[-49:], "y"]
# opinion = df.loc[df.index[-49:], "opinion"]
# print(opinion)
# #%%
# grid_opinion = np.zeros((grid_width,grid_height))
# #for agent in AgentID:
# for agent in range(num_agents):
#     agent_opinion = opinion
#     grid_opinion[x][y] = agent_opinion
# print(grid_opinion)
# #%%
# plt.set_cmap('bwr')
# plt.imshow(grid_opinion, interpolation='nearest')
# plt.colorbar()
# plt.show()
# %%
# %%


# %%
#now to make this data into a grid
# plt.scatter(agents_x_coords,agents_y_coords,s=600,c=agents_opinion,marker='s', cmap='bwr')
# cbar=plt.colorbar()
# 6# %%
# import itertools
# itertools.product('0123456789', repeat=2)

# %%
#this doesn't work
# pos=(agents_x_coords,agents_y_coords)
# combined=list(itertools.zip_longest(pos, agents_opinion))
# print(combined)
# %%


#data
# data = agents_opinion
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(agents_x_coords, agents_y_coords, c=data, cmap='bwr')

# %%

def grid_plot(agent_df, stepcount, ax=None):
    pos_col = pd.DataFrame(agent_df.loc[[stepcount], ["position"]])
    pos_col_list = pos_col["position"].tolist()
    opinion_col = pd.DataFrame(agent_df.loc[[stepcount], ["opinion"]])
    opinion_col_list = opinion_col["opinion"].tolist()

    agents_x_coords = [i[0] for i in pos_col_list]
    agents_y_coords = [i[1] for i in pos_col_list]
    agents_opinion = [i for i in opinion_col_list]
    #print("final agent opinion",agents_opinion)
    if ax == None:
        fig, ax = plt.subplots()

    ax.set_title(f'stepcount:{stepcount}')
    ax.set_xlim(-1,10)
    ax.set_ylim(-1,10)
    scat=ax.scatter(agents_x_coords,agents_y_coords,s=400,c=agents_opinion,marker='s', cmap='bwr')
    # cbar=fig.colorbar(scat,ticks=[1,9])
    # cbar.ax.set_yticklabels(['left','right'])

# %%
#
# %%
def sim_grid_plot(agent_df,first, last):
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle("Title")
    grid_plot(agent_df,first, axs[0])
    grid_plot(agent_df,last, axs[1])

sim_grid_plot(agent_df,1,10)
# %%
