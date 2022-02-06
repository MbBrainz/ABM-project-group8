#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from util import testagent_df
plt.style.use("seaborn")

#%%
from matplotlib import cm
def grid_plot(agent_df, plot_step, sidelength, ax=None):
    # last_step = agent_df.index.max()[0]
    agentdf = pd.DataFrame(agent_df.loc[[plot_step], ["position", "opinion"]])\
        .sort_values("position")\
        .droplevel(0)

    op_grid = np.empty((sidelength, sidelength))
    op_grid[:] = np.NaN

    for index, row in agentdf.iterrows():
        op_grid[row["position"]] = row["opinion"]

    if ax == None:
        fig, ax = plt.subplots()


    current_cmap = cm.get_cmap("bwr").copy() #type: ignore
    current_cmap.set_bad(color='lightgray')


    ax.imshow(op_grid, cmap=current_cmap)
    ax.set_title(f'step number {plot_step}')
    ax.grid(False)

# %%
# grid_plot(testagent_df, 10, 7)

# %%
def sim_grid_plot(agent_df, grid_axis=[]):
    max_step = agent_df.index.max()[0]
    sidelength = agent_df["position"].max()[0] + 1
    print(sidelength)
    if len(grid_axis) == 0:
        fig, grid_axis = plt.subplots(1,2, figsize=(10,4))
        fig.suptitle("Grid plot")
    grid_plot(agent_df, 1, sidelength, grid_axis[0])
    grid_plot(agent_df, max_step, sidelength, grid_axis[1])

# sim_grid_plot(testagent_df)

# %%
