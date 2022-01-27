#%%
import os, sys
from dump.graph import create_graph
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import ModelParams


params_list = [
        ModelParams(sidelength=20, density=0.6, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=100, happiness_threshold=0.8),
        ModelParams(sidelength=20, density=0.8, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=100, happiness_threshold=0.8),
        ModelParams(sidelength=20, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=100, happiness_threshold=0.8),
        ModelParams(sidelength=20, density=0.9, m_barabasi=2, social_factor=0.6, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=100, happiness_threshold=0.8),
        ModelParams(sidelength=20, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=10, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=100, happiness_threshold=0.8),
        ModelParams(sidelength=20, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=100, happiness_threshold=0.6),
    ]

# %%
DATA_DIR = "./data/"

from dump.run_parallel import read_dataframe

plot_data = []

for params in params_list:
    plot_data.append(
        (read_dataframe(params, DATA_DIR))
    )


# %%


for index, data in enumerate(plot_data):
    print(params_list[index])
    create_graph(*data, params_list[index])
    sim_grid_plot(data[0], 1, params_list[index].total_steps)


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
    ax.set_xlim(-1,20)
    ax.set_ylim(-1,20)
    scat=ax.scatter(agents_x_coords,agents_y_coords,s=150,c=agents_opinion,marker='s', cmap='bwr')
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

# sim_grid_plot(agent_df,1,10)
# %%
