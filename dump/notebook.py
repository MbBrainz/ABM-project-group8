#%%
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from polarization.benchmarking import benchmark
from polarization.model import CityModel
from polarization.model import ModelParams
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

#from run_parallel import simulate_parallel, read_dataframe

stepcount = 50
params = ModelParams(sidelength=20, density=0.8, m_barabasi=2, social_factor=0.95, connections_per_step=5, fermi_alpha=1, fermi_b=3, opinion_max_diff=2, total_steps = 50, happiness_threshold = 0.6)
model = CityModel(params)
print(benchmark(model, step_count=stepcount))

#%%
model.run_model(step_count = stepcount)
model_df = pd.DataFrame(model.datacollector.get_model_vars_dataframe())
agent_df = pd.DataFrame(model.datacollector.get_agent_vars_dataframe())

#%%
print(agent_df)
#%%
# #HISTOGRAMS: OPINION DISTRIBUTION
plt.hist(agent_df.loc[[2], ["opinion"]], density = True)
plt.title("Initial belief distribution")
plt.show()

plt.hist(agent_df.loc[[stepcount], ["opinion"]], density = True)
plt.title("End belief distribution")
plt.show()

# %%

#LINE PLOTS GRAPH ANALYSIS
#plt.plot(range(stepcount+1), model_df.cluster_coefficient, label = "Cluster Coefficient")
plt.plot(range(stepcount), model_df.graph_modularity, label = "Modularity")
plt.legend()
plt.title("Modularity")
plt.show()

#LINE PLOTS MOVERS PER STEP
plt.plot(range(stepcount), model_df.movers_per_step, label = "Movers per step")
plt.legend()
plt.title("Number of movers per step")
plt.show()

#LINE PLOT ENTROPY INDEX (MEASURE OF SEGREGATION)
#plt.plot(range(stepcount+1), model_df.leibovici_entropy_index, label = "Leibovici Entropy Index")
plt.plot(range(stepcount), model_df.altieri_entropy_index, label = "Altieri Entropy Index")

plt.legend() 
plt.title("Entropy Indices")
plt.show()

# %%
#3D SCATTER PLOT
#before
pos_col = pd.DataFrame(agent_df.loc[[2], ["position"]])
pos_col_list = pos_col["position"].tolist()
opinion_col = pd.DataFrame(agent_df.loc[[2], ["opinion"]])
opinion_col_list = opinion_col["opinion"].tolist()

ax = plt.axes(projection='3d')
agents_x_coords = [i[0] for i in pos_col_list]
agents_y_coords = [i[1] for i in pos_col_list]
agents_opinion = [i for i in opinion_col_list]

ax.scatter3D(agents_x_coords, agents_y_coords, agents_opinion, c=agents_opinion, cmap=cm.coolwarm)
plt.title("Distribution before")
plt.xlabel("grid x")
plt.ylabel("grid y")
plt.show()

#after
pos_col = pd.DataFrame(agent_df.loc[[stepcount], ["position"]])
pos_col_list = pos_col["position"].tolist()
opinion_col = pd.DataFrame(agent_df.loc[[stepcount], ["opinion"]])
opinion_col_list = opinion_col["opinion"].tolist()

ax = plt.axes(projection='3d')
agents_x_coords = [i[0] for i in pos_col_list]
agents_y_coords = [i[1] for i in pos_col_list]
agents_opinion = [i for i in opinion_col_list]

ax.scatter3D(agents_x_coords, agents_y_coords, agents_opinion, c=agents_opinion, cmap=cm.coolwarm)
plt.title("Distribution after")
plt.xlabel("grid x")
plt.ylabel("grid y")
plt.show()

# %%
from polarization.plot_grid import sim_grid_plot 
from polarization.plot_graph import create_graph
create_graph(agent_df, model_df)
sim_grid_plot(agent_df)
# %%
from polarization.util import plot_fermidirac
plot_fermidirac()

# %%

# simulate_parallel(params, distinct_samples=2)
# agent_df, model_df = read_dataframe(params, nsamples=2)

# all_data = pd.DataFrame([])
# reps = 3
# for i in range(reps):
#     model.run_model(step_count=stepcount)
#     model_df = pd.DataFrame(model.datacollector.get_model_vars_dataframe())
#     all_data = pd.concat([all_data, model_df])
    
# print(all_data)

# by_row_index = all_data.groupby(all_data.index)
# df_means = by_row_index.mean()
# df_stds = by_row_index.std()

# print(df_means)

# average_mod = df_means.graph_modularity
# mod_stds = df_stds.graph_modularity
# average_entropy = df_means.altieri_entropy_index
# ent_stds = df_stds.altieri_entropy_index

# plt.figure(dpi=300)
# plt.plot(range(stepcount), average_mod, label = "Mean progression of Modularity", linewidth = 0.9)
# plt.fill_between(range(stepcount), average_mod-mod_stds, average_mod+mod_stds, color='skyblue', alpha=0.5)
# plt.xlabel("Time steps")
# plt.ylabel("Mean Modularity")
# plt.title("Mean Modularity and Standard Deviations for Default Parameters")
# plt.legend()
# plt.grid(color = "grey", linestyle = "--")
# plt.show()

# plt.figure(dpi=300)
# plt.plot(range(stepcount), average_entropy, label = "Mean progression of Entropy", linewidth = 0.9)
# plt.fill_between(range(stepcount), average_entropy-ent_stds, average_entropy+ent_stds, color='skyblue', alpha=0.5)
# plt.xlabel("Time steps")
# plt.ylabel("Mean Entropy Index")
# plt.title("Mean Entropy Index and Standard Deviations for Default Parameters")
# plt.legend()
# plt.grid(color = "grey", linestyle = "--")
# plt.show()
