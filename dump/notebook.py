#%%
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from polarization.benchmarking import benchmark
from polarization.model import CityModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


stepcount = 50
params = ModelParams(sidelength=20, density=0.8, m_barabasi=2, social_factor=0.95, connections_per_step=5, fermi_alpha=1, fermi_b=3, opinion_max_diff=2, total_steps = 50, happiness_threshold = 0.6)
model = CityModel(params)
print(benchmark(model, step_count=stepcount))

#%%
model.run_model(step_count=stepcount)

agent_df = pd.DataFrame(model.datacollector.get_agent_vars_dataframe())
model_df = pd.DataFrame(model.datacollector.get_model_vars_dataframe())

# %%
print(agent_df.head())
print(agent_df.describe())
print(model_df.head())

#%%

#HISTOGRAMS: OPINION DISTRIBUTION
plt.hist(agent_df.loc[[1], ["opinion"]], density = True)
plt.title("Initial belief distribution")
plt.show()

plt.hist(agent_df.loc[[stepcount], ["opinion"]], density = True)
plt.title("End belief distribution")
plt.show()
# %%
print(model_df.head())

# %%

#LINE PLOTS GRAPH ANALYSIS
#plt.plot(range(stepcount+1), model_df.cluster_coefficient, label = "Cluster Coefficient")
plt.plot(range(stepcount+1), model_df.graph_modularity, label = "Modularity")
plt.legend()
plt.title("Modularity")
plt.show()

#LINE PLOTS MOVERS PER STEP
plt.plot(range(stepcount+1), model_df.movers_per_step, label = "Movers per step")
plt.legend()
plt.title("Number of movers per step")
plt.show()

#LINE PLOT ENTROPY INDEX (MEASURE OF SEGREGATION)
#plt.plot(range(stepcount+1), model_df.leibovici_entropy_index, label = "Leibovici Entropy Index")
plt.plot(range(stepcount+1), model_df.altieri_entropy_index, label = "Altieri Entropy Index")

plt.legend() 
plt.title("Entropy Indices")
plt.show()
# %%

# %%
#3D SCATTER PLOT
#before
pos_col = pd.DataFrame(agent_df.loc[[1], ["position"]])
pos_col_list = pos_col["position"].tolist()
opinion_col = pd.DataFrame(agent_df.loc[[1], ["opinion"]])
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
