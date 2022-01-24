#%%
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from polarization.benchmarking import benchmark
from polarization.model import CityModel
import pandas as pd
import matplotlib.pyplot as plt

stepcount = 20
model = CityModel(width = 30, height = 30)
print(benchmark(model, step_count=stepcount))

#%%
model.run_model(step_count=stepcount)

agent_df = pd.DataFrame(model.datacollector.get_agent_vars_dataframe())
model_df = pd.DataFrame(model.datacollector.get_model_vars_dataframe())


# %%
print(agent_df.head())
print(agent_df.loc[[1], ["opinion"]])
print(agent_df.describe())

plt.hist(agent_df.loc[[1], ["opinion"]], density = True)
plt.title("Initial belief distribution")
plt.show()

plt.hist(agent_df.loc[[stepcount], ["opinion"]], density = True)
plt.title("End belief distribution")
plt.show()
# %%
print(model_df.head())
print(model_df)

# %%

plt.plot(range(stepcount+1), model_df.cluster_coefficient, label = "Cluster Coefficient")
plt.plot(range(stepcount+1), model_df.graph_modularity, label = "Modularity")
plt.legend()
plt.show()
# %%
plt.plot(range(stepcount+1), model_df.movers_per_step, label = "Movers per step")
plt.legend()
plt.show()
# %%
# width = 30
# height = 30



# import numpy as np
# import scipy.stats as st
# positions = np.vstack([width.ravel(), height.ravel()])
# values = np.vstack(agents_x_coords, agents_y_coords])
# kernel = st.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, width.shape)

# fig = plt.figure(figsize=(13, 7))
# ax = plt.axes(projection='3d')
# surf = ax.plot_surface(range(width), range(height), rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
# ax.set_xlabel('width')
# ax.set_ylabel('height')
# ax.set_zlabel('Political opinion')
# ax.set_title('Surface plot of political opinion over the grid at the end')
# fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
# ax.view_init(60, 35)
# %%
