#%%
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from polarization.model import CityModel
import pandas as pd

model = CityModel()

model.run_model(step_count=100)

agent_df = pd.DataFrame(model.datacollector.get_agent_vars_dataframe())
model_df = pd.DataFrame(model.datacollector.get_model_vars_dataframe())


# %%
agent_df.describe()
# %%
model_df.head()
print(model_df)
