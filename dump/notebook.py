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

# %%
from mesa.batchrunner import BatchRunnerMP
from polarization.model import CityModel
def main():
    fixed_params = {"width": 10, "height": 10,}
    var_params = {"density": [0.6, 0.8], "m_barabasi": [1,2,3]}

    batch = BatchRunnerMP(CityModel,
                            nr_processes=1,
                            variable_parameters=var_params,
                            fixed_parameters=fixed_params,
                            display_progress=True,
                            iterations=1)
    batch.run_all()
# %%
if __name__ == "__main__":
    main()