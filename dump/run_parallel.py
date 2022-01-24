
import multiprocessing
from time import sleep, time_ns
import pandas as pd
from tqdm import tqdm

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import ModelParams
from polarization.model import CityModel

def load_pickles(dirs:list[str]) -> pd.DataFrame:
    df_list = []
    for dir in dirs:
        df_list.append(pd.read_pickle(dir))

    df = pd.concat(df_list, ignore_index=True)
    return df #type: ignore

def init_and_start_model(params:ModelParams, ):
    model = CityModel(params)
    current = multiprocessing.current_process()
    pos = current._identity[0]-1
    model.run_model(10, desc=f"sim {pos}", pos=pos)
    agent_df = model.datacollector.get_agent_vars_dataframe()
    model_df = model.datacollector.get_model_vars_dataframe()
    return (agent_df, model_df)

def main():
    params_list = [
        ModelParams(sidelength=10, density=0.5, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2),
        ModelParams(sidelength=10, density=0.6, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2),
        ModelParams(sidelength=10, density=0.8, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2),
        ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2),
    ]

    step_count = 5
    file_name = "testparallel_df"

    tqdm.set_lock(multiprocessing.RLock())
    pool = multiprocessing.Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    number = 0
    for result in pool.imap(init_and_start_model, params_list):
        result[0].to_pickle(f"./dump/data/{file_name}_agentdf_{number}.pkl")
        result[1].to_pickle(f"./dump/data/{file_name}_modeldf_{number}.pkl")
        number +=1

    pool.close()
    pool.join()


    agent_data_dirs = [f"./dump/data/{dir}" for dir in os.listdir("./dump/data/") if dir.startswith(f"{file_name}_agentdf_")]
    model_data_dirs = [f"./dump/data/{dir}" for dir in os.listdir("./dump/data/") if dir.startswith(f"{file_name}_modeldf_")]


    agent_df = load_pickles(agent_data_dirs)
    model_df = load_pickles(model_data_dirs)

    print(agent_df.describe())
    print(model_df.head())


   # %%
if __name__ == "__main__":
    main()
