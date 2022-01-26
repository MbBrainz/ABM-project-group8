
from fileinput import filename
import multiprocessing
import time
import pandas as pd
from tqdm import tqdm

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import ModelParams
from polarization.model import CityModel

DATA_DIR = "./data/"

def load_pickles(dirs:list[str]) -> pd.DataFrame:
    df_list = []
    for dir in dirs:
        df_list.append(pd.read_pickle(dir))

    df = pd.concat(df_list, ignore_index=True)
    return df #type: ignore

def init_and_start_model(params: ModelParams, parallel=True):
    """Initilalises model with *params , starts the modelsimulation and stores the data to a .pkl file in data directory

    Args:
        params (ModelParams): Params to run the model for
        parallel (bool, optional): If parallel this var is used to determine the core on which the sim is run. Defaults to True.

    Returns:
        tuple[DataFrame, DataFrame]: agent_df and model_df. These are not used in parallel simulation as it saves the persists the data after every sim
    """

    model = CityModel(params)
    pos = 0
    if parallel:
        current = multiprocessing.current_process()
        pos = current._identity[0]-1

    model.run_model(10, desc=f"sim {pos}", pos=pos)
    agent_df = model.datacollector.get_agent_vars_dataframe()
    agent_df.to_pickle(f"{DATA_DIR}_agentdf_{params.to_dir()}.pkl")

    model_df = model.datacollector.get_model_vars_dataframe()
    model_df.to_pickle(f"{DATA_DIR}_modeldf_{params.to_dir()}.pkl")
    return (agent_df, model_df)

def simulate_parallel(params_list: list[ModelParams]):
    """Simulates in parallel for each item in the parameterlist and saves it to the data folder.
    Automatically chooses the amount of cores matching the ones available to your machine.

    This function used Init_and_start_model() and automatically schedules the jobs to the cores

    Args:
        params_list (list[ModelParams]): List if parameter variables to simulate for
    """

    tqdm.set_lock(multiprocessing.RLock())
    pool = multiprocessing.Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    results = pool.map(init_and_start_model, params_list)

    pool.close()
    pool.join()

def read_dataframe(params):
    """Reads dataframe from .pkl file that is created by the simulate_parallel function. It uses *params to see get the matching directory

    Args:
        params (ModelParams): Parameters that you want to read the data for

    Returns:
        tuple(Dataframe,Dataframe): agent dataframe[0] and model dataframe[1]
    """
    agent_dir = f"{DATA_DIR}_agentdf_{params.to_dir()}.pkl"
    agent_df  = pd.read_pickle(agent_dir)
    model_dir = f"{DATA_DIR}_modeldf_{params.to_dir()}.pkl"
    model_df  = pd.read_pickle(model_dir)
    return (agent_df, model_df)

def example():
    print("This is an exmple to show how paralisation should be used :)")
    params_list = [
        ModelParams(sidelength=10, density=0.5, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
        ModelParams(sidelength=10, density=0.6, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
        ModelParams(sidelength=10, density=0.8, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
        ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
        ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
        ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
        ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10),
    ]

    simulate_parallel(params_list)

    # here the files are read. This should be done separately from simulation, e.g. at visualisation.
    params = params_list[0]
    read_dataframe(params)

   # %%
if __name__ == "__main__":
    example()

# def compare_parallel():
#         start_time = time.time()
#     simulate_parallel(params_list, filename)
#     prll_time = time.time() - start_time

#     start_time = time.time()
#     for index, params in enumerate(params_list):
#         agent_df, model_df = init_and_start_model(params)
#         agent_df.to_pickle(f"./dump/data/{file_name}_agentdf_{params.to_dir()}_{index}.pkl")
#         model_df.to_pickle(f"./dump/data/{file_name}_modeldf_{params.to_dir()}_{index}.pkl")

#     seq_time = time.time() - start_time
