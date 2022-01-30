
import multiprocessing
import os
from telnetlib import STATUS
import pandas as pd
from tqdm import tqdm
from polarization.model import CityModel, ModelParams

DATA_DIR = "./dump/data/"

def init_and_start_model(params, dir=DATA_DIR):
    """Initilalises model with *params , starts the modelsimulation and stores the data to a .pkl file in data directory

    Args:
        params tuple(ModelParams, sample_nr): Params to run the model for

    """
    param_set:  ModelParams = params[0]
    sample_nr:  int         = params[1]
    sample_id:  int         = params[2]
    nr_samples: int         = params[3]
    # params[0]: ModelParams, params[1]: int(sample_nr)
    if sample_nr == 1: # if it is the first sample for the parameter set
        print(f"\n Started first simulation of parameter set: \n {param_set} \n")

    model = CityModel(param_set)

    current = multiprocessing.current_process()
    pos = current._identity[0]-1

    model.run_model(param_set, desc=f"core {pos}, sample {sample_id}: {sample_nr}/{nr_samples}", pos=pos, collect_during=False)

    agent_df = model.datacollector.get_agent_vars_dataframe()
    agent_df.insert(0,"Sample", sample_nr)
    agent_df.to_pickle(f"{dir}_agentdf_{param_set.to_dir()}{sample_nr}.pkl")

    model_df = model.datacollector.get_model_vars_dataframe()
    model_df.insert(0, "Sample", sample_nr) # to be able do distinguish results from specific samples
    model_df.to_pickle(f"{dir}_modeldf_{param_set.to_dir()}{sample_nr}.pkl")


def simulate_parallel(params_list: list[ModelParams], distinct_samples=1):
    """Simulates in parallel for each item in the parameterlist and saves it to the data folder.
    Automatically chooses the amount of cores matching the ones available to your machine.

    This function used Init_and_start_model() and automatically schedules the jobs to the cores

    Args:
        params_list (list[ModelParams]): List if parameter variables to simulate for
    """
    data_and_samples = []

    # should first map all the distinct samples for a single parameter set
    for index, params in enumerate(params_list):
        params_and_data_list = [(params, x+1, index, distinct_samples) for x in range(distinct_samples)]
        data_and_samples.extend(params_and_data_list)


    tqdm.set_lock(multiprocessing.RLock())
    pool = multiprocessing.Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    pool.map(init_and_start_model, data_and_samples)

    pool.close()
    pool.join()

def read_dataframe_for_param_set(params, distinct_samples, dir=DATA_DIR):
    """Read the dataframe for all the samples of a specific parameter set.

    Args:
        params (ModelParams): parameters to get the data for
        distinct_samples (int): number of samples that you want to collect form data folder
        dir ([type], optional): Directory in which the data is saved. Defaults to DATA_DIR -> "./dump/data/" .

    Returns:
        tuple(Dataframe, Dataframe): [0] agent dataframe, [1] model dataframe
    """
    agent_df_list = []
    model_df_list = []

    for i in range(0,distinct_samples):
        sample_df = read_dataframe(params=params, sample_nr=i+1, dir=dir)
        agent_df_list.append(sample_df[0])
        model_df_list.append(sample_df[1])

    agent_df = pd.concat(agent_df_list)
    model_df = pd.concat(model_df_list)
    return agent_df, model_df # with agent dataframe and model dataframe


def read_dataframe(params, sample_nr=1, dir=DATA_DIR):
    """Reads dataframe from .pkl file that is created by the simulate_parallel function. It uses *params to see get the matching directory

    Args:
        params (ModelParams): Parameters that you want to read the data for

    Returns:
        tuple(Dataframe,Dataframe): agent dataframe[0] and model dataframe[1]
    """
    agent_dir = f"{dir}_agentdf_{params.to_dir()}{sample_nr}.pkl"
    agent_df  = pd.read_pickle(agent_dir)
    model_dir = f"{dir}_modeldf_{params.to_dir()}{sample_nr}.pkl"
    model_df  = pd.read_pickle(model_dir)
    return (agent_df, model_df)

def example():
    print("This is an exmple to show how paralisation should be used \n")
    params_list = [
        ModelParams(sidelength=10, density=0.6, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.8),
        ModelParams(sidelength=10, density=0.8, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.8),
        # ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.8),
        # ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.6, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.8),
        # ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=10, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.8),
        # ModelParams(sidelength=10, density=0.9, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.6),
    ]

    distinct_samples=2

    simulate_parallel(params_list, distinct_samples=distinct_samples)

    # here the files are read. This should be done separately from simulation, e.g. at visualisation.
    params = params_list[0]
    agent_df, model_df = read_dataframe_for_param_set(params, distinct_samples=distinct_samples)

    print(model_df)

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
