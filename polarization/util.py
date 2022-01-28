from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fermi_dirac_graph(d, FERMI_ALPHA, FERMI_B):
    """
    A graph to visualise the effect of alpha and b on the probability to connect for a given distance.
    """
    pij = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(d) - FERMI_B)))
    return pij

def plot_fermidirac():
    params = [(10,1),(1,3)]

    distances = np.linspace(0, 10, 100)
    plot_data =[]
    for param in params:
        data = {
            "y": fermi_dirac_graph(distances, param[0], param[1]),
            "alpha": param[0],
            "b": param[1]
            }
        plot_data.append(data)

    for data in plot_data:
        plt.plot(distances, data['y'], label = f"alpha = {data['alpha']}, b = {data['b']}")
    plt.xlabel("Absolute distance in the political belief-space")
    plt.ylabel("Probability to connect")
    plt.title("Fermi-Dirac probability to connect to another node based on political distance")
    plt.legend()
    plt.show()


BaseModelParams = namedtuple("ModelParams", [
    "sidelength",
    "density",
    "m_barabasi",
    "social_factor",
    "connections_per_step",
    "fermi_alpha",
    "fermi_b",
    "opinion_max_diff",
    "total_steps",
    "happiness_threshold",
])

class ModelParams(BaseModelParams):
    def to_dir(self):
        filedir=""
        for item in self:
            filedir += str(item).replace(".","_") + "-"
        return filedir

default_params = ModelParams(sidelength=10, density=0.5, m_barabasi=2, social_factor=0.8, connections_per_step=5, fermi_alpha=5, fermi_b=3, opinion_max_diff=2, total_steps=10, happiness_threshold=0.8)

def read_dataframe(params, dir="./data/"):
    """Reads dataframe from .pkl file that is created by the simulate_parallel function. It uses *params to see get the matching directory

    Args:
        params (ModelParams): Parameters that you want to read the data for

    Returns:
        tuple(Dataframe,Dataframe): agent dataframe[0] and model dataframe[1]
    """
    agent_dir = f"{dir}_agentdf_{params.to_dir()}.pkl"
    agent_df  = pd.read_pickle(agent_dir)
    model_dir = f"{dir}_modeldf_{params.to_dir()}.pkl"
    model_df  = pd.read_pickle(model_dir)
    return (agent_df, model_df)

testagent_df, testmodel_df = (
    pd.read_pickle("./mock_data/test_agentdf.pkl"),
    pd.read_pickle("./mock_data/test_modeldf.pkl")
)