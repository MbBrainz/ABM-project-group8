"""This script plots the data generated from the experiments.py file"""
import matplotlib.pyplot as plt
import numpy as np
from polarization.model import CityModel
import pandas as pd
import networkx as nx
import seaborn as sns;
sns.set_theme()
sns.set_color_codes()
from polarization.plot_graph import plot_single_graph
from polarization.plot_grid import grid_plot

def plot_errorHue(mean_list, std_list, label, start=0,sample_data=None, sample_style='-r', ax=None):
    """Plotting the information from all repetitions of a run.

    Args:
        mean_list : mean result from all repetitions
        std_list : std of result from all repetitions
        label : name of output measure
        start : Defaults to 0.
        sample_data : The sample experiment run. Defaults to None.
        sample_style : colouring the sample data differently. Defaults to '-r'.
    """
    if ax == None: fig, ax = plt.subplots(1,1)
    x_array = range(start, len(mean_list)+start)
    ax.plot(
        x_array,
        mean_list,
        label=label
    )
    if not (sample_data.empty):
        ax.plot(x_array, sample_data, sample_style, label="sample")

    ax.fill_between(
        x_array,
        mean_list + std_list,
        mean_list - std_list,
        alpha = 0.5
    )
    ax.legend()

PARAMS_NAMES = [ "sidelength", "density","m_barabasi", "fermi_alpha", "fermi_b", "social_factor", "connections_per_step","opinion_max_diff", "happiness_threshold" ]


def run_experiment(iterations, stepcount, experiment):
    """ Running experiment and collecting data

    Args:
        iterations : number of repetitions of run
        stepcount : length of run
        experiment : particular set of parameters

    Returns:
        agents_dfs, model_dfs
    """
    model_dfs = []
    agent_dfs = []
    for i in range(iterations):
        model = CityModel(*(experiment["values"]))
        model.run_model(step_count=stepcount, desc=f'step {i}', collect_initial=True)

        model_dataframe = model.datacollector.get_model_vars_dataframe()
        model_dataframe['step'] = model_dataframe.index
        model_dfs.append(model_dataframe)
        agent_dfs.append( model.datacollector.get_agent_vars_dataframe())
    return agent_dfs, model_dfs


def plot_experiment(agent_dfs, model_dfs, stepcount, experiment):
    """ Plots a 2x3 grid of visual results from an experiment. 
    Visuals included are:
    Network graph, Modularity, Movers per step, Spatial grid, Entropy, Sample Opinion Distribution.
    Where applicable, the plots show the data from all repetitions with mean and std in blue and then 
    alos plots the sample run in red. 

    Args:
        agent_dfs : df containing data from agent reporters
        model_dfs : df containing data from model reporters
        stepcount : length of run
        experiment : particular set of parameters
    """
    for si in range(5):
        sample = agent_dfs[si], model_dfs[si]
        model_df = pd.concat(model_dfs)
        model_df = model_df.drop(columns=['edges', 'leibovici_entropy_index'])
        agent_df = pd.concat(agent_dfs)
        agg_model_df = model_df.groupby(by='step').aggregate(['std', 'mean'])


        nrows = len(agg_model_df.columns.unique(level=0))
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
        headers=agg_model_df.columns.unique(level=0)

        ax[0][0].set(title=f"Network Graph")
        plot_single_graph(sample[1], sample[0], ax=ax[0][0], layout=nx.nx_pydot.graphviz_layout)
        ax[1][0].set(title="Grid")
        grid_plot(sample[0], stepcount, experiment["values"][0], ax=ax[1][0])
        # Modularity
        plot_errorHue(agg_model_df[headers[0]]['mean'], agg_model_df[headers[0]]['std'], ax=ax[0][1], sample_data=sample[1][headers[0]], label=headers[0].title().replace("_"," "))
        # Entropy
        plot_errorHue(agg_model_df[headers[3]]['mean'], agg_model_df[headers[3]]['std'], ax=ax[1][1], sample_data=sample[1][headers[3]], label=headers[3].title().replace("_"," "))
        # Movers
        plot_errorHue(agg_model_df[headers[1]]['mean'][1:], agg_model_df[headers[1]]['std'][1:], ax=ax[0][2], sample_data=sample[1][headers[1]][1:], label=headers[1].title().replace("_"," "), start=1)
        # Opinion Distribution
        ax[1][2].hist(sample[0].loc[[stepcount], ["opinion"]], color='r', density = True)

        ax[0][1].set( xlabel="step", title="Modularity")
        ax[1][1].set( xlabel="step", title="Entropy")
        ax[0][2].set( xlabel="step", title="Movers")
        ax[1][2].set( xlabel="Opinion", title="Sample Opinion Distribution")

        plt.tight_layout()

        values =experiment["values"]
        indices=PARAMS_NAMES

        filename = experiment["name"]
        for i in range(len(values)):
            filename = filename + f"{indices[i]}={values[i]}"
        plt.savefig(f"figures/experiments/{filename}{si}.svg")
