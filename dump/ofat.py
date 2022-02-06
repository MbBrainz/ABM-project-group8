#%%
#import SAlib
#from SALib.sample import saltelli
#from SALib.analyze import sobol
import os, sys
from turtle import color
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from polarization.model import CityModel, Resident
from mesa.batchrunner import BatchRunner, BatchRunnerMP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import combinations
import csv
plt.style.use('seaborn')

#from run_parallel import simulate_parallel

#%%
#define the variables and bounds
# problem = {
#     'num_vars':10,
#     'names':['sidelength','total_steps','density', 
#     'm_barabasi','fermi_alpha','fermi_b', 'social factor',
#     'connections per step','opinion_max_dif', 'happiness threshold'],
#     'bounds':[[10,10],[100,100],[0.9,0.9],[2,2],[2,10],[0,10],[0,1],[1,5],[1,10],[0,1]]
# }

problem = {
    'num_vars':6,
    'names':['fermi_alpha','fermi_b', 'social_factor',
    'connections_per_step','opinion_max_diff', 'happiness_threshold'],
    'bounds':[[2,10],[0,10],[0,1],[1,5],[0,5],[0,1]],
}

#%%
#set repitions(compensate for stochasticity), number of steps and amount of distinct values per variable(N. 100 is good for us)
#total sample size = N * (num_vars+2)  
replicates = 5
max_steps = 25
distinct_samples = 100
#%%
#set output
#not sure how to connect this to our schedule 
model_reporters={"graph_modularity": lambda m:m.calculate_modularity(),
                "altieri_entropy_index": lambda m:m.calculate_a_entropyindex()}
#how it is stated in the Model class - "graph_modularity": self.calculate_modularity 

data={}
#%%
for i,var in enumerate(problem['names']):
    #get bounds for the variable and get <distinct_samples> samples within this space (uniform)
    param_values = np.linspace(*problem['bounds'][i],num=distinct_samples)
    if var == 'connections_per_step':
        param_values = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int)

    batch = BatchRunnerMP(CityModel,
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var:param_values},
                        model_reporters=model_reporters,
                        display_progress=True)
    batch.run_all()

    data[var]=batch.get_model_vars_dataframe()
#%%
print(data)
#%%
#plotting double output
def plot_param_var_conf(ax,df,var,param,i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.

    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]
    #%%

    replicates = df.groupby(var)[param].count()
    err_series = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)
    err = err_series.tolist()
    #print(err)
    
    ax.plot(x,y,c='k')
    ax.fill_between(x,y-err,y+err,alpha=0.2,color='k')

    if var == 'fermi_alpha': xlabel = "Fermi-Dirac alpha"
    elif var == 'fermi_b': xlabel = "Fermi-Dirac b"
    elif var == "social_factor": xlabel = "Social Network Influence"
    elif var == "connections_per_step": 
        xlabel = "Connections per step"
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    elif var == "opinion_max_diff": xlabel = "Max. difference in opinion"
    elif var == "happiness_threshold": xlabel = "Happiness threshold"

    ax.set_xlabel(xlabel)
    #ax.set_ylabel(param)

def plot_all_vars(df,param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """
    fig,axs=plt.subplots(6, figsize=(8,15))

    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], df[var], var, param, i)
    
    ylabel = "Graph Modularity" if param == "graph_modularity" else "Altieri Entropy Index"
    #fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
    fig.supylabel(ylabel)
    fig.tight_layout()

        

# for key,value in datadict.items():
#     print(key)
# for param in (['graph_modularity','altieri_entropy_index']):
#     print(f"Param: {param}")
#     plot_all_vars(data, param)
#     plt.show()

#%%
# Saving and loading functions

def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("./data/ofat_{}.csv".format(str(key)))

    with open("./data/keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))

def loader():
    """Reading data from keys"""
    with open("./data/keys.txt", "r") as f:
        keys = eval(f.read())

    dictex = {}    
    for key in keys:
        dictex[key] = pd.read_csv("./data/ofat_{}.csv".format(str(key)))

    return dictex

#%%
saver(data)

#%%
data_loaded = loader()
for param in (['graph_modularity','altieri_entropy_index']):
    print(f"Param: {param}")
    plot_all_vars(data_loaded, param)

plt.show()

# # %%

# %%
