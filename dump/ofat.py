#%%
#import SAlib
#from SALib.sample import saltelli
#from SALib.analyze import sobol
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from polarization.model import CityModel, Resident
from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

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
    'bounds':[[2,10],[0,10],[0,1],[1,5],[1,10],[0,1]],
}

#%%
#set repitions(compensate for stochasticity), number of steps and amount of distinct values per variable(N. 100 is good for us)
#total sample size = N * (num_vars+2)  
replicates = 2
max_steps = 10
distinct_samples = 10
#%%
#set output
#not sure how to connect this to our schedule 
model_reporters={"Network modularity": lambda m:m.calculate_modularity}
#how it is stated in the Model class - "graph_modularity": self.calculate_modularity 

data={}

#%%
#doing what maups said - only for one varying parameter

# fixed = [10,10,0.6,2]
# for i,var in enumerate(test['names']):
#     print(var)
#     default = [1,1,1,1]
#     varied=np.linspace(*test['bounds'][i], num=distinct_samples)
#     for option in varied:
#         param_set=fixed+[option]+default
#         #i know that .insert() is faster but it doesn't work here (?)
#         print(param_set)

# #now to make it general:
#need to make it look through varible params and adjust the defaults 

#%%
for i,var in enumerate(problem['names']):
    #get bounds for the variable and get <distinct_samples> samples within this space (uniform)
    param_values = np.linspace(*problem['bounds'][i],num=distinct_samples)

    batch = BatchRunner(CityModel,
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var:param_values},
                        model_reporters=model_reporters,
                        display_progress=True)
    batch.run_all()

    data[var]=batch.get_agent_vars_dataframe()

#%%
#plotting 
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

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    ax.plot(x,y,c='k')
    ax.fill_between(x,y-err,y+err)

    ax.set_xlabel(var)
    ax.set_ylabel(param)

def plot_all_vars(df,param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """
    f,axs=plt.subplots(3, figsize=(7,10))

    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param, i)

for param in ('Network modularity'):
    plot_all_vars(data, param)
    plt.show()
