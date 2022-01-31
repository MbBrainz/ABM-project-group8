#%%
#import SAlib
#from SALib.sample import saltelli
#from SALib.analyze import sobol
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from polarization.model import CityModel, Resident
from mesa.batchrunner import BatchRunner, BatchRunnerMP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
replicates = 2
max_steps = 25
distinct_samples = 15
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
    f,axs=plt.subplots(6, figsize=(7,10))


    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param, i)

for param in (['graph_modularity','altieri_entropy_index']):
    print(f"Param: {param}")
    plot_all_vars(data, param)
    plt.show()


# %%
# open file for writing, "w" is writing
w = csv.writer(open("data_15samples_2repl_DICT.csv", "w"))

# loop over dictionary keys and values
for key, val in data.items():

    # write every key and value to file
    w.writerow([key, val])


# %%
