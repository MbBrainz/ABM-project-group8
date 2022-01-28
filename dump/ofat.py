#%%
#import SAlib
#from SALib.sample import saltelli
#from SALib.analyze import sobol
from polarization.model import CityModel, Resident
from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

#%%
#define the variables and bounds
problem = {
    'num_vars':6,
    'names':['social factor','happiness threshold', 'connections per step','fermi_alpha','fermi_b','opinion_max_dif'],
    'bounds':[[0,1],[0,1],[2,10],[0,10],[1,10],[1,5]]
}

#set repitions(compensate for stochasticity), number of steps and amount of distinct values per variable(N. 100 is good for us)
#total sample size = N * (num_vars+2)  
replicates = 5
max_steps = 200
distinct_samples = 100

#set output
#not sure how to connect this to our schedule 
model_reporters={"Grid entropy": lambda m:m.schedule.spatial_whatever_we_call_this(),
                "Network modularity": lambda m:m.schedule.modularity} 

data={}

for i,var in enumerate(problem['names']):
    #get bounds for the variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*problem['bounds'][i],num=distinct_samples)

#NB FROM NOTEBOOK - params must be integers 
# Keep in mind that wolf_gain_from_food should be integers. You will have to change
# your code to acommodate for this or sample in such a way that you only get integers.

#the way that they fix that:
    if var =='particular variable':
        samples = np.linspace(*problem['bounds'][i],num=distinct_samples, dtype=int)

    batch = BatchRunner(CityModel,
                        max_steps=max_steps,
                        iterations=replicates,
                        variables_parameters={var:samples},
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

for param in ('Resident'):
    plot_all_vars(data, param)
    plt.show()
