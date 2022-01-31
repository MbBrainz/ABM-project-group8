#%%
from time import time
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from itertools import combinations
import numpy as np
from numpy import indices
import matplotlib.pyplot as plt
import pandas as pd
from polarization.model import CityModel, Resident
from mesa.batchrunner import BatchRunner, BatchRunnerMP
from IPython.display import clear_output

replicates = 1 # maybe do 5 for time
max_steps = 75
distinct_samples = 128



problem = {
    'num_vars':6,
    'names':['fermi_alpha','fermi_b', 'social_factor',
    'connections_per_step','opinion_max_diff', 'happiness_threshold'],
    'bounds':[[2,10],[0,10],[0,1],[1,5],[1,10],[0,1]],
}

model_reporters={"Network modularity": lambda m:m.calculate_modularity()}

#total number of param values = N(D+2) = 800 in our case***
# %%
#getting samples using saltelli.sample
param_values = saltelli.sample(problem, distinct_samples, calc_second_order=False)

#splitting up over all of us looks like this depending on our total number:
param_Nina = param_values[0:160]
param_Maurits = param_values[160:320]
param_Johanna = param_values[320:480]
param_Sasha = param_values[480:640]
param_Noah = param_values[640:]

param_test = param_values[:1]

param_values = param_test

batch = BatchRunnerMP(CityModel,
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in problem['names']},
                    model_reporters=model_reporters)

count = 0
data = pd.DataFrame(index=range(replicates*len(param_values)),

                                columns=['fermi_alpha','fermi_b', 'social factor','connections per step','opinion_max_dif', 'happiness threshold'])
#these are the outputs that we are measureing
data['Run'],data['Modularity']=None, None

# %%
start_time = time()
for i in range(replicates):
    for vals in param_values:
        #change params that should be integers - ??? still don't get this
        vals = list(vals)
        vals[3] = int(vals[3])

        #transform to dict with parameter names and their values
        variable_params = {}
        for name, val in zip(problem['names'], vals):
            variable_params[name] = val

        batch.run_iteration(variable_params, tuple(vals),count)
        iteration_data = batch.get_model_vars_dataframe().iloc[count]
        iteration_data['Run'] = count #this is quite nb i think
        data.iloc[count,0:6] = vals
        data.iloc[count,6:] = iteration_data
        count += 1
        clear_output()
        print(f'{count/len((param_values)*(replicates))*100:.2f}% done')

process_time = time() - start_time
print(f"Total simulation {process_time}")
#%%
# print(batch.datacollector_model_reporters)
print(data)
#save to file
#data.to_csv(data_file_name)?? or to pickle - ask maups :

#%%
#bringing together everyone's data:
# #use pickle instead
# nina0=pd.read_csv("filename-for repetition1")
# nina1=pd.read_csv("filename-for repetition2")
# nina2=pd.read_csv("filename-for repetition3")
# nina3=pd.read_csv("filename-for repetition4")
# nina4=pd.read_csv("filename-for repetition5")
# nina_allreps=pd.concat([nina0,nina1,nina2,nina3,nina4])
# print(nina_allreps.shape)
# #do this for everyone

# total=pd.concat([nina_allreps,noah_allreps etc]).fillna(0)
# print(total.shape)
# print(total.head())


#%%
#analyse
si_resident= sobol.analyze(problem, data['Modularity'].values, print_to_console=True, calc_second_order=False)

#%%
#plotting
def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params,2))
        indices = s['S' + i].reshape((p**2))
        indices = indices[~np.isnan(indices)]
        errors = s['S'+ i + '_conf'].reshape((p**2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S'+i]
        errors = s['S'+ i + '_conf']
        plt.figure()

    #set some aesthetics here
    l=len(indices)
    #edit this when we know what it looks like
    # plt.ylim([-0.2,l-1+0.2])
    # plt.xlim(-0.1, 1.1)

    plt.yticks(range(l), params)
    plt.xticks()
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o', capsize=2)
    plt.axvline(0,c='k')
    plt.tight_layout()
    # plt.savefig("filename.png")
# %%
plot_index(si_resident, problem["names"], i="1",title= "first order sensitivity")
# %%
