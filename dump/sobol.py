#import SAlib
#from SALib.sample import saltelli
#from SALib.analyze import sobol
from itertools import combinations
from numpy import indices
import matplotlib.pyplot as plt 
import pandas as pd
from ofat import problem, model_reporters
from polarization.model import CityModel, Resident

replicates = 10
max_steps = 100
distinct_samples = 100

#getting samples using saltelli.sample
# figure out a different way of doing 
# this since we wont be using the problem dictionary - or we need just change the prob dict 
# but keeping it a dict
param_values = saltelli.sample(problem, distinct_samples,bool=False)

#just from the notebook, but we won't be using
batch = BatchRunner(CityModel,
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in problem['names']},
                    model_reporters=model_reporters)

count = 0 
data = pd.DataFrame(index=range(replicates*len(param_values)),
                                columns=['param1', 'param2', 'param3'])
#the notebook has Run, Sheep, Wolves
data['Run'],data['Resident?']=None, None 

for i in range(replicates):
    for vals in param_values:
        #change params that should be integers - ??? still don't get this
        vals = list(vals)
        vals[2] = int(vals[2])
        #transform to dict with parameter names and their values
        variable_params = {}
        for name, val in zip(problem['names'], vals):
            variable_params[name] = val
        
        batch.run_interation(variable_params, tuple(vals),count)
        iteration_data = batch.get_model_vars_dataframe().iloc[count]
        iteration_data['Run'] = count #this is quite nb i think
        data.iloc[count,0:3] = vals
        data.iloc[count,3:6] = iteration_data
        count += 1

        clear_output()
        print(f'{count/len((param_values)*(replicates))*100:.2f}% done')

#%%
print(data)

#%%
#analyse
si_resident= sobol.analyze(problem, data['Resident'].values, print_to_console=True)

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
