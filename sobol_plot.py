#%%

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from SALib.analyze import sobol

problem = {
    'num_vars':5,
    'names':['Fermi-Dirac alpha','Fermi-Dirac b', 'Social Network Influence',
    'Max. difference in opinion', 'Happiness Threshold'],
    'bounds':[[0,4],[0,6],[0,1],[0,4],[0,1]],
}
# problem = {
#     'num_vars':5,
#     'names':['fermi_alpha','fermi_b', 'social_factor',
#     'opinion_max_diff', 'happiness_threshold'],
#     'bounds':[[0,4],[0,6],[0,1],[0,4],[0,1]],
# }

DATA_DIR = "./data/sobol/"
datafiles = [dir for dir in os.listdir(DATA_DIR) if dir.startswith("sobol3")]

print(datafiles)
# %%
dataframes = []
for dir in datafiles:
    dataframes.append(pd.read_csv(f"{DATA_DIR}{dir}"))

sobol_df = pd.concat(dataframes)
#%%
sobol_df = sobol_df.drop(columns=["Unnamed: 0"])
#%%


# sobol_df.head()
#%%
sobol_df.describe()
# %%
# Creating a sortable column so the dataframe is ordered
sobol_df['indexing'] = sobol_df[['fermi_alpha', 'fermi_b', 'social_factor', 'opinion_max_diff', 'happiness_threshold']].sum(axis=1)
sobol_id_describe = sobol_df.groupby(by=["indexing"]).describe()
sobol_df = sobol_df.sort_values("indexing")
#%% Counting duplicates
sobol_id_describe["fermi_alpha"]["count"].value_counts()
# %%
columns = sobol_df.columns
columns = [x.title().replace("_"," ") for x in columns]
columns[0] = "Fermi-Dirac alpha"
columns[1] = "Fermi-Dirac b"
columns[2] = "Social Network Influence"
columns[3] = "Max. difference in opinion"
sobol_df.columns = columns
sobol_df.head()

si_resident_modularity = sobol.analyze(problem, sobol_df['Network Modularity'].values[:4438], print_to_console=True, calc_second_order=False)
si_resident_entropy = sobol.analyze(problem, sobol_df['Altieri Entropy Index'].values[:4438], print_to_console=True, calc_second_order=False)

#%%
def plot_index(s, params, i, ax=None, title=''):
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
        # plt.figure()

    #set some aesthetics here
    l=len(indices)
    #edit this when we know what it looks like
    # plt.ylim([-0.2,l-1+0.2])
    # plt.xlim(-0.1, 1.1)

    if ax == None:
        fig, ax = plt.subplots()
    ax.set_yticks(range(l), params)
    ax.set(title=title)
    # ax.set_xticks()
    ax.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o', capsize=2)
    ax.axvline(0,c='k')
    plt.tight_layout()
    # plt.savefig("filename.png")
# %%
plt.style.use('seaborn')
fig, ax = plt.subplots(2,2, figsize=(9,5), sharey=True, sharex=True)
# ax[0][0].set(supylabel="N|na")
plot_index(si_resident_modularity , problem["names"],   i="1",ax=ax[0][0],title= "1st Order Sensitivity\nModularity")
plot_index(si_resident_entropy , problem["names"],      i="1",ax=ax[1][0],title= "Entropy")
plot_index(si_resident_modularity , problem["names"],   i="T",ax=ax[0][1],title= "Total S\nModularity")
plot_index(si_resident_entropy , problem["names"],      i="T",ax=ax[1][1],title= "Entropy")
plt.savefig("figures/GSA.svg")
# %%
