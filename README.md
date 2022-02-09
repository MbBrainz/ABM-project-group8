# Agent Based Modeling Project - Group 8

### Students with student number:
- Maurits Bos: 14014777
- Sasha Boykova: 12106070
- Noah van de Bunt: 11226218
- Johanna Gehlen: 12108006
- Nina Holzbach: 13827464

## Summary
An Agent-based Model developed in fufillment of course #5284AGBM6Y at the University of Amsterdam. 
The aim of this project was to investigate the dynamics of opinion polarisation, where an individual is influenced by both their social network and spatial neighbours. 

## Installation
To install the dependencies use pip and the requirements.txt in this directory. e.g.
```
$ pip install -r requirements.txt
```

## How to run 
To run the model interactively, run `mesa runserver` in this directory. e.g.
```
$ mesa runserver
```
This will open a browser with an interactive visualisation of the spatial grid and agents' opinions ranging from Blue (0) to Red (10).

## Folders & Files
* `data` folder contains all data from sensitivity analysis.
* `figures` folder contains all figures from tests, sensitivity analysis and experiments.
* `polarisation` folder includes the following files:
  - `experiment_run.py` : Contains functions for plotting data from experiment runs
  - `experiments.py`: Generates results from a set of experiments with the model
  - `model.py`: Contains the model and agent classes
  - `plot_graph.py`: Contains functions to plot graphs of the social network of the system
  - `plot_grid.py`: Contains functions to plot the agents on the spatial grid
  - `run.py`: Launches a model visualisation server.
  - `server.py`: Defines classes for visualising the model in the browser via Mesa and instantiates a visualisation server.
  - `util.py`: Contains various utilities used throughout the project
* `sensitivityanalysis` folder contains the following files:
  - `batch_run.py`: Runs Sobol (Global) Sensitivity Analysis with BatchRunnerMP
  - `ofat.py`: Generates and runs OFAT (Local) Sensitivity Analysis
  - `sobol-datacollector.py`:The initial files used to run Sobol, replaced with the faster `batch_run.py`
  - `sobol_plot.py`: Generates plots from the data generated from Sobol Sensitivity Analysis
* `tests` : contains some test agent and model files 

<br>
...