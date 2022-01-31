#this is just a copy of the tutorial on mesa for interactive grid visualisation
#%%
from mesa import Model
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from polarization.model import CityModel,ModelParams,Resident

#the simplest form of this:
def agent_portrayal(agent):
    if agent == None:
        return

    portrayal = {"Shape": "sqaure",
                 "Color":"red",
                 "Filled": "true",
                 "r": 0.5}
                 
    return portrayal
grid= CanvasGrid(agent_portrayal,10,10,500,500)
# model_params = ModelParams()._asdict

model_params=dict(sidelength=10, density=0.6, m_barabasi=2, fermi_alpha=5, fermi_b=3, social_factor=0.8, connections_per_step=5, opinion_max_diff=2, happiness_threshold=0.8)

#%%

server=ModularServer(CityModel,
                    [grid],
                    "City Model",
                    model_params)
server.port=8521
server.launch()




#%%
# we actually don't want an agent portrayal but a colour gradient of the 
# belief of the opinion. Not sure if possible to integrate colormaps
# so have just defined a color for each fraction of the range

#can opinion be called just like agent.opinion?

def agent_portrayal(agent):
    portrayal = {"Shape": "sqaure",
                 "Filled": "true",
                 "r": 0.5}

    if 0 <= agent.opinion < 0.1:
        portrayal["Color"] = "#1c1cff"
    elif 0.1 <= agent.opinion < 0.2:
        portrayal["Color"] = "#2e2eff"
    elif 0.2 <= agent.opinion < 0.3:
        portrayal["Color"] = "#4d4dff" 
    elif 0.3 <= agent.opinion < 0.4:
        portrayal["Color"] = "#7a7aff"
    elif 0.4 <= agent.opinion < 0.5:
        portrayal["Color"] = "#adadff"
    elif 0.5 <= agent.opinion < 0.6:
        portrayal["Color"] = "#e3e3e3"
    elif 0.6 <= agent.opinion < 0.7:
        portrayal["Color"] = "#ff8f8f"
    elif 0.7 <= agent.opinion < 0.8:
        portrayal["Color"] = "#ff5252"
    elif 0.8 <= agent.opinion < 0.9:
        portrayal["Color"] = "#ff3333"
    elif 0.9 <= agent.opinion < 1:
        portrayal["Color"] = "#f70000"
    return portrayal


#10x10 grid, drawn in 500x500 pixels
grid = CanvasGrid(agent_portrayal,10,10,500,500)

#he data needed is number agents moved - what is that called in data collector?
#make sure that data collector name is correct

chart = ChartModule([{"Label":"move_per_step",
                     "Color":"Black"}],
                     data_collector_name='datacollector')

server = ModularServer(CityModel,
                       [grid,chart],
                        "City Model",
                        {"N":100, "width":10, "height":10})
server.port = 8521
server.launch()

