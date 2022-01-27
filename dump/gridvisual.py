#this is just a copy of the tutorial on mesa for interactive grid visualisation
# do we even want this?
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from mesa.visualization.modules import ChartModule

from polarization.model import CityModel

# we actually don't want an agent portrayal but a colour gradient of the 
# belief of the 
def agent_portrayal(agent):
    portrayal = {"Shape":"circle",
                 "Color" : "red",
                 "Filled":"true",
                 "Layer":0,
                 "r": 0.5}
    return portrayal


#10x10 grid, drawn in 500x500 pixels
grid = CanvasGrid(agent_portrayal,10,10,500,500)

#need to define the info we want to track (i think we want number of agents moving per time step,
# therefore the data needed to collect is number of agents moved)
#make sure that data collector name is correct
chart = ChartModule([{"Label":"Gini",
                     "Color":"Black"}],
                     data_collector_name='datacollector')



server = ModularServer(CityModel,
                       [grid,chart],
                        "City Model",
                        {"N":100, "width":10, "height":10})
server.port = 8521
server.launch()



#different scenario - just a snapshot of the grid, with # agents in each cell
import numpy as np
import matplotlib as plt

agent_opinion = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content,x,y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()