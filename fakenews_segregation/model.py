from audioop import avg
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
import random
import matplotlib.pyplot as plt
import networkx as nx

"""This file should contain the model class. 
If the file gets large, it may make sense to move the complex bits into other files, 
 this is the first place readers will look to figure out how the model works.
"""

random.seed(711)

# for reference, https://github.com/projectmesa/mesa/tree/main/examples/schelling is a good structure
# docs: https://mesa.readthe docs.io/en/master/best-practices.html 

# this is the proportion of external influence determined by socials and by neighbors
SOCIAL = 0.8
NEIGHBORS = 1 - SOCIAL

class Resident(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        # Variable attributes: 
        self.pos = pos
        self.belief = random.uniform(0,1) # we can use different distributions if we want to

        # Fixed attributes
        # set parameters for belief-changing
        self.vulnerability = random.uniform(0,1) # we can use different distributions if we want to
        self.weight_own = 1 - self.vulnerability
        self.weight_socials = SOCIAL * self.vulnerability 
        self.weight_neighbors = NEIGHBORS * self.vulnerability

        self.theta = random.uniform(0,1) # we can use different distributions if we want to

        # TO DO: SOCIAL NETWORK ATTRIBUTE

    def update_belief(self):
        pass

    def new_social(self):
        pass

    def remove_social(self):
        pass

    def move_pos(self):
        pass

    def step(self):
        print(f"Now at agent {self.unique_id}")
        

class CityModel(Model):
    def __init__(self, width=5, height=5, m_barabasi=2):
        print("init")
        # grid variables
        self.width = width
        self.height = height
        self.density = 0.9 # some spots need to be left vacant
        self.m_barabasi = m_barabasi
        
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(self.width, self.height, torus=True)

        self.n_agents = 0
        self.agents = []

        # setting up the Residents:
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]

            if random.uniform(0,1) < self.density:
                agent = Resident(self.n_agents, self, (x,y))
                self.grid.position_agent(agent, (x,y))
                self.schedule.add(agent)
                self.n_agents += 1

        # build a social network 
        self.Graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.m_barabasi)

    def step(self):
        """Run one step of the model."""
        # the scheduler uses the step() functions of the agents 
        self.schedule.step()

        # here, we need to collect data with a DataCollector

    def run_model(self, step_count=1):
        """Method that runs the model for a fixed number of steps"""
        # A better way to do this is with a boolean 'running' that is True when initiated, 
        # and becomes False when our end condition is met
        for i in range(step_count):
            self.step()

# Testing: 
model = CityModel()
model.run_model()