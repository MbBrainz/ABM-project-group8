from audioop import avg
from operator import contains
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
        self.view = self.random.uniform(0,1) # we can use different distributions if we want to

        # Fixed attributes
        # set parameters for changing political view
        self.vulnerability = self.random.uniform(0,1) # we can use different distributions if we want to
        self.weight_own = 1 - self.vulnerability
        self.weight_socials = SOCIAL * self.vulnerability
        self.weight_neighbors = NEIGHBORS * self.vulnerability

        self.theta = self.random.uniform(0,1) # we can use different distributions if we want to


    def get_external_influences(self, socials_ids):
        """Calculate the external influence for an agent. 
        Average view of friends and average view of neighbors is calculated.

        Args:
            socials_ids (list of Agent() objects): list of friends

        Returns:
            float, float: average social influence and average neighbor influence respectively
        """
        nbr_influence = 0
        n_nbrs = 0
        social_influence = 0
        n_socials = 0

        # loop through social network and calculate influence
        socials = [social for social in self.model.schedule.agents if social.unique_id in socials_ids]
        for social in socials:
            n_socials += 1
            social_influence += social.view
        avg_social = social_influence / n_socials if n_socials != 0 else 0

        # loop through spatial neighbors and calculate influence 
        for nbr in self.model.grid.get_neighbors(pos=self.pos,moore=True,include_center=False,radius=1):
            n_nbrs += 1
            nbr_influence += nbr.view
        avg_nbr = nbr_influence / n_nbrs

        return avg_social, avg_nbr

    
    def update_view(self):
        """Update political view with a weighted average of own view, friends' view, and neighbors' view.
        Vulnerability determines strength of external and internal influence. 
        External influence is divided into 80% friends, 20% neighbors. 
        """
        # need to determine friends again after updating the network before updating the view
        socials_ids = [social_id for social_id in self.model.Graph[self.unique_id]]

        # update own political view based on external and internal influence
        social_infl, nbr_infl = self.get_external_influences(socials_ids)
        new_view = (self.weight_own * self.view) + (self.weight_socials * social_infl) + (self.weight_neighbors * nbr_infl)
        self.view = new_view


    def new_social(self, socials_ids):
        """Adds a new random connection from the agent with a probability determined by the Fermi-Dirac distribution. 
        Choice of addition depends on similarity in political view. 

        Args:
            socials_ids (list): IDs of social connections of agent
        """
        # select random un-connected agent, determine whether to form a new connection 
        unconnected = [agent for agent in self.model.schedule.agents if agent.unique_id not in socials_ids and \
             agent.unique_id != self.unique_id]
        choice = random.choice(unconnected)
        other_view = choice.view

        # calculate the probability to form a connection from a normal distribution
        if random.uniform(0,1) < self.normal_dist(other_view, self.view, 0.25*self.theta):
            # make a new connection 
            self.model.Graph.add_edge(self.unique_id, choice.unique_id)


    def remove_social(self, socials_ids):
        """Removes a random connection from the agent with a probability determined by the Fermi-Dirac distribution. 
        Choice of removal depends on similarity in political view. 

        Args:
            socials_ids (list): IDs of social connections of agent
        """
        socials = [social for social in self.model.schedule.agents if social.unique_id in socials_ids]
        choice = random.choice(socials)
        other_view = choice.view

        if random.uniform(0,1) < 1 - self.normal_dist(other_view, self.view, 0.25*self.theta):
            # remove connection 
            self.model.Graph.remove_edge(self.unique_id, choice.unique_id)

    def normal_dist(self, x, mu, std):
        # NOT THIS ONE, FERMI-DIRAC !!!! 
        """Calculates probability from a normal distribution with mean mu and standard deviation sigma.

        Args:
            x (float): value to calculate probability for
            mu (float): distribution mean
            std (float): distribution standard deviation

        Returns:
            float: probability for x under normal distribution
        """
        return (1 / (std * np.sqrt(2* np.pi)))* np.exp(-0.5*((x-mu)/std)**2)


    def move_pos(self):
        pass


    def step(self):
        """A full step of the agent, consisting of: 
        ...
        """
        socials_ids = [social_id for social_id in self.model.Graph[self.unique_id]]
        self.new_social(socials_ids)
        self.remove_social(socials_ids)
        self.update_view()




class CityModel(Model):
    def __init__(self, width=5, height=5, m_barabasi=2, seed=711):
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

            if self.random.uniform(0,1) < self.density:
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