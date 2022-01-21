
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
import random
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
N_POTENTIAL_CONNECTIONS = 5
FERMI_ALPHA = 1
FERMI_B = 3

class Resident(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        # Variable attributes:
        self.pos = pos
        self.view = self.random.uniform(0,10) # we can use different distributions if we want to

        # Fixed attributes
        # set parameters for changing political view
        self.vulnerability = self.random.uniform(0,1) # we can use different distributions if we want to
        self.weight_own = 1 - self.vulnerability
        self.weight_socials = SOCIAL * self.vulnerability
        self.weight_neighbors = NEIGHBORS * self.vulnerability

        self.theta = self.random.uniform(0,1) # we can use different distributions if we want to

    @property
    def socials_ids(self):
        return [social_id for social_id in self.model.graph[self.unique_id]]

    @property
    def socials(self):
        return [social for social in self.model.schedule.agents if social.unique_id in self.socials_ids]

    @property
    def unconnected_ids(self):
        return [id for id in self.model.graph.nodes if id not in self.socials_ids]

    @property
    def unconnected(self):
        return  [unconnected for unconnected in self.model.schedule.agents if unconnected.unique_id not in self.socials_ids]


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
        for social in self.socials:
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

        # update own political view based on external and internal influence
        social_infl, nbr_infl = self.get_external_influences(self.socials_ids)

        new_view = (self.weight_own * self.view) + (self.weight_socials * social_infl) + (self.weight_neighbors * nbr_infl)
        self.view = new_view


    def new_social(self):
        """Adds a new random connection from the agent with a probability determined by the Fermi-Dirac distribution.
            Choice of addition depends on similarity in political view.

            Args:
                socials_ids (list): IDs of social connections of agent
        """
        # select random un-connected agent, determine whether to form a new connection
        if len(self.unconnected_ids) < N_POTENTIAL_CONNECTIONS:
            n_potentials = len(self.unconnected_ids)
        else:
            n_potentials = N_POTENTIAL_CONNECTIONS

        # randomly select 'n_potentials' from people your not connected to
        pot_make_ids = np.random.choice(self.unconnected_ids, size=n_potentials, replace=False)

        # get agents from model.schedule with the id's from the pot_make_ids
        pot_makes = [social for social in self.model.schedule.agents if social.unique_id in pot_make_ids]

        for potential in pot_makes:
            self.consider_connection(potential_agent=potential, method="ADD")

    def remove_social(self):
        """Removes a few random connections from the agent with a probability determined by the Fermi-Dirac distribution.
            Choice of removal depends on similarity in political view.

            Args:
                socials_ids (list): IDs of social connections of agent
        """
        if len(self.socials_ids) < N_POTENTIAL_CONNECTIONS:
            n_potentials = len(self.socials_ids)
        else:
            n_potentials = N_POTENTIAL_CONNECTIONS

        # randomly select 'n_potentials' from the your network
        pot_break_ids = np.random.choice(self.socials_ids, size=n_potentials, replace=False)

        # get agents from model.schedule with the id's from the pot_break_ids
        pot_breaks = [social for social in self.model.schedule.agents if social.unique_id in pot_break_ids]

        for potential in pot_breaks:
            self.consider_connection(potential, method="REMOVE")


    def consider_connection(self, potential_agent, method):
        """Calculate the porobability of agent being connected to 'potential agent' and based on method add or remove the connection randomly

        Args:
            potential_agent (Resident): the resident to consider
            method (str): "ADD" or "REMOVE"
        """
        p_ij = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(self.view - potential_agent.view) - FERMI_B)))

        if method == "ADD":
            if p_ij > random.random():
                self.model.graph.add_edge(self.unique_id, potential_agent.unique_id)

        if method == "REMOVE":
            if p_ij < random.random():
                self.model.graph.remove_edge(self.unique_id, potential_agent.unique_id)

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
        # update network
        self.new_social()
        self.remove_social()

        #update grid

        #update view
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

        # setting up the Residents:
        self.initialize_population()

        # build a social network
        self.graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.m_barabasi)


    def initialize_population(self):
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]

            if self.random.uniform(0,1) < self.density:
                agent = Resident(self.n_agents, self, (x,y))
                self.grid.position_agent(agent, *(x,y))
                self.schedule.add(agent)
                self.n_agents += 1

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
# model = CityModel()
# model.run_model()