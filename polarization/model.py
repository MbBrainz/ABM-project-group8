
from ast import arg
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
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
        self.opinion = self.random.uniform(0,10) # we can use different distributions if we want to

        # Fixed attributes
        # set parameters for changing political opinion
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
        return [id for id in self.model.graph.nodes if (id not in self.socials_ids + [self.unique_id])]

    @property
    def unconnected(self):
        return  [unconnected for unconnected in self.model.schedule.agents if unconnected.unique_id not in self.socials_ids]


    def get_external_influences(self):
        """Calculate the external influence for an agent.
        Average opinion of friends and average opinion of neighbors is calculated.

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
            social_influence += social.opinion
        avg_social = social_influence / n_socials if n_socials != 0 else 0

        # loop through spatial neighbors and calculate influence
        for nbr in self.model.grid.get_neighbors(pos=self.pos,moore=True,include_center=False,radius=1):
            n_nbrs += 1
            nbr_influence += nbr.opinion
        avg_nbr = nbr_influence / n_nbrs

        return avg_social, avg_nbr

    def update_opinion(self):
        """Update political opinion with a weighted average of own opinion, friends' opinion, and neighbors' opinion.
            Vulnerability determines strength of external and internal influence.
            External influence is divided into 80% friends, 20% neighbors.
        """

        # update own political opinion based on external and internal influence
        social_infl, nbr_infl = self.get_external_influences()

        new_opinion = (self.weight_own * self.opinion) + \
            (self.weight_socials * social_infl) + \
            (self.weight_neighbors * nbr_infl)
        self.opinion = new_opinion

    def new_social(self):
        """Adds a new random connection from the agent with a probability determined by the Fermi-Dirac distribution.
            Choice of addition depends on similarity in political opinion.

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
            Choice of removal depends on similarity in political opinion.

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
        p_ij = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(self.opinion - potential_agent.opinion) - FERMI_B)))

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
        """
        Moves the location of an agent if they are unhappy. Once we have implemented the simulation time, we can make the probability
        of moving increase as the time since the last move increases.
        """

        # get the average opinion of the neighbours (nbr_infl)
        social_infl, nbr_infl = self.get_external_influences()

        # compare your opinion with the average of your neighbours using the fermi dirac equation.
        happiness = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(self.opinion - nbr_infl) - FERMI_B)))

        # if happiness is below some threshold, move to a random free position in the neighbourhood.
        if happiness < 0.5:
            self.model.grid.move_to_empty(self)


    def step(self):
        """A full step of the agent, consisting of:
        ...
        """
        # update network
        self.new_social()
        self.remove_social()

        #update grid
        self.move_pos()

        #update opinion
        self.update_opinion()


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

        self.datacollector = DataCollector(
            model_reporters={
                "graph_modularity": self.calculate_modularity,

            },
            agent_reporters={
                "opinion": lambda x: x.opinion
            }
        )

    def calculate_modularity(self):
        max_mod_communities = greedy_modularity_communities(self.graph)
        mod = modularity(self.graph, max_mod_communities)
        return mod

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
        self.datacollector.collect(self)

    def run_model(self, step_count=1):
        """Method that runs the model for a fixed number of steps"""
        # A better way to do this is with a boolean 'running' that is True when initiated,
        # and becomes False when our end condition is met
        for i in range(step_count):
            self.step()

import sys
from benchmarking import benchmark

def main(argv):
    if len(argv) != 1:
        print ("usage: model.py <steps>")
        return

    model = CityModel()
    proceed = benchmark(model, int(argv[0]))
    if proceed:
        model.run_model()

if __name__=="__main__":
    import sys
    main(sys.argv[1:])