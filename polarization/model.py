
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm import tqdm, tnrange, trange
from spatialentropy import leibovici_entropy
from spatialentropy import altieri_entropy



import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cluster import average_clustering
import numpy as np

from util import ModelParams, default_params

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
FERMI_ALPHA = 5
FERMI_B = 3

class Resident(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        # Variable attributes:
        self.pos = pos
        self.opinion = self.random.uniform(0,10) # we can use different distributions if we want to

        # Fixed attributes
        # set parameters for changing political opinion
        self.vulnerability = self.random.uniform(0,0.5) # we can use different distributions if we want to
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
            if abs(social.opinion-self.opinion) < self.model.params.opinion_max_diff:
                social_influence += social.opinion
                n_socials += 1
        avg_social = social_influence / n_socials if n_socials != 0 else 0

        # loop through spatial neighbors and calculate influence
        for nbr in self.model.grid.get_neighbors(pos=self.pos,moore=True,include_center=False,radius=1):
            if abs(nbr.opinion-self.opinion) < self.model.params.opinion_max_diff:
                n_nbrs += 1
                nbr_influence += nbr.opinion
        avg_nbr = nbr_influence / n_nbrs if n_nbrs != 0 else 0

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


    def move_pos(self):
        """
            Moves the location of an agent if they are unhappy.
        """
        # get the average opinion of the neighbours (nbr_infl)
        social_infl, nbr_infl = self.get_external_influences()

        # compare your opinion with the average of your neighbours using the fermi dirac equation.
        happiness = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(self.opinion - nbr_infl) - FERMI_B)))

        # if happiness is below some threshold, move to a random free position in the neighbourhood.
        if happiness < self.model.params.happiness_threshold:
            self.model.grid.move_to_empty(self)
            self.model.movers_per_step += 1


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
    def __init__(self, params: ModelParams = default_params):
        # grid variables
        self.params = params
        self.schedule = RandomActivation(self)
        self.movers_per_step = 0
        self.n_agents = 0

        # setting up the Residents:
        self.grid = SingleGrid(self.params.sidelength, self.params.sidelength, torus=True)
        self.initialize_population()

        # build a social network
        self.graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.params.m_barabasi)

        self.datacollector = DataCollector(
            model_reporters={
                "graph_modularity": self.calculate_modularity,
                "movers_per_step": lambda m: m.movers_per_step,
                "cluster_coefficient": self.calculate_clustercoef,
                "edges": self.get_graph_dict,
                "leibovici_entropy_index": self.calculate_l_entropyindex,
                "altieri_entropy_index": self.calculate_a_entropyindex,

            },
            agent_reporters={
                "opinion": lambda x: x.opinion,
                "position": lambda p: p.pos,
            }
        )
        self.running = True

    def calculate_modularity(self):
        max_mod_communities = greedy_modularity_communities(self.graph)
        mod = modularity(self.graph, max_mod_communities)
        return mod

    def calculate_clustercoef(self):
        cluster_coefficient = average_clustering(self.graph)
        return cluster_coefficient

    def get_graph_dict(self):
        graph_dict = nx.convert.to_dict_of_dicts(self.graph)
        return graph_dict

    def calculate_l_entropyindex(self):
        agent_infolist = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
        points = []
        types = []

        for i in range(len(agent_infolist)):
            points.append([agent_infolist[i][0][0], agent_infolist[i][0][1]])
        
        for i in agent_infolist:
                if i[1]<3:
                    types.append("left")
                
                elif 3<i[1]<7:
                    types.append("middle")
                else:
                    types.append("right")
        
        points = np.array(points)
        types = np.array(types)

        e = leibovici_entropy(points, types, d=2)
        e_entropyind = e.entropy 

        return e_entropyind
    
    def calculate_a_entropyindex(self):
        agent_infolist = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
        points = []
        types = []

        for i in range(len(agent_infolist)):
            points.append([agent_infolist[i][0][0], agent_infolist[i][0][1]])
        
        
        for i in agent_infolist:
            if i[1]<3:
                types.append("left")
            elif 3<i[1]<7:
                types.append("middle")
            else:
                types.append("right")

        points = np.array(points)
        types = np.array(types)
        
        a = altieri_entropy(points, types, cut=2)
        a_entropyind = a.entropy
            
        return a_entropyind

    def initialize_population(self):
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]

            if self.random.uniform(0,1) < self.params.density:
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

        #set the counter of movers per step back to zero
        self.movers_per_step = 0

    def run_model(self, step_count=1, desc="", pos=0):
        """Method that runs the model for a fixed number of steps"""
        # A better way to do this is with a boolean 'running' that is True when initiated,
        # and becomes False when our end condition is met
        for i in trange(self.params.total_steps, desc=desc, position=pos):
            self.step()


import sys
def main(argv):
    steps=1
    if len(argv) != 1:
        print ("usage: model.py <steps>")
    else:
        steps=int(argv[0])

    model = CityModel()
    # # proceed = benchmark(model, int(argv[0]))
    # if proceed:
    model.run_model()

if __name__=="__main__":
    import sys
    main(sys.argv[1:])