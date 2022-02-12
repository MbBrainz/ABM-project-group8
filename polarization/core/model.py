#%%
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm import tqdm, trange
import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cluster import average_clustering
import numpy as np
from spatialentropy import leibovici_entropy
from spatialentropy import altieri_entropy
import sys

"""This file contains the agent and model classes.
"""

random.seed(711)

class Resident(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        # Variable attributes:
        self.pos = pos
        self.opinion = self.random.uniform(0,10)

        # Fixed attributes
        self.vulnerability = self.random.uniform(0,0.5)
        self.weight_own = 1 - self.vulnerability
        self.weight_socials = self.model.social_factor * self.vulnerability
        self.weight_neighbors = (1 - self.model.social_factor) * self.vulnerability

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

    @property
    def neighbours(self):
        return self.model.grid.get_neighbors(self.pos, moore=True,include_center=False,radius=1)


    def get_external_influences(self):
        """Calculate the external influence for an agent.
        Average opinion of network and average opinion of neighbors is calculated.

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
            if abs(social.opinion-self.opinion) < self.model.opinion_max_diff:
                social_influence += social.opinion
                n_socials += 1
        avg_social = social_influence / n_socials if n_socials != 0 else 0

        # loop through spatial neighbors and calculate influence
        for nbr in self.model.grid.get_neighbors(pos=self.pos,moore=True,include_center=False,radius=1):
            if abs(nbr.opinion-self.opinion) < self.model.opinion_max_diff:
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

        new_opinion = self.opinion

        #if the agent has both network connections and neighbours, use original weighted average
        if social_infl != 0 and nbr_infl != 0:
            new_opinion = \
                (self.weight_own * self.opinion) + \
                (self.weight_socials * social_infl) + \
                (self.weight_neighbors * nbr_infl)

        #if the agent does not have any network connections, adjust weighted average
        elif social_infl == 0 and nbr_infl != 0:
            new_opinion = \
                (self.weight_own * self.opinion) + \
                ((1-self.weight_own) * nbr_infl)

        #similarly, if the agent does not have any neighbours, adjust weighted average
        elif nbr_infl == 0 and social_infl != 0:
            new_opinion = \
                (self.weight_own * self.opinion) + \
                ((1-self.weight_own) * social_infl)

        self.opinion = new_opinion

    def new_social(self):
        """Adds a new random connection from the agent with a probability determined by the Fermi-Dirac distribution.
            Choice of addition depends on similarity in political opinion.

            Args:
                socials_ids (list): IDs of social connections of agent
        """
        # select random un-connected agent, determine whether to form a new connection
        if len(self.unconnected_ids) < self.model.connections_per_step:
            n_potentials = len(self.unconnected_ids)
        else:
            n_potentials = self.model.connections_per_step

        # randomly select 'n_potentials' from people the agent is not connected to
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
        if len(self.socials_ids) < self.model.connections_per_step:
            n_potentials = len(self.socials_ids)
        else:
            n_potentials = self.model.connections_per_step

        # randomly select 'n_potentials' from the agent's network
        pot_break_ids = np.random.choice(self.socials_ids, size=n_potentials, replace=False)

        # get agents from model.schedule with the id's from the pot_break_ids
        pot_breaks = [social for social in self.model.schedule.agents if social.unique_id in pot_break_ids]

        for potential in pot_breaks:
            self.consider_connection(potential, method="REMOVE")


    def consider_connection(self, potential_agent, method):
        """Calculate the (Fermi Dirac) probability of agent being connected to 'potential agent' and based on method add or remove the connection randomly

        Args:
            potential_agent (Resident): the resident to consider
            method (str): "ADD" or "REMOVE"
        """
        p_ij = 1 / ( 1 + np.exp(self.model.fermi_alpha * (abs(self.opinion - potential_agent.opinion) - self.model.fermi_b)))

        if method == "ADD":
            if p_ij > random.random():
                self.model.graph.add_edge(self.unique_id, potential_agent.unique_id)

        if method == "REMOVE":
            if p_ij < random.random():
                self.model.graph.remove_edge(self.unique_id, potential_agent.unique_id)

    def move_pos(self):
        """
        Moves the location of an agent if they are unhappy based on happiness threshold, theta.
        """

        # # Correct the error in the model
        # neighbours = self.neighbours
        # opinions = 0
        # for neighbour in neighbours:
        #     opinions += neighbour.opinio
        # av_nbr_op = opinions/len(neighbours)

        social_infl, av_nbr_op = self.get_external_influences()

        # compare your opinion with the average of your neighbours using the fermi dirac equation.
        happiness = 1 / ( 1 + np.exp(self.model.fermi_alpha * (abs(self.opinion - av_nbr_op) - self.model.fermi_b)))

        # if happiness is below some threshold, move to a random free position in the neighbourhood.
        if happiness < self.model.happiness_threshold:
            self.model.grid.move_to_empty(self)
            self.model.movers_per_step += 1



    def step(self):
        """A full step of the agent, consisting of: updating network, updating location, updating opinion.
        """
        # update network
        self.new_social()
        self.remove_social()

        #update grid
        self.move_pos()

        #update opinion
        self.update_opinion()


class CityModel(Model):
    #these are the default parameters
    def __init__(self,
                 sidelength=20,
                 density=0.8,
                 m_barabasi=2,
                 fermi_alpha=5,
                 fermi_b=3,
                 social_factor=0.8,
                 connections_per_step=5,
                 opinion_max_diff=2,
                 happiness_threshold=0.8):

        # model variables
        self.sidelength = sidelength
        self.density = density
        self.m_barabasi = m_barabasi
        self.fermi_alpha = fermi_alpha
        self.fermi_b = fermi_b
        self.social_factor = social_factor
        self.connections_per_step = connections_per_step
        self.opinion_max_diff = opinion_max_diff
        self.happiness_threshold = happiness_threshold

        self.schedule = RandomActivation(self)
        self.movers_per_step = 0
        self.n_agents = 0

        # setting up the Residents:
        self.grid = SingleGrid(self.sidelength, self.sidelength, torus=True)
        self.initialize_population()

        # build a Barabasi Albert social network
        self.graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.m_barabasi)

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
        """Calculation of the Leibovici entropy index, using the spatial entropy packaged as described
            on the following github: https://github.com/Mr-Milk/SpatialEntropy

        Returns:
            [float]: [Leibovici entropy index]
        """
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
        """Calculation of the Altieri entropy index, using the spatial entropy packaged as described
            on the following github: https://github.com/Mr-Milk/SpatialEntropy

        Returns:
            [float]: [Altieri entropy index]
        """
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
        """Initialisation of the population on the 2D grid, with the density prescribed.
        """
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

    def run_model(self, step_count=1, desc="", pos=0, collect_during=True, collect_initial=False):
        """Method that runs the model for a fixed number of steps"""
        # A better way to do this is with a boolean 'running' that is True when initiated,
        # and becomes False when our end condition is met
        if collect_initial:
            self.datacollector.collect(self)

        for i in trange(step_count, desc=desc, position=pos):
            self.step()

            # collect data
            if collect_during:
                self.datacollector.collect(self)

                #set the counter of movers back to zero
                self.movers_per_step = 0

        if not collect_during:
            self.datacollector.collect(self)

#this has been replaced by batch_run.py
def main(argv):
    from .plot_graph import create_graph
    from .plot_grid import sim_grid_plot
    from matplotlib.pyplot import savefig, subplots, hist
    import networkx as nx

    model = CityModel(density=0.9,fermi_alpha=4, fermi_b=1, sidelength=15, opinion_max_diff=0.5, happiness_threshold=0.2)
    stepcount = 50

    model.run_model(step_count=stepcount)
    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    fig, axes = subplots(2,2)
    axes = axes.reshape(-1)

    sim_grid_plot(agent_df, grid_axis=[axes[2], axes[3]])
    create_graph(
        agent_df,
        model_df,
        graph_axes=axes[:2],
        layout=nx.spring_layout
        )
    fig.show()
    fig, ax = fig, ax = subplots(1, 2, )
    ax[0].hist(agent_df.loc[[stepcount], ["opinion"]], density = True)
    ax[1].plot(range(stepcount), model_df.movers_per_step, label = "Movers per step")
    fig.show()


if __name__=="__main__":
    import sys
    main(sys.argv[1:])

# %%

# %%
