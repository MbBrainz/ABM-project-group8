import unittest
from mesa.time import RandomActivation
from networkx import barabasi_albert_graph, complete_graph, empty_graph, to_edgelist

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from polarization.model import CityModel, Resident
from test_model import clear_model
from unittest.mock import patch


global N_POTENTIAL_CONNECTIONS
N_POTENTIAL_CONNECTIONS = 1



class TestResident(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CityModel(width=5,height=5, m_barabasi=2, seed=711)
        clear_model(self.model)
        self.model.density = 1
        self.model.initialize_population()
        self.model.graph = barabasi_albert_graph(n=self.model.n_agents, m=self.model.m_barabasi)

        self.test_agent = self.model.schedule.agents[0]

        self.random_random_patcher = patch('polarization.model.random.random')
        self.random_choices_patcher = patch('polarization.model.random.choices')

        return super().setUp()

    def test_initialisation(self):
        assert self.test_agent != 0
        assert self.test_agent.model != 0
        assert self.test_agent.pos != 0
        assert self.test_agent.pos == (0,0)
        assert len(self.model.schedule.agents) == self.model.width * self.model.height

    def test_new_social(self):
        # Inject the return value of the random.choices()
        self.patch_random()

        self.model.graph = empty_graph(n=self.model.n_agents)
        self.test_agent.new_social([1])

        assert len(to_edgelist(self.model.graph)) == 1, "edgelist is empty"
        assert to_edgelist(self.model.graph)[0] == (1)

    def test_remove_social(self):
        # Inject the return value of the random.choices() to
        self.patch_random()

        self.model.graph = complete_graph(n=self.model.n_agents)
        self.test_agent.remove_social([1])
        assert ((self.test_agent.unique_id, 1) in to_edgelist(self.model.graph)) == False, "Edge has not been removed"

    def patch_random(self):
        self.test_resident = Resident(unique_id=1, model=self.model, pos=(0,1))

        self.random_patch = self.random_random_patcher.start()
        self.random_patch.return_value = 1

        self.choices_patch = self.random_choices_patcher.start()
        self.choices_patch.return_value = [self.test_resident]

        self.addCleanup(self.random_random_patcher.stop)
        self.addCleanup(self.random_choices_patcher.stop)
