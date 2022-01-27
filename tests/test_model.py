# This is the test file for the main.py file
from polarization.model import CityModel
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import unittest
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def clear_model(model: CityModel):
    model.agents = []
    model.n_agents = 0
    model.schedule = RandomActivation(model)
    model.grid = SingleGrid(model.params.sidelength, model.params.sidelength, torus=False)

class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = CityModel()
        return super().setUp()

    def test_model_initialisation(self):
        assert self.model != None
