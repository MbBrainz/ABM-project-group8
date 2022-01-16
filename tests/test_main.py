# This is the test file for the main.py file
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from fakenews_segregation.model import CityModel


def test_model_initialisation():
	model = CityModel(n_agents=10)
	assert model != None
