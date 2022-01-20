from mesa import Agent, Model
from mesa import Agent, Model
# from mesa.time import RandomActivation

"""This file should contain the model class. 
If the file gets large, it may make sense to move the complex bits into other files, 
 this is the first place readers will look to figure out how the model works.
"""

# for reference, https://github.com/projectmesa/mesa/tree/main/examples/schelling is a good structure
# docs: https://mesa.readthedocs.io/en/master/best-practices.html 


class Resident(Agent):
    def __init__(self, name, model):
        pass

    def step(self):
        pass
        # Whatever an agent does when activated --> use other methods. 
        # MESA assumes (I believe) that each agent has a step() method, which is called when 'running' the model 

class CityModel(Model):
    def __init__(self, n_agents):
        pass

    def step(self):
        # the scheduler uses the step() functions of the agents 
        self.schedule.step()

# this is a comment by Sasha
