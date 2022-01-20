# testing normal distribution related to views ----NOAH

import numpy as np
import matplotlib.pyplot as plt
 
# Creating a series of data of in range of 1-50.
x = np.linspace(0,1,200)

#Own view: 0.3533081091241377, theta: 0.9122463295755393
 
#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = (1 / (sd * np.sqrt(2* np.pi))) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density
 
#Calculate mean and Standard deviation.
mean = 0.3533081091241377
sd = 0.25*0.9122463295755393
 
#Apply function to the data.
pdf = normal_dist(x,mean,sd)
print(pdf)

#Plotting the Results
plt.plot(x,pdf , color = 'red')
plt.show()
# plt.xlabel('Data points')
# plt.ylabel('Probability Density')
# plt.show()