import numpy as np
import matplotlib.pyplot as plt

def fermi_dirac_graph(d, FERMI_ALPHA, FERMI_B):
    """
    A graph to visualise the effect of alpha and b on the probability to connect for a given distance.
    """
    pij = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(d) - FERMI_B)))
    return pij

distances = np.linspace(0, 10, 100)
probabilities2 = fermi_dirac_graph(distances, 1, 2)
probabilities3 = fermi_dirac_graph(distances, 1, 3)
probabilities4 = fermi_dirac_graph(distances, 1, 4)


# plt.plot(distances, probabilities2, label = "alpha = 1, b = 2")
# plt.plot(distances, probabilities3, label = "alpha = 1, b = 3")
# plt.plot(distances, probabilities4, label = "alpha = 1, b = 4")
# plt.xlabel("Absolute distance in the political belief-space")
# plt.ylabel("Probability to connect")
# plt.title("Fermi-Dirac probability to connect to another node based on political distance")
# plt.legend()
# plt.show()
