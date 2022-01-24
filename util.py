import numpy as np
import matplotlib.pyplot as plt

def fermi_dirac_graph(d, FERMI_ALPHA, FERMI_B):
    """
    A graph to visualise the effect of alpha and b on the probability to connect for a given distance.
    """
    pij = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(d) - FERMI_B)))
    return pij

def plot_fermidirac():
    params = [(10,1),(1,3)]

    distances = np.linspace(0, 10, 100)
    plot_data =[]
    for param in params:
        data = {
            "y": fermi_dirac_graph(distances, param[0], param[1]),
            "alpha": param[0],
            "b": param[1]
            }
        plot_data.append(data)

    for data in plot_data:
        plt.plot(distances, data['y'], label = f"alpha = {data['alpha']}, b = {data['b']}")
    plt.xlabel("Absolute distance in the political belief-space")
    plt.ylabel("Probability to connect")
    plt.title("Fermi-Dirac probability to connect to another node based on political distance")
    plt.legend()
    plt.show()

plot_fermidirac()
