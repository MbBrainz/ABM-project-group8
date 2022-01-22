# this file is meant to enable benchmarking of the models current state
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt



def time_model_step(width, height, model):
    start_time = perf_counter()
    model.step()
    elapsed_time = perf_counter() - start_time

    return elapsed_time

# benchmark the model for multiple sizes with one step
def main():
    print(f"start benchmarking")
    # generate models with different number of agents and time single step
    width_list = np.arange(5, 50, 5)
    agents_list = np.square(width_list)
    time_list = []

    for width in width_list:
        time_list.append(time_model_step(width, width, None))

    # create a plot of the time data
    plt.plot(agents_list, time_list, label="agents")
    plt.xlabel("number of agents in model")
    plt.ylabel("time per step (s)")
    plt.yscale('log')
    plt.show()
    # (optional) Extrapolate data


if __name__ == "__main__":
    main()