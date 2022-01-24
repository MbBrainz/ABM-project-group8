from mesa.batchrunner import BatchRunnerMP
from polarization.model import CityModel
def main():
    fixed_params = {"width": 10, "height": 10,}
    var_params = {"density": [0.6, 0.8], "m_barabasi": [1,2,3]}

    batch = BatchRunnerMP(CityModel,
                            nr_processes=1,
                            variable_parameters=var_params,
                            fixed_parameters=fixed_params,
                            display_progress=True,
                            iterations=1)
    batch.run_all()
# %%
if __name__ == "__main__":
    main()