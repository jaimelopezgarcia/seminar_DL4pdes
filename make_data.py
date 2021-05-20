import os
from solvers import solve_heat_equation, post_process_and_save_he, solve_allen_cahn, post_process_and_save_ac
import fire

class RunSimulations():

    def __init__(self, results_dir = None):

        if not(results_dir):
            self._results_dir = './data'
        else:
            self._results_dir = results_dir

    def solve_heat_equation(self, nsamples, plot = False, dt = 1e-4, T_dt = 400, initial_conditions = "rectangles"):

        save_dir = os.path.join(self._results_dir,"HE")

        for sample in range(nsamples):

            name_simulation = initial_conditions+"{}_dt_{}_{}".format(sample,
                                                                      str(dt).replace("-","m"),
                                                                      str(T_dt).replace(".","p"))

            sols = solve_heat_equation(dt = dt, T_dt = T_dt,
                                       initial_condition = initial_conditions
                                       )

            post_process_and_save_he(sols, results_dir = save_dir, save_plots = plot,
                                    name_simulation = name_simulation)

    def solve_allen_cahn(self, nsamples, plot = False, n_elements = 60, T_dt = 200, ratio_speed = 10, eps = 0.01, save_dir = "AC",
                         initial_conditions = "random", duration = 0.05):

        save_dir = os.path.join(self._results_dir,save_dir)

        for sample in range(nsamples):

            name_simulation = initial_conditions+"{}_eps_{}_{}".format(sample,
                                                                      str(eps).replace("-","m"),
                                                                      str(T_dt).replace(".","p"))

            sols = solve_allen_cahn(eps = eps, T_dt = T_dt, ratio_speed = ratio_speed,


                                       initial_conditions = initial_conditions, n_elements = n_elements
                                       )

            post_process_and_save_ac(sols, results_dir = save_dir, save_plots = plot,
                                    name_simulation = name_simulation, duration = duration)






if __name__ == "__main__":
    fire.Fire(RunSimulations)
