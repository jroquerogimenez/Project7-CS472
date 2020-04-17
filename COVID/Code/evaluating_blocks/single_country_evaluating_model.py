import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
from ..epidemiology_blocks.equation_model import EquationODE
from ..epidemiology_blocks.parameter_model import ConstantParameter, VaryingParameter
from ..customize_blocks.aux_functions import plot_country_ground_truth




class SingleCountryEvaluatingModel(object):
    '''
    This class analyzes the output of ONE set of prior distributions (i.e. one model in the Bayesian model selection framework).
    This class evaluates the output of the ABCSMC training procedure on one single country.

    Input:
	- abc_history_model : Contains the posterior distribution of the parameters in the chosen model.
	- n_days_simulation (int): Unlike the training class, here we are given all the necessary tools to generate the time series as far as needed.
	- idiosyncratic_params (dict, suited for the particular country): Other parameters needed for simulating the time series.


    Output:
	- self.posterior_simulations (np.ndarray 11 x n_days_simulation x number of posterior samples): 
	- self.posterior_simulations_mean (np.ndarray 11 x n_days_simulation): 
	- self.posterior_simulations_std (np.ndarray 11 x n_days_simulation): 

        - self.plot_posterior_aggregate(indices to plot)
    '''


    def __init__(self, 
		 abc_procedure,
                 abc_history,
                 name_country,
		 n_days_simulation = None,
                 index_model = 0,
                 index_population_iteration = None,
		 checks = False
		):



        self.abc_procedure = abc_procedure
        self.abc_history = abc_history
        self.name_country = name_country
        self.checks = checks

        self.param_posterior_samples, self.param_posterior_weights = self.abc_history.get_distribution(m = index_model, t = index_population_iteration)        
        self.idiosyncratic_params = self.abc_procedure.idiosyncratic_params_dict[self.name_country]
        self.initial_population = self.idiosyncratic_params['population_size_{}'.format(self.name_country)]
        self.ground_truth = self.abc_procedure.ground_truth_dict[self.name_country]

        if n_days_simulation is not None:
            self.n_days_simulation = n_days_simulation
        else:
            self.n_days_simulation = self.ground_truth.shape[1]


        self.legend_base = ['Susceptible',
                            'Exposed',
                            'Infectious Asymptomatic',
                            'Infectious Symptomatic',
                            'Diagnosed',
                            'Hospitalized',
                            'Fatality',
                            'Recovered',
			    'Total number of undiagnosed recoveries',
                            'Total number of diagnosed cases'
			   ]


        self.posterior_simulations = np.empty((10,self.n_days_simulation,0))

        self._generate_posterior_samples()
        self._generate_posterior_statistics()

        assert self.posterior_simulations_mean.shape == (10, self.n_days_simulation)

    def _generate_posterior_statistics(self):

        self.posterior_simulations_mean = np.average(self.posterior_simulations,
						     weights = self.param_posterior_weights,
						     axis = -1
						    )

        self.posterior_simulations_std = np.sqrt(np.average((self.posterior_simulations 
						             - np.expand_dims(self.posterior_simulations_mean, axis = -1))**2, 
						    	    weights = self.param_posterior_weights,
						            axis = -1
						  	   )
						)
        

    def _generate_posterior_samples(self):

        for _, row in self.param_posterior_samples.iterrows(): 

            self._generate_model_equation(row.to_dict(), self.idiosyncratic_params)

            self.posterior_simulations = np.concatenate((self.posterior_simulations,
						 	 np.expand_dims(self._generate_simulation(), axis = -1)
							),
							axis = -1)

        self.n_reps_posterior_samples = self.posterior_simulations.shape[-1]


    def _generate_model_equation(self, pars, idiosyncratic_params):
        # For a given set of pars, construct the model_equation class.
        # First transform the pars input into the two dictionaries required for model_equation

        self.constant_parameters_class = ConstantParameter(**pars, **idiosyncratic_params)
        self.varying_parameters_class = VaryingParameter(**pars, **idiosyncratic_params)

        self.equation_model = EquationODE(self.constant_parameters_class, self.varying_parameters_class)


    def _generate_simulation(self):
        # This simulation is for a single country. The characteristics for the posterior simulations for that country are in the self.equation_model class and in the self.initial_population value.
        # The self.equation_model class in turn depends on the country through the idiosyncratic_params dict.
        y0 = np.zeros(self.equation_model.vector_dim)
        y0[0] = self.initial_population

        simulation = spint.solve_ivp(fun = self.equation_model.ODE_equation,
                                     t_span = (0, self.n_days_simulation - 1), # Same as saying np.arange(self.n_days_simulation)
                                     y0 = y0,
                                     t_eval = np.arange(self.n_days_simulation)
                                    )

        return simulation['y']


    def plot_posterior_aggregate(self,
				 indices,
				 plot_ground_truth = False,
				 t_max = None
				):

        if type(indices)==list and type(indices[0]) == str:
            new_index = []
            while len(indices)>0:
                new_index.append(self.legend_base.index(indices.pop()))
            indices = np.array(new_index)
        assert isinstance(indices, np.ndarray), 'indices is either a np.array of int or a list of str from self.legend_base.'

        if t_max is None:
            t_max = self.n_days_simulation


        fig, ax = plt.subplots(figsize=(20, 10))

        for index in indices:
            ax.errorbar(x = np.arange(t_max),
                     y = self.posterior_simulations_mean[index,:],
                     yerr = 2*self.posterior_simulations_std[index,:],
		     label = self.legend_base[index]
                    )

        if plot_ground_truth:
            t_max_ground_truth = np.minimum(t_max, self.ground_truth.shape[1])
            ax.plot(np.arange(t_max_ground_truth),
                    self.ground_truth[0,:t_max_ground_truth],
                    label = 'Ground truth diagnosed.')
            ax.plot(np.arange(t_max_ground_truth),
                    self.ground_truth[1,:t_max_ground_truth],
                    label = 'Ground truth fatality.')

        ax.legend()
        _ = plt.plot()

    def plot_posterior_individual(self,
				 indices,
				 plot_ground_truth = False,
				 t_max = None,
                                 particle_index = 0
				):

        if type(indices)==list and type(indices[0]) == str:
            new_index = []
            while len(indices)>0:
                new_index.append(self.legend_base.index(indices.pop()))
            indices = np.array(new_index)
        assert isinstance(indices, np.ndarray), 'indices is either a np.array of int or a list of str from self.legend_base.'

        if t_max is None:
            t_max = self.n_days_simulation

        fig, ax = plt.subplots(figsize=(20, 10))

        for index in indices:
            ax.plot(np.arange(t_max),
                    self.posterior_simulations[index,:t_max,particle_index],
		    label = self.legend_base[index]
                    )

        if plot_ground_truth:
            t_max_ground_truth = np.minimum(t_max, self.ground_truth.shape[1])
            ax.plot(np.arange(t_max_ground_truth),
                    self.ground_truth[0,:t_max_ground_truth],
                    label = 'Ground truth diagnosed.')
            ax.plot(np.arange(t_max_ground_truth),
                    self.ground_truth[1,:t_max_ground_truth],
                    label = 'Ground truth fatality.')

        ax.legend()
        _ = plt.plot()

