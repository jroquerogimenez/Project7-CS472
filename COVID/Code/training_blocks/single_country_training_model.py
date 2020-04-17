import numpy as np
import pandas as pd
import pyabc, time
from pyabc import IntegratedModel, ModelResult
import scipy.integrate as spint
from ..epidemiology_blocks.equation_model import EquationODE
from ..epidemiology_blocks.parameter_model import ConstantParameter, VaryingParameter


class SingleCountryTrainingModelABC(IntegratedModel):
    '''
    This is the ABC model class that integrates the sample generation and the acceptance/rejection steps.
    Computes on-the-fly cumulated loss to reject the sample as soon as some threshold is attained.
    Improves speed as ODE does not need to be fully integrated to reject it.
    '''

    def __init__(self,
 		 ground_truth,
		 index_ground_truth,
                 idiosyncratic_params = {},
                 n_days = None,
		 loss_function = lambda x,y : np.abs(x-y),
		 factor_eps = 1,
		 early_evaluation = True,
                 checks = False,
                ):


        # Subclass IntegratedModel, need to implement integrated_simulate.
        super().__init__()


        self.checks = checks
        if self.checks:
            print('Constructing a SingleCountryTrainingModelABC instance.')
            assert isinstance(ground_truth, np.ndarray), "The ground truth data and the index_array must be a np.array" 
            assert isinstance(index_ground_truth, np.ndarray), "The ground truth data and the index_array must be a np.array" 
            assert isinstance(idiosyncratic_params, dict), "idiosyncratic_params must be a dict."
            assert (ground_truth.ndim == 2) and (index_ground_truth.ndim == 1), "Ground truth must be a 2-dim array with rows equal to the different ground truth time series."


        # Individual country ground truth.
        self.ground_truth = ground_truth
        self.index_ground_truth = index_ground_truth
        self.idiosyncratic_params = idiosyncratic_params
        self.name = idiosyncratic_params['name_country']
        self.initial_population = idiosyncratic_params['population_size_{}'.format(self.name)]


        # Parameters for SingleCountryModelABC
        self.loss_function = loss_function
        self.factor_eps = factor_eps
        self.early_evaluation = early_evaluation


        if n_days is None:
            self.n_days = self.ground_truth.shape[1]
        else:
            self.n_days = n_days

        

    def integrated_simulate(self, pars, eps):

        self.eps = eps

        distance, sum_stats = self._evaluate_simulation(pars, self.idiosyncratic_params)

        return ModelResult(accepted=(distance<eps), distance=distance, sum_stats=sum_stats)



    def _evaluate_simulation(self, pars, idiosyncratic_params):
        # For a given set of pars, construct the model_equation class. 
        # First transform the pars input into the two dictionaries required for model_equation

        self.constant_parameters_class = ConstantParameter(**pars, **idiosyncratic_params)
        self.varying_parameters_class = VaryingParameter(**pars, **idiosyncratic_params)

        self.equation_model = EquationODE(self.constant_parameters_class, self.varying_parameters_class) 

        y0 = np.zeros(self.equation_model.vector_dim)
        y0[0] = self.initial_population

        simulation = spint.solve_ivp(fun = self.equation_model.ODE_equation,
         			     t_span = (0, self.n_days - 1), # Same as saying np.arange(self.n_days)
				     y0 = y0, 
 				     t_eval = np.arange(self.n_days)
				    )

        sum_stats = None # Nothing yet to add here.


        distance = np.mean([self.loss_function(
			   np.array(simulation['y'])[self.index_ground_truth, t],
 	                   self.ground_truth[:,t]
				  )
                               for t in np.arange(self.n_days)
 		       ])
        return distance, sum_stats



