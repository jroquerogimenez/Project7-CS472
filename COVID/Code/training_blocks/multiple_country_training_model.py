import numpy as np
from pyabc import IntegratedModel, ModelResult
from .single_country_training_model import SingleCountryTrainingModelABC
import time





class MultipleCountryTrainingModelABC(IntegratedModel):
    '''
    This class is used if we run a Bayesian model selection procedure with pyabc.
    This class transforms dictionaries of ground truth data into a IntegratedModel instance with multiple models.
    '''
    def __init__(self,
 		 ground_truth_dict,
		 index_ground_truth_dict,
                 idiosyncratic_params_dict,
                 n_days_dict = {},
		 loss_function = lambda x,y : np.abs(x-y),
		 factor_eps = 1,
		 early_evaluation = True,
                 checks = False,
                 name = 'model0'):

        # Subclass IntegratedModel, need to implement integrated_simulate.
        super().__init__(name = name)

        self.name = name
        self.checks = checks # Type checks and dict key checks.
        if self.checks:

            print('Constructing a MultipleCountryTrainingModelABC instance.')
            assert isinstance(ground_truth_dict, dict), "The ground truth data and the index_array must be a dict" 
            assert isinstance(index_ground_truth_dict, dict), "The ground truth data and the index_array must be a dict" 
            assert isinstance(n_days_dict, dict)
            assert (index_ground_truth_dict.keys() == ground_truth_dict.keys()) and (idiosyncratic_params_dict.keys() == ground_truth_dict.keys()) and (initial_population_dict.keys() == ground_truth_dict.keys()), "Dict keys of ground_truth_dict and index_ground_truth_dict and/or idiosyncratic_params_dict and/or initial_population_dict do not match."

        # Multi-country dictionaries.
        self.ground_truth_dict = ground_truth_dict
        self.index_ground_truth_dict = index_ground_truth_dict
        self.idiosyncratic_params_dict = idiosyncratic_params_dict
        self.n_days_dict = n_days_dict
        self.n_countries = len(self.ground_truth_dict.keys())

        # Parameters for SingleCountryModelABC.
        self.loss_function = loss_function
        self.factor_eps = factor_eps
        self.early_evaluation = early_evaluation

        self.single_country_model_dict = {}

        # Store the country-wise IntegratedModel instances with SingleCountryModelABC instantiation.
        for key, ground_truth in self.ground_truth_dict.items():
            self.single_country_model_dict[key] = SingleCountryTrainingModelABC(ground_truth = ground_truth,
										index_ground_truth = self.index_ground_truth_dict[key],
										idiosyncratic_params = self.idiosyncratic_params_dict[key],
										n_days = self.n_days_dict.get(key, None),
										loss_function = self.loss_function,
										factor_eps = self.factor_eps,
										early_evaluation = self.early_evaluation,
									        checks = self.checks
								               )

        
    def integrated_simulate(self, pars, eps):

        self.eps = eps

        output_result = ModelResult(accepted = True, distance = 0)

        for key in self.single_country_model_dict.keys():
            if output_result.accepted is False:
                break

            sg_country_model_result = self.single_country_model_dict[key].integrated_simulate(pars, self.eps)

            # As soon as one of the countries rejects, the whole parameter set is rejected (i.e. as if sup distance for rejection).
	    # However, the final distance output will be the average distance.
            output_result.accepted = output_result.accepted and sg_country_model_result.accepted
            output_result.distance += sg_country_model_result.distance

        output_result.distance /= self.n_countries

        sum_stats = None # Not yet summary stats

        return ModelResult(accepted=(output_result.distance<self.eps) and output_result.accepted,
				     distance=output_result.distance,
				     sum_stats=sum_stats
       			  )

