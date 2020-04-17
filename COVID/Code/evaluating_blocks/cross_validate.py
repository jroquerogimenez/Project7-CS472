import numpy as np
import scipy as sp
import copy
from .single_country_evaluating_model import SingleCountryEvaluatingModel





class CrossValidateModel(object):

    def __init__(self,
		 abc_procedure,
		 minimum_epsilon = 0.1,
		 max_nr_populations = 3,
                ):

        self.abc_procedure_store = copy.deepcopy(abc_procedure)

        self.minimum_epsilon = minimum_epsilon
        self.max_nr_populations = max_nr_populations


    def cross_validate(self,
                       index_model_to_cross_validate,
		       training_dates_iterable,
		       n_days_cv,
                       country_to_cross_validate_list = None
		      ):

        assert index_model_to_cross_validate < len(self.abc_procedure_store.model_list)

        cv_loss = 0

        if country_to_cross_validate_list is None:
            country_to_cross_validate_list = list(self.abc_procedure_store.ground_truth_dict.keys())
        
        for cv_date in training_dates_iterable:

            history = self._one_cv_date_cross_validate(cv_date,
						       index_model_to_cross_validate,
						      )

            for country_to_cross_validate in country_to_cross_validate_list:

                cv_loss += self._one_cv_country_cross_validate(history,
							       cv_date,
							       country_to_cross_validate,
							       n_days_cv)

        cv_loss /= (len(training_dates_iterable) * len(country_to_cross_validate_list))

        return cv_loss


    def _one_cv_date_cross_validate(self,
				    cv_date,
				    index_model_to_cross_validate,
				   ):

        self.abc_procedure = copy.deepcopy(self.abc_procedure_store)

        n_days_dict = {key: cv_date for key in self.abc_procedure.ground_truth_dict.keys()}

        self.abc_procedure.set_country_model_ground_truth(ground_truth_dict = self.abc_procedure.ground_truth_dict,
    						      index_ground_truth_dict = self.abc_procedure.index_ground_truth_dict,
    						      initial_population_dict = self.abc_procedure.initial_population_dict,
    						      idiosyncratic_params_dict = self.abc_procedure.idiosyncratic_params_dict,
    					              n_days_dict = n_days_dict,
    						      checks = self.abc_procedure.checks
    						     )	

        self.abc_procedure.model_list, self.abc_procedure.model_prior_list = [], []

        selected_model_to_cross_validate = self.abc_procedure.model_prior_distributions_list[index_model_to_cross_validate]
        self.abc_procedure.add_parameter_prior_model(model_prior_distributions_list = [selected_model_to_cross_validate])
        self.abc_procedure.transitions_list = [self.abc_procedure.transitions_list[index_model_to_cross_validate]]

        abc = self.abc_procedure.generate_abc_class()
        _ = abc.new("sqlite:///test_local.db")
        history = abc.run(minimum_epsilon = self.minimum_epsilon, max_nr_populations = self.max_nr_populations)

        return history

    def _one_cv_country_cross_validate(self,
				       history,
				       cv_date,
				       country_to_cross_validate,
				       n_days_cv
				      ):

        initial_population_country = self.abc_procedure_store.initial_population_dict[country_to_cross_validate]
        idiosyncratic_params_country = self.abc_procedure_store.idiosyncratic_params_dict[country_to_cross_validate]

        posterior_simulations = SingleCountryEvaluatingModel(abc_history_model = history.get_distribution(),
                                                            n_days_simulation = cv_date + n_days_cv,
                                                            initial_population = initial_population_country,
                                                            idiosyncratic_params = idiosyncratic_params_country,
                                                            checks = self.abc_procedure_store.checks)

        distance = np.mean([self.abc_procedure_store.loss_function(
    posterior_simulations.posterior_simulations_mean[self.abc_procedure_store.index_ground_truth_dict[country_to_cross_validate], t],
    self.abc_procedure_store.ground_truth_dict[country_to_cross_validate][:,t]
		                                                  ) for t in np.arange(cv_date, cv_date + n_days_cv)
                           ])
# If we want the CV loss to be meaningful, maybe we need to swap the two elements here: the loss is an average of rescaled time point losses, rescaled by the value of the ground truth. Maybe for the CV we want to rescale by the posterior simulation: that allows us to extrapolate outside of the ground truth.
        return distance



