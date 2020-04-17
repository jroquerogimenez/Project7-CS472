import numpy as np
from pyabc import ABCSMC
from .multiple_country_training_model import MultipleCountryTrainingModelABC
from ..customize_blocks.aux_functions import loss_real_world
from pyabc import MedianEpsilon, LocalTransition, SingleCoreSampler



class ModelSelectionABC(object):
    '''
    This class contains the lists that are fed to the ABCSMC class. In particular, we construct the lists for the different models, each of them characterized by a different choice of parameters and of generating process. 
    Each model is in turn an IntegratedModel, potentially based on several countries.
    Inputs: 
    - ground_truth_dict: dict(keys = strings with country names).



    '''
    def __init__(self):

        # These are the lists that will store the main inputs of ABCSMC.
        self.model_prior_list = []
        self.model_list = []
        self.class_already_created = False


    def set_country_model_ground_truth(self,
    		                       ground_truth_dict,
		 	               index_ground_truth_dict,
		 	               idiosyncratic_params_dict,
				       n_days_dict = {},
                                       checks = False
                	              ):

        # Need to run this function prior to adding parameter (i.e. prior) models.
        self.checks = checks
        if self.checks:

            assert isinstance(ground_truth_dict, dict), "The ground_truth_dict input must be a dict."
            assert isinstance(index_ground_truth_dict, dict), "The index_ground_truth_dict input must be a dict."
            assert isinstance(idiosyncratic_params_dict, dict), "The idiosyncratic_params_dict input must be a dict."
            assert isinstance(n_days_dict, dict), "The n_days_dict input must be a dict."
            assert (index_ground_truth_dict.keys() == ground_truth_dict.keys()), "Dict keys of ground_truth_dict and index_ground_truth_dict do not match." 
            assert (initial_population_dict.keys() == ground_truth_dict.keys()), "Dict keys of ground_truth_dict and initial_population_dict do not match."
            assert (idiosyncratic_params_dict.keys() == ground_truth_dict.keys()), "Dict keys of ground_truth_dict and idiosyncratic_params_dict do not match."
            print("Setting up the ground truth data on which we fit the Bayesian models.")
            print("Ground truth data coming from {} number of countries.".format(len(ground_truth_dict.keys())))


        # These dictionaries will be directly fed to the IntegratedModel classes according to the number of countries in the dicts.
        # No need to do anything as they are added along with new model prior parameter sets.
        self.ground_truth_dict = ground_truth_dict
        self.index_ground_truth_dict = index_ground_truth_dict
        self.idiosyncratic_params_dict = idiosyncratic_params_dict
        self.n_days_dict = n_days_dict


        self.n_countries = len(self.ground_truth_dict.keys())


        # Set some default values when instance is created.
        if not self.class_already_created:
            self.set_country_model_properties()
            self.set_abc_model_properties()
            self.class_already_created = True
        # Attention: calling again this method to modify the set of ground truth data from countries does not reset the prior parameter model.



    def set_country_model_properties(self,
		  		     loss_function = loss_real_world,
				     factor_eps = 1,
				     early_evaluation = True,
				    ):


        # Run this function to overwrite default.
        self.loss_function = loss_function
        self.factor_eps = factor_eps
        self.early_evaluation = early_evaluation


    def add_parameter_prior_model(self,
				  model_prior_distributions_list
				 ):


        if self.checks:
            print("Currently {} models to give to the ABCSMC class.".format(len(self.model_list)))
            print("Adding {} new models to the ABCSMC class.".format(len(model_prior_distributions_list)))


        for ind in np.arange(len(model_prior_distributions_list)):

            self.model_prior_list.append(model_prior_distributions_list[ind])

            self.model_list.append(MultipleCountryTrainingModelABC(ground_truth_dict = self.ground_truth_dict,
						                   index_ground_truth_dict = self.index_ground_truth_dict,
						                   idiosyncratic_params_dict = self.idiosyncratic_params_dict,
						                   n_days_dict = self.n_days_dict,
						                   loss_function = self.loss_function,
						                   factor_eps = self.factor_eps,
						                   early_evaluation = self.early_evaluation,
						                   checks = self.checks,
						                   name = 'model_{}'.format(ind)
						                  )
                                  )


    def set_abc_model_properties(self,
                                 population_strategy = 100,
				 transitions_list = [LocalTransition(k_fraction=.3)],
				 epsilon_strategy = MedianEpsilon(1000, median_multiplier = 1),
				 sampler = SingleCoreSampler()
				 ):


        # Run this function to overwrite default.
        self.population_strategy = population_strategy
        self.transitions_list = transitions_list
        self.epsilon_stategy = epsilon_strategy
        self.sampler = sampler


    def generate_abc_class(self):


        assert len(self.model_list) == len(self.model_prior_list), "Mismatch: Model_list {}. Model_prior_list {}.".format(len(self.model_list), len(self.model_prior_list))
        assert len(self.model_list) == len(self.transitions_list), "Mismatch: Model_list {}. Transitions_list {}.".format(len(self.model_list), len(self.transitions_list))


        # This is the main function and goal of ModelSelectionABC class: to provide the ABCSMC instance.
        abc = ABCSMC(models = self.model_list,
                     parameter_priors = self.model_prior_list,
                     population_size = self.population_strategy,
                     transitions = self.transitions_list,
	             eps = self.epsilon_stategy,                          
                     sampler = self.sampler
                    )

        return abc





