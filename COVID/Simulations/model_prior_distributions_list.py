#!/miniconda3/bin/python
import numpy as np
from pyabc import Distribution, RV
import _pickle as pkl


list_distributions = [Distribution(tau_Inc = RV("uniform", 8, 10),\
                                   tau_Rec = RV("uniform", 10, 10),\
                                   tau_Hos = RV("uniform", 10, 10),\
                                   lambda_A = RV("uniform", 0.1, 0.3),\
                                   rho_Hos = RV("uniform", 0.01, 0.05),\
				   rho_Fat_initial_rate = RV("uniform", 0.01, 0.2),\
				   rho_Fat_growth_rate = RV("uniform", 0.01,0.2),\
                                   tau_DiaA_initial_rate = RV("uniform", 1, 3),\
                                   tau_DiaS_initial_rate = RV("uniform", 1, 3),\
                                   tau_DiaS_growth_rate = RV("uniform", 1, 10),\
                                   tau_ConA_initial_rate = RV("uniform", 0.1,0.3),\
                                   tau_ConS_initial_rate = RV("uniform", 0.1,0.3),\
                                   F_ext_initial_rate = RV("uniform", 0.01,0.2),\
  				   isolation_1_effect = RV("uniform", 0.1, 0.5),\
  				   isolation_2_effect = RV("uniform", 0.1, 0.5),\
				   isolation_3_effect = RV("uniform", 1, 2),\
                                              )]




with open('.pkl_model_prior_distributions.pkl', 'wb') as f:
    pkl.dump({'data': list_distributions}, f)



