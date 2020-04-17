import numpy as np
import scipy as sp
from ..customize_blocks.parameter_function_generators import *


class ConstantParameter(object):

    def __init__(self, **pars):

        self.constant_parameters_dict = {'tau_Inc': pars['tau_Inc'],  # Gaussian prior N(5,2)
                            'tau_Rec': pars['tau_Rec'],  # Gaussian prior N(15,3)
                            'tau_Hos': pars['tau_Hos'],  # Gaussian prior N(14,3)
                            'lambda_A': pars['lambda_A'], # Gaussian prior N(0.35, 0.1)
                            'rho_Hos': pars['rho_Hos'] # Gaussian prior N(0.02,0.005)   
                            }


class VaryingParameter(object):

    def __init__(self, **pars):
        self.tau_ConA_func = generate_tau_ConA(**pars)
        self.tau_ConS_func = generate_tau_ConS(**pars)
        self.tau_DiaA_func = generate_tau_DiaA(**pars)
        self.tau_DiaS_func = generate_tau_DiaS(**pars)
        self.rho_Fat_func = generate_rho_Fat(**pars)
        self.F_ext_func = generate_F_ext(**pars)

        self.varying_parameters_dict = {'tau_ConA_func': self.tau_ConA_func, 
                 'tau_ConS_func': self.tau_ConS_func,
                 'tau_DiaA_func': self.tau_DiaA_func,
                 'tau_DiaS_func': self.tau_DiaS_func,
                 'rho_Fat_func': self.rho_Fat_func,
                 'F_ext_func': self.F_ext_func,
                 }


