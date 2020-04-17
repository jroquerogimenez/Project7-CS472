import numpy as np

class EquationODE(object):

    def __init__(self, constant_parameters_class, varying_parameters_class):

        self._generate_constant_parameters(constant_parameters_class.constant_parameters_dict)
        self.varying_parameters_dict = varying_parameters_class.varying_parameters_dict
        self.vector_dim = 10


    def ODE_equation(self, t, N):

        self._generate_varying_parameters(t, N)

        self._generate_matrix()
        self._generate_vector()
        return self.matrix.dot(N) + self.vector

    def _generate_constant_parameters(self, constant_parameters_dict):
        self.constant_parameters_dict = constant_parameters_dict
        self.tau_Inc = self.constant_parameters_dict.get('tau_Inc', 5) # Typical time: incubation period.
        self.tau_Rec = self.constant_parameters_dict.get('tau_Rec', 14) # Typical time: time to recovery.
        self.tau_Hos = self.constant_parameters_dict.get('tau_Hos', 21) # Typical time: hospitalization duration.
        self.lambda_A = self.constant_parameters_dict.get('lambda_A', 0.35) # Fraction of asymptomatic infectious cases.
        self.rho_Hos = self.constant_parameters_dict.get('rho_Hos', 0.02) # Rate of symptomatic patients requiring hospitalization.


    def _generate_varying_parameters(self, t, N):
        self.t, self.N = t, N
        self.tau_ConA = self.varying_parameters_dict['tau_ConA_func'](t,N) # Typical time: new contagion by asymptomatic case.
        self.tau_ConS = self.varying_parameters_dict['tau_ConS_func'](t,N) # Typical time: new contagion by symptomatic case.
        self.tau_DiaA = self.varying_parameters_dict['tau_DiaA_func'](t,N) # Rate of diagnosis for asymptomatic cases.
        self.tau_DiaS = self.varying_parameters_dict['tau_DiaS_func'](t,N) # Rate of diagnosis for symptomatic cases.
        self.rho_Fat = self.varying_parameters_dict['rho_Fat_func'](t,N) # Rate of fatal cases in hospitalized cases.
        self.F_ext = self.varying_parameters_dict['F_ext_func'](t,N) # Incoming asymptomatic patients from external source.

    def _generate_matrix(self):
        #N[0] = N_S; susceptible population:
        susceptible_row = np.array(		[0, 0, -1/self.tau_ConA, -1/self.tau_ConS, 0, 0, 0, 0, 0, 0])
        #N[1] = N_E; d N_e(t)/dt = 
        exposed_row = np.array(			[0, -1/self.tau_Inc, 1/self.tau_ConA, 1/self.tau_ConS, 0, 0, 0, 0, 0, 0])
        #N[2] = N_IA; d N_Asym(t)/dt = 
        asymptomatic_row = np.array(		[0, self.lambda_A/self.tau_Inc, -1/self.tau_Rec - 1/self.tau_DiaA, 0, 0, 0, 0, 0, 0, 0])
        #N[3] = N_IS; d N_Sym(t)/dt = 
        symptomatic_row = np.array(		[0, (1-self.lambda_A)/self.tau_Inc, 0, -1/self.tau_Rec - 1/self.tau_DiaS - self.rho_Hos, 0, 0, 0, 0, 0, 0])
        #N[4] = N_D; d N_D(t)/dt =
        diagnosed_row = np.array(		[0, 0, 1/self.tau_DiaA, 1/self.tau_DiaS, -1/self.tau_Rec, 0, 0, 0, 0, 0])
        #N[5] = N_H; d N_H(t)/dt = 
        hospitalized_row = np.array(		[0, 0, 0, self.rho_Hos, 0, -1/self.tau_Hos - self.rho_Fat, 0, 0, 0, 0])
        #N[6] = N_F; d N_F(t)/dt = 
        fatality_row = np.array(		[0, 0, 0, 0, 0, self.rho_Fat, 0, 0, 0, 0])
        #N[7] = N_R; recovered from hospital:
        recovered_row = np.array(		[0, 0, 0, 0, 1/self.tau_Rec, 1/self.tau_Hos, 0, 0, 0, 0])
        #N[8] = N_UR; undiagnosed recoveries. 
        recovered_undiagnosed_row = np.array(	[0, 0, 1/self.tau_Rec, 1/self.tau_Rec, 0, 0, 0, 0, 0, 0])
        #N[9] = Total number of diagnosed cases. Sum of N[4:8] (exclude 8)
        sum_diagnosed_cases = np.array(		[0, 0, 1/self.tau_DiaA, 1/self.tau_DiaS + self.rho_Hos, 0, 0, 0, 0, 0, 0])
        # Stacking rows to generate full matrix.
        self.matrix = np.vstack([susceptible_row,
                                 exposed_row,
                                 asymptomatic_row,
                                 symptomatic_row,
                                 diagnosed_row,
                                 hospitalized_row,
                                 fatality_row,
                                 recovered_row,
				 recovered_undiagnosed_row,
				 sum_diagnosed_cases])
        assert np.allclose(np.sum(self.matrix[:-1,:], axis = 0), np.zeros(self.vector_dim)),  't{}N{}tau{}'.format(self.t, self.N, self.tau_ConA)

    def _generate_vector(self):
        # d N(t)/dt = AN(t) + b
        self.vector = np.array([0, self.F_ext, 0, 0, 0, 0, 0, 0, 0, 0])











