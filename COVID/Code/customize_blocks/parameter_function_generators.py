import numpy as np
import scipy as sp
import datetime

date_first_report = datetime.date(2020,1,22)
future = datetime.date.today() + datetime.timedelta(1000)

def aux_func_contagion(**pars):

    name_country = pars['name_country']
    isolation_1_date = (pars.get('isolation_1_{}'.format(name_country), future) - date_first_report).days
    isolation_2_date = (pars.get('isolation_2_{}'.format(name_country), future) - date_first_report).days
    isolation_3_date = (pars.get('isolation_3_{}'.format(name_country), future) - date_first_report).days
    isolation_date_array = np.array([isolation_1_date, isolation_2_date, isolation_3_date])

    if pars.get('isolation_1_effect_{}'.format(name_country), None) is not None:
        isolation_effect_array = np.array([pars['isolation_1_effect_{}'.format(name_country)],
  			  	       pars['isolation_2_effect_{}'.format(name_country)],
				       pars['isolation_3_effect_{}'.format(name_country)]])

    else:
        isolation_effect_array = np.array([pars['isolation_1_effect'],
  			  	       pars['isolation_2_effect'],
				       pars['isolation_3_effect']])
    return isolation_date_array, isolation_effect_array



def generate_tau_ConA(**pars):
    # Asymptomatic infectious individuals typical contagion time.
    isolation_date_array, isolation_effect_array = aux_func_contagion(**pars)
    name_country = pars['name_country']
    multiplier = pars.get('isolation_effect_multiplier_{}'.format(name_country),1)

    initial_rate = pars['tau_ConA_initial_rate'] 
    def function(t, N):
        rate = (initial_rate +
                multiplier * np.sum(np.multiply(np.multiply(isolation_effect_array,(t-isolation_date_array)), t>isolation_date_array)) )

        scaling = (np.sum(N)/N[0])#
        return rate*scaling

    return function 


def generate_tau_ConS(**pars):
    # Symptomatic infectious individuals typical contagion time.
    isolation_date_array, isolation_effect_array = aux_func_contagion(**pars)

    name_country = pars['name_country']
    multiplier = pars.get('isolation_effect_multiplier_{}'.format(name_country),1)
    initial_rate = pars['tau_ConS_initial_rate']
    def function(t, N):
        rate = (initial_rate +
                multiplier * np.sum(np.multiply(np.multiply(isolation_effect_array,(t-isolation_date_array)), t>isolation_date_array)) )

        scaling = (np.sum(N)/N[0])#
        return rate*scaling
    return function

def generate_tau_DiaA(**pars):
    # Asymptomatic individuals diagnosis time
    name_country = pars['name_country']
    testing_capacity = pars.get('testing_capacity_{}'.format(name_country), None)
    initial_rate = pars['tau_DiaA_initial_rate_{}'.format(name_country)]
    initial_rate_squared = pars.get('tau_DiaA_initial_rate_sq_{}'.format(name_country), 0)
    
    if testing_capacity is not None:
        function = lambda t,N: (initial_rate)/(1+testing_capacity[int(t)]) + initial_rate_squared/(1+testing_capacity[int(t)])**2
    else:
        function = lambda t,N: initial_rate*(1+np.log(1+np.abs(N[2])))

    return function

def generate_tau_DiaS(**pars):
    # Symptomatic individuals diagnosis time
    name_country = pars['name_country']
    testing_capacity = pars.get('testing_capacity_{}'.format(name_country), None)
    initial_rate = pars['tau_DiaS_initial_rate_{}'.format(name_country)]
    initial_rate_squared = pars.get('tau_DiaS_initial_rate_sq_{}'.format(name_country), 0)

    if testing_capacity is not None:
        function = lambda t,N: (initial_rate)/(1+testing_capacity[int(t)]) + initial_rate_squared/(1+testing_capacity[int(t)])**2
    else:
        function = lambda t,N: initial_rate
#/(1+np.sum(np.multiply(isolation_effect_array, t>isolation_date_array)))


    return function

def generate_rho_Fat(**pars):
    # Fatality rate
    # Potential model parameters: rho_Fat_initial_rate,elderly_population_pct_{},rho_Fat_growth_rate_{},rho_Fat_growth_rate_sqrt_{},
    name_country = pars['name_country']
    hospital_capacity = pars['hospital_capacity_{}'.format(name_country)]
    
    if pars.get('rho_Fat_initial_rate_{}'.format(name_country), None) is not None:
        initial_rate = pars['rho_Fat_initial_rate_{}'.format(name_country)]#*pars.get('elderly_population_pct_{}'.format(name_country), 1)
    else:
        initial_rate = pars['rho_Fat_initial_rate']*pars.get('elderly_population_pct_{}'.format(name_country), 1)
    
    growth_rate = pars.get('rho_Fat_growth_rate_{}'.format(name_country), 0)
    growth_rate_log = pars.get('rho_Fat_growth_rate_sqrt_{}'.format(name_country), 0)


    def function(t, N):
        increase = growth_rate*(N[5]>hospital_capacity)
        increase2 = growth_rate_log*np.log(1 + np.abs((N[5]-hospital_capacity)))*(N[5]>hospital_capacity)
        return initial_rate + increase + increase2
    return function

def generate_F_ext(**pars):
    # Arrival rate of asymptomatic individuals from exterior source.
    isolation_date_array, isolation_effect_array = aux_func_contagion(**pars)
    name_country = pars['name_country']

    initial_date = pars["F_ext_initial_date_{}".format(name_country)]
    if pars.get('F_ext_initial_rate_{}'.format(name_country), None) is not None:
        initial_rate = pars['F_ext_initial_rate_{}'.format(name_country)]
    else:
        initial_rate = pars["F_ext_initial_rate"]
    function = lambda t,N: initial_rate*(t>initial_date)

    return function
