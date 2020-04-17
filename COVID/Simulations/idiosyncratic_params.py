import numpy as np
import scipy as sp
import _pickle as pkl
import datetime


'''
For each country, need to update the two dictionaries.
keys are country names.
'''

testing_united_kingdom = np.array([
0,0,0,0,0, #week 01/22
0,0,0,0,50,50,50, #week 01/27
100,100,100,100,100,100,100,#week 02/01 start:120end:1100
500,500,500,500,500,500,500, #week 02/10 start: 1100end:4500
300,300,300,300,300,300,300, # week 02/17 start:4500end6500
1000,1000,1000,1000,1000,1000,1000, #week 02/24 start:6500end13500
1700,1700,1700,1700,1700,1700,1700, #week 03/02 start:13500end:25000
2800,2800,2800,2800,2800,2800,2800,  #week 03/09 start:25000end:44000
5700,5700,5700,5700,5700,5700,5700, #week 03/16 start:44000end:84000
7100,7100,7100,7100,7100,7100,7100,#week 03/23 start:84000end:135000
11000,11000,11000,11000,11000,11000,11000,#week 03/30 start:135000end:210000
11000,11000,11000,11000,11000,11000,11000#week 04/06 start:210000 no data.
])

testing_south_korea = np.array([
0,0,0,0,0, #week 01/22
0,0,0,0,300,300,300, #week 01/27
300,300,300,300,300,300,300,#week 02/03
800,800,800,800,800,800,800, #week 02/10 start: 2800
3000,3000,3000,3000,3000,3000,3000, # week 02/17 start:8000
11000,11000,11000,11000,11000,11000,11000, #week 02/24 start:28000
12000,12000,12000,12000,12000,12000,12000, #week 03/02 start:105000
9000,9000,9000,9000,9000,9000,9000,  #week 03/09 start:189000
9000,9000,9000,9000,9000,9000,9000, #week 03/16 start:251000
8000,8000,8000,8000,8000,8000,8000,#week 03/23 start:315000
8000,8000,8000,8000,8000,8000,8000,#week 03/30 start:372000
9000,9000,9000,9000,9000,9000,9000#week 04/06 start:437000
])

testing_italy = np.array([
0,0,0,0,0, #week 01/22
0,0,0,0,0,0,0, #week 01/27
0,0,0,0,0,0,0,#week 02/03
0,0,0,0,0,0,0, #week 02/10
600,600,600,600,600,600,600, # week 02/17 end: 4300
2600, 2600,2600,2600,2600,2600,2600,#week 02/24 start:4300, end:23000
4400, 4400,4400,4400,4400,4400,4400,#week 03/02 start:23000, end54000
12000,12000,12000,12000,12000,12000,12000,#week 03/09 start:54000, end 138000
20000,20000,20000,20000,20000,20000,20000,#week 03/16 start:138000,end275000
29000,29000,29000,29000,29000,29000,29000,#week 03/23 start:275000, end477000
35000,35000,35000,35000,35000,35000,35000,#week 03/30 start:477000end721000
35000,35000,35000,35000,35000,35000,35000#week 04/06 start:721000
])

testing_belgium = np.array([
0,0,0,0,0, #week 01/22
0,0,0,0,0,0,0, #week 01/27
0,0,0,0,0,0,0,#week 02/03
0,0,0,0,0,0,0, #week 02/10
0,0,0,0,0,0,0, # week 02/17
0,0,0,0,0,0,0, #week 02/24
500,500,500,500,500,500,500, #week 03/02 start:0
1200,1200,1200,1200,1200,1200,1200, #week 03/09 start:4000
2100,2100,2100,2100,2100,2100,2100, #week 03/16 start:15000
3400,3400,3400,3400,3400,3400,3400,#week 03/23 start:30000
6000,6000,6000,6000,6000,6000,6000,#week 03/30 start:72000
6000,6000,6000,6000,6000,6000,6000#week 04/06 start:437000
])

testing_france = np.array([
0,0,0,0,0, #week 01/22
0,0,0,0,0,0,0, #week 01/27
0,0,0,0,0,0,0,#week 02/03
0,0,0,0,0,0,0, #week 02/10
0,0,0,0,0,0,0, # week 02/17
400,400,400,400,400,400,400, #week 02/24 start:0 end 2900
1300,1300,1300,1300,1300,1300,1300, #week 03/02 start:2900 end:12500
3500,3500,3500,3500,3500,3500,3500, #week 03/09 start:12500end:37000
10000,10000,10000,10000,10000,10000,10000, #week 03/16 start:37000end:107000
17000,17000,17000,17000,17000,17000,17000,#week 03/23 start:107000end:224000
20000,20000,20000,20000,20000,20000,20000,#week 03/30 start:224000 no data: extrap.20000
20000,20000,20000,20000,20000,20000,20000#week 04/06 start:437000
])





# each value contains at least 'name_country', plus other elements needed in parameter_function_generators.py
dict_idiosyncratic_params_dict = {'Spain':{'name_country': 'Spain', 
                                           'population_size_Spain': 50000000,
					   'isolation_1_Spain': datetime.date(2020,3,9),
					   'isolation_2_Spain': datetime.date(2020,3,13),
					   'isolation_3_Spain': datetime.date(2020,3,14),
					   'hospital_capacity_Spain': 1000,                                          
                                           'elderly_population_pct_Spain': 0.19,
                                          },
                                  'Korea, South':{'name_country': 'Korea, South',
                                           'population_size_Korea, South': 50000000,
					   'isolation_1_Korea, South': datetime.date(2020,3,4),
					   'isolation_2_Korea, South': datetime.date(2020,3,4),
					   'isolation_3_Korea, South': datetime.date(2020,3,14),
					   'hospital_capacity_Korea, South': 1000,                                          
                                           'elderly_population_pct_Korea, South': 0.14,
                                           'testing_capacity_Korea, South': testing_south_korea,
                                          },
                                  'Italy':{'name_country': 'Italy',
                                           'population_size_Italy': 60000000,
					   'isolation_1_Italy': datetime.date(2020,3,5),
					   'isolation_2_Italy': datetime.date(2020,3,5),
					   'isolation_3_Italy': datetime.date(2020,3,11),
					   'hospital_capacity_Italy': 1000,
                                           #'elderly_population_pct_Italy': 0.23, 
                                           #'testing_capacity_Italy': testing_italy,
					  },
				  'France':{'name_country': 'France',
                                            'population_size_France': 65000000,
					    'isolation_1_France': datetime.date(2020,3,13),
					    'isolation_2_France': datetime.date(2020,3,14),
					    'isolation_3_France': datetime.date(2020,3,17),
					    'hospital_capacity_France': 2000,
                                            'elderly_population_pct_France': 0.19,
                                            'testing_capacity_France':testing_france, 
					  },
                                  'Belgium':{ 'name_country': 'Belgium',
                                              'population_size_Belgium': 12000000,
                                                   'isolation_1_Belgium': datetime.date(2020,3,12),
                                                   'isolation_2_Belgium': datetime.date(2020,3,14),
                                                   'isolation_3_Belgium': datetime.date(2020,3,18),
                                                   'hospital_capacity_Belgium': 1000,
                                                   'elderly_population_pct_Belgium': 0.19,
                                                   'testing_capacity_Belgium': testing_belgium,
                                          },
                                  'United Kingdom':{'name_country': 'United Kingdom',
                                                    'population_size_United Kingdom': 65000000,
                                                    'isolation_1_United Kingdom': datetime.date(2020,3,16),
                                                    'isolation_2_United Kingdom': datetime.date(2020,3,21),
                                                    'isolation_3_United Kingdom': datetime.date(2020,3,24),
                                                    'hospital_capacity_United Kingdom': 1000,
                                                    'elderly_population_pct_United Kingdom': 0.18,
                                                    'testing_capacity_United Kingdom': testing_united_kingdom, 
                                          }
                                 }


with open('.pkl_idiosyncratic_params.pkl', 'wb') as f:
    pkl.dump(dict_idiosyncratic_params_dict, f)



