import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#!!!!!!!!!!!!  experimental losses.


#def loss_real_world(simulation, ground_truth, loss_function_base):
#   # Loss for real-world dataset. Matching 2 time-series: number of diagnosed cases and number of fatal cases.
#    assert (len(simulation) == 5) and (len(ground_truth) == 2)
#    return lambda simulation, ground_truth: loss_function_base(np.sum(simulation), ground_truth[0]) + loss_function_base(simulation[3], ground_truth[1])



def loss_real_world(simulation, ground_truth):
    loss_function_base = lambda y0, y1 : np.abs(y0-y1)/np.sqrt(y1+1)
    
    return loss_function_base(np.sum(simulation), ground_truth[0]) + loss_function_base(simulation[2], ground_truth[1])








def format_country_ground_truth(ground_truth_df, country_name):
    country_df = ground_truth_df.get_group(country_name).copy()
    country_df.drop(['province_state', 'country_region', 'latitude', 'longitude', 'location_geom', 'type'], 
                    axis = 1, 
                    inplace=True)
    output = country_df.to_numpy()
    if output[0,-1] < output[1, -1]:
        print('Pandas df has inverted Diagnosed/Fatality.')
        output[[0,1]] = output[[1,0]]
    return output

def plot_country_ground_truth(ground_truth_df, country_name = None, previous_figure = None):
    if type(ground_truth_df) == pd.core.groupby.generic.DataFrameGroupBy:
        output = format_country_ground_truth(ground_truth_df, country_name)
    else:
        assert type(ground_truth_df) == np.ndarray
        output = ground_truth_df
    if previous_figure is not None:
        fig, ax = previous_figure
    else:
        fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(np.arange(output.shape[1]),output[0,:], label = country_name+'_diagnosed')
    ax.plot(np.arange(output.shape[1]),output[1,:], label = country_name+'_fatality')
    return fig, ax


def plot_list_country_ground_truth(ground_truth_df, country_list):

    fig, ax = plt.subplots(figsize=(20, 10))
    for country in country_list:
        fig, ax = plot_country_ground_truth(ground_truth_df, country, (fig,ax))

    return fig, ax





