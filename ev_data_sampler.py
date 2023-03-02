import pandas as pd
import numpy as np
from scipy.stats import truncnorm, rv_discrete
import yaml
import math
from scipy.optimize import curve_fit

from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.distributions.mixture_rvs import mixture_rvs

def read_yaml(file):
    with open(file, 'r') as yml_file:
        config = yaml.load(yml_file, Loader=yaml.FullLoader)
        return config
		
		
#hour_dict = read_yaml(r'J:\Thesis_code\thesis_code_saidur\Thesis_Code_LATEST_ONLY\EV\num_ev_arrival_by_hour.yaml')
hour_dict = read_yaml(r'num_ev_arrival_by_hour.yaml')

def poisson_dist_func(lam, k):
    return (pow(lam,k) * np.exp(-1*lam))/math.factorial(k)

def sample_num_EV_arrivals(rng, hour):

    hour_set=set(hour_dict[str(hour)])
    d={h:hour_dict[str(hour)].count(h) for h in hour_set}

    x_values = list(d.keys())
    y_values = list(d.values())/np.sum(list(d.values()))
            
    lam = np.sum(np.array(list(x_values)) * np.array(list(y_values))/np.sum(np.array(list(y_values))))
    
    pdf = []
    for k in x_values:
        pdf.append(poisson_dist_func(lam, k))
        
    probab = np.array(pdf)
    probab /= probab.sum()
    
    num_arrived_EVs = rng.choice(x_values, 1, p = probab)
      
    return num_arrived_EVs[0]  


#stay_time_dict = read_yaml(r'J:\Thesis_code\thesis_code_saidur\Thesis_Code_LATEST_ONLY\EV\ev_stay_t_by_arrival_hour.yaml')
stay_time_dict = read_yaml(r'ev_stay_t_by_arrival_hour.yaml')


#============ This function samples the EV stay time using Kernel Density Estimation ============ 

def sample_ev_stay_time(rng, hour):

    kde = sm.nonparametric.KDEUnivariate(stay_time_dict[str(hour)])
    kde.fit()

    x_vals = kde.support
    pdf = kde.density/np.sum(kde.density)

    ev_stay_time = (int(np.around(np.abs(rng.choice(x_vals, 1, p = pdf)))[0]))

    return ev_stay_time 
    #return ev_stay_time, kde.support, kde.density # For plottin
	
	
#data_config = read_yaml(r'J:\Thesis_code\thesis_code_saidur\Thesis_Code_LATEST_ONLY\config\data_config.yml')
data_config = read_yaml(r'config/data_config.yml')

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def sample_init_soc(seed):
    np.random.seed(seed)
    battery_specs = data_config['sessions_processing']['battery_start']
    init_soc = get_truncated_normal(battery_specs['mean'], battery_specs['scale'],
                                                        battery_specs['lower_bound'],
                                                        battery_specs['upper_bound']).rvs()

    return init_soc