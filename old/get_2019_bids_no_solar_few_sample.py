#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from utility import utility as util
from utility import utils as utils
import yaml

#from tqdm.notebook import tqdm_notebook
import copy
import math
import os
import numpy.random as rand
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import latest_deterministic_solver_ev_penetration_no_solar
import generate_expected_EV_values as E_ev_generator

import ev_data_sampler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-P", "--perc", help = "percentage of vehicles that participate in v2g", type=int, required=True)
args = parser.parse_args()
perc_allow_EV_discharge = [args.perc]


# In[10]:
def create_ev_dict_from_df(df_ev, day): # day int: {0 - 364}
    FINAL_SOC = 0.97
    ALPHA_C = 11
    B_CAP = 80
    ETA_C = 0.98
    df_ev_day = df_ev[df_ev["day_no"] == day]
    
    # Init dict
    ev_dict = {}
    ev_outer_keys = ['init_soc','ev_stay_t','ev_laxity']
    for key in ev_outer_keys:
        ev_dict[key] = {}
        for h in range(24):
            ev_dict[key]["hour_{}".format(str(h))] = []

    for hour in range(24):
        df_ev_hour = df_ev_day[df_ev_day["start_hour"] == hour]    
        for _, row in df_ev_hour.iterrows():
            stay_t = row.new_connected_time
            init_soc = row.initial_soc
            depart_time = hour + stay_t
            
            if(depart_time > 23):
                depart_time = 23
                stay_t = depart_time - hour
            laxity =  stay_t - (FINAL_SOC - init_soc)*B_CAP/(ALPHA_C * ETA_C)
            if(laxity >= 0):
                ev_dict['ev_stay_t']['hour_{}'.format(hour)].append(stay_t)
                ev_dict['init_soc']['hour_{}'.format(hour)].append(init_soc)
                ev_dict['ev_laxity']['hour_{}'.format(hour)].append(laxity)              

    return ev_dict

def create_ev_dict(seed):
    raise Exception("Using deprecated create de dict")
    
    rng = np.random.default_rng(seed)

    
    DAY_HRS = 24
    FINAL_SOC = 0.97
    ALPHA_C = 11
    B_CAP = 80
    ETA_C = 0.98

    ev_dict = {}
    ev_outer_keys = ['init_soc','ev_stay_t','ev_laxity']

    for key in ev_outer_keys:
        ev_dict[key] = {}

    for key in ev_outer_keys:
        for h in range(DAY_HRS):
            ev_dict[key]['hour_{}'.format(str(h))] = []
                 
    for hour in range(DAY_HRS):
        num_arrived_ev = ev_data_sampler.sample_num_EV_arrivals(rng, hour)
        
#         print('in origg ', num_arrived_ev)
        
        for _ in range(num_arrived_ev):
            stay_t = ev_data_sampler.sample_ev_stay_time(rng, hour)
            init_soc = ev_data_sampler.sample_init_soc(seed)
            seed += 1
            depart_time = hour + stay_t
            if(depart_time > 23):
                depart_time = 23
                stay_t = depart_time - hour
            laxity = stay_t - (FINAL_SOC - init_soc)*B_CAP/(ALPHA_C * ETA_C)
            if(laxity >= 0):
                ev_dict['ev_stay_t']['hour_{}'.format(hour)].append(stay_t)
                ev_dict['init_soc']['hour_{}'.format(hour)].append(init_soc)
                ev_dict['ev_laxity']['hour_{}'.format(hour)].append(laxity)
                
    return ev_dict


# In[11]:


def create_forecast_ev_dict(seed, sample_no):
    
    rng = np.random.default_rng(seed)
    ev_rng = np.random.default_rng(seed+sample_no)

    
    DAY_HRS = 24
    FINAL_SOC = 0.97
    ALPHA_C = 11
    B_CAP = 80
    ETA_C = 0.98
    NOISE_STD = 0.1

    ev_dict = {}
    ev_outer_keys = ['init_soc','ev_stay_t','ev_laxity']

    for key in ev_outer_keys:
        ev_dict[key] = {}

    for key in ev_outer_keys:
        for h in range(DAY_HRS):
            ev_dict[key]['hour_{}'.format(str(h))] = []
                 
    for hour in range(DAY_HRS):
        num_arrived_ev = ev_data_sampler.sample_num_EV_arrivals(rng, hour)
                
        num_arrived_ev_noise = ev_rng.normal(0, (num_arrived_ev*NOISE_STD))
        
        num_arrived_ev += num_arrived_ev_noise
        
#         if(num_arrived_ev_noise > 0):
#             num_arrived_ev = np.ceil(num_arrived_ev)
#         if(num_arrived_ev > 0.5 and num_arrived_ev <= 1):
#             num_arrived_ev = 1
        
        num_arrived_ev = int(num_arrived_ev)
    
        
        if(num_arrived_ev <= 0):
            num_arrived_ev = 0
        
        for _ in range(num_arrived_ev):
            stay_t = ev_data_sampler.sample_ev_stay_time(rng, hour)
            stay_t_noise = ev_rng.normal(0, (stay_t*NOISE_STD))
            stay_t += stay_t_noise
            if(stay_t_noise > 0):
                stay_t = np.ceil(stay_t)
            stay_t = int(stay_t)

            
            if(stay_t >= (DAY_HRS-1)):
                stay_t = DAY_HRS-1
            elif(stay_t < 0):
                stay_t = 0
            
            init_soc = ev_data_sampler.sample_init_soc(seed)
            init_soc_noise = ev_rng.normal(0, (init_soc*NOISE_STD))
            init_soc += init_soc_noise
            seed += 1

            depart_time = hour + stay_t
            if(depart_time > 23):
                depart_time = 23
                stay_t = depart_time - hour
                
            if(init_soc >= FINAL_SOC):
                init_soc = FINAL_SOC
            elif(init_soc < 0):
                init_soc = 0
                                
            laxity = stay_t - (((FINAL_SOC - init_soc)*B_CAP)/(ALPHA_C * ETA_C))
            if(laxity >= 0):
                ev_dict['ev_stay_t']['hour_{}'.format(hour)].append(stay_t)
                ev_dict['init_soc']['hour_{}'.format(hour)].append(init_soc)
                ev_dict['ev_laxity']['hour_{}'.format(hour)].append(laxity)
    
    return ev_dict


# In[12]:


class EV:
    
    def __init__(self, arrival_time, stay_time, soc_init, laxity):
        self.arrival_time = arrival_time
        self.stay_time = stay_time
        self.departure_time = self.arrival_time + self.stay_time
        self.soc_final = 0.97
        self.soc_init = soc_init
        self.soc_t = soc_init
        self.laxity = laxity
        self.battery_cap = 80 #kW
        self.alpha_c = 11 #kW
        self.eta_c = 0.98
        self.alpha_d = -11 #kW
        self.eta_d = 0.98
        self.allow_discharge = False
        self.priority_charge = False
        self.bool_c_d = False
        self.completed = False
        self.time_to_full_soc = 0
        self.incentive_valuation = 0
        self.actual_payback = 0
        self.discharge_threshold = 1.25


# In[13]:


def get_connected_ev(ev_dict, hour): 
    
    just_arrived_ev_lst = []

    if(len(ev_dict['init_soc']['hour_{}'.format(str(hour))]) != 0):
        for ev_num in range(len(ev_dict['init_soc']['hour_{}'.format(str(hour))])):
            just_arrived_ev_lst.append(EV(hour,
                                         ev_dict['ev_stay_t']['hour_{}'.format(str(hour))][ev_num],
                                         ev_dict['init_soc']['hour_{}'.format(str(hour))][ev_num],
                                          ev_dict['ev_laxity']['hour_{}'.format(str(hour))][ev_num]
                                         ))
        return just_arrived_ev_lst
    else:
        return [-1]


# In[14]:


def surplus_pv_gen(current_time, pv_diff, available_ev_lst, im_price_lst):

    imbalance_sell = 0
    e_extra = pv_diff
    
    charge_e_dict = {}

    for ev in available_ev_lst:
        if(ev.bool_c_d == False and ev.soc_t < ev.soc_final):
            charge_e_dict[ev] = ev.laxity


    # sort based on ascending order of Laxity. Smaller lax => more urgent to charge!
    if(len(charge_e_dict) != 0):
        charge_e_dict = sorted(charge_e_dict.items(), key=lambda x: x[1], reverse=False)

        for ev in charge_e_dict:
            if(e_extra > 0):
                e_charging = charge_EV(ev[0], e_extra)
                e_extra -= e_charging
#                 if(im_price_lst[current_time] > 0):
#                     imbalance_sell += e_charging * im_price_lst[current_time]
    
    # Sell remaining surplus energy to imbalance market after EV charging complete!
    if(e_extra > 0 and im_price_lst[current_time] > 0):
        imbalance_sell += e_extra  * im_price_lst[current_time]
    return imbalance_sell


# In[15]:


def charge_EV(ev, e_charging):
    
    ETA_C = 0.98
    
    e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
    
    if(e_charging >= e_required):
        e_charging = e_required
        e_required = 0

    if(e_charging >= ev.alpha_c):
        e_charging = ev.alpha_c
    
    ev.soc_t =  ev.soc_t + (e_charging*ETA_C)/ev.battery_cap
    ev.stay_time -= 1
    ev.laxity = ev.stay_time - ((ev.soc_final - ev.soc_t) * ev.battery_cap)/(ev.alpha_c * ETA_C)
    if(ev.laxity == 0):
        ev.laxity = 0
    ev.bool_c_d = True
    
    return e_charging


# In[16]:


def shortage_pv_gen(current_time, pv_diff, available_ev_lst, da_price_lst, im_price_lst):
    
    imbalance_buy = 0
    DISCHRG_THRESH = 0
    
    e_short = np.abs(pv_diff)

    discharge_e_dict = {}
    sum_disch_E = 0
    
    #print("BEFOREEE DISCHRGG E-short ", e_short)
    
    # Obtain the EVs that have POSITIVE LAXITY and ALLOW DISCHARGE
    for ev in available_ev_lst:
        if(ev.laxity > DISCHRG_THRESH and ev.allow_discharge == True and ev.bool_c_d == False):
            # storing the EV obj address in the dict as "key" and the available discharge energy as the "item"
            discharge_e_dict[ev] = get_available_discharge_energy(ev)
    #print('discharge_e_dict ', discharge_e_dict)
    if(len(discharge_e_dict) != 0):
        # Descending sort
        discharge_e_dict = sorted(discharge_e_dict.items(), key=lambda x: x[1], reverse=True)

        for ev in discharge_e_dict:
            sum_disch_E += ev[1]
            
        e_short_orig = e_short
        for ev in discharge_e_dict:
            if (e_short > 0 and ev[1] != 0):
                if(sum_disch_E > e_short):
                    e_discharged = discharge_EV(ev[0], (ev[1]/sum_disch_E)*e_short_orig, e_short)
                    #ev[0].dsch_revenue_contrib += e_discharged * im_price_lst[current_time]
                else:    
                    e_discharged = discharge_EV(ev[0], ev[1], e_short)    
                e_short -= e_discharged
                #print('EEE_DISCCC ', e_discharged)
                #revenue_from_ev_dischrg += e_discharged * im_price_lst[current_time]
        
        #print('SSORRRTT ',e_short)
    return e_short


def get_available_discharge_energy(ev):
    
    e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
    look_ahead_e_required = ev.alpha_c * ev.eta_c * (ev.stay_time - 1)
    e_discharge = look_ahead_e_required - e_required
    
    if(e_discharge < 0):
        e_discharge = 0
        
    elif(e_discharge >= np.abs(ev.alpha_d)):
        e_discharge = np.abs(ev.alpha_d)
    
    return e_discharge


# In[17]:


def discharge_EV(ev, e_discharge_available, e_short):
    
    ETA_D = 0.98
    
    if(e_discharge_available > np.abs(ev.alpha_d)):
        e_discharge_available = np.abs(ev.alpha_d)
    elif(e_discharge_available <= 0):
        e_discharge_available = 0
    
#     if(e_discharge_available != 0):       
#         if(e_short <= e_discharge_available):
#             e_discharge_available = e_short
#             e_short = 0
#         else:
#             e_short -= e_discharge_available

        ev.soc_t = ev.soc_t - (e_discharge_available/ETA_D)/ev.battery_cap
        ev.stay_time -= 1
        ev.laxity = ev.stay_time - ((ev.soc_final - ev.soc_t) * ev.battery_cap)/(ev.alpha_c * ETA_C)
        
        if(ev.laxity == 0):
            ev.laxity = 0
        ev.bool_c_d = True
        
    return e_discharge_available


# In[18]:


def update_ev_dict_allowed_discharge_evs(seed, ev_dict, ratio):
        
    rng = np.random.default_rng(seed)
    
    total_evs = 0
    for h in range(24):
        total_evs += len(ev_dict['init_soc']['hour_{}'.format(str(h))])
    
    ev_allowed_discharge = rng.choice(total_evs, size=round(total_evs*(ratio/100)), replace=False)
    
    ev_dict['ev_allowed_discharge'] = ev_allowed_discharge
    return ev_dict


# In[20]:


def get_online_alg_result_mc_simul(seed, day_no, current_date, unique_dates, sampling_unique_dates, orig_ev_dict, ratio_EV_discharge, pv_gen_df, price_df, sampling_pv_gen_df, sampling_price_df, num_samples):
       
    all_bids_sample_paths = []
    bids_sample_paths = []
              
    DAY_HRS = 24
    NOISE_STD = 0.1
    FINAL_SOC = 0.97
    B_CAP = 80
    ALPHA_C = 11
    ETA_C = 0.98
    
    my_rng = np.random.default_rng(seed)
        
    # actual realizations of the date
    pv_gen_lst = np.array(list(pv_gen_df.loc[(pv_gen_df.index == current_date)]['PV_Vol']))
    da_price_lst = np.array(list(price_df.loc[(price_df.index == current_date)]['price_da']))
    im_price_lst = np.array(list(price_df.loc[(price_df.index == current_date)]['price_imbalance']))

    # sampling the white gaussian noise to add
    pv_forecast_noise = my_rng.normal(0, (pv_gen_lst*NOISE_STD), (num_samples,DAY_HRS))
    da_price_forecast_noise = my_rng.normal(0, np.abs(da_price_lst)*NOISE_STD, (num_samples,DAY_HRS))  
    im_price_forecast_noise = my_rng.normal(0, np.abs(im_price_lst)*NOISE_STD, (num_samples,DAY_HRS))
       
        
    for sample in (range(num_samples)):
        
        forecast_ev_dict = create_forecast_ev_dict(seed, sample)
        
        pv_gen_forecast = pv_gen_lst + pv_forecast_noise[sample]
                
        da_price_forecast = da_price_lst + da_price_forecast_noise[sample]
        im_price_forecast = im_price_lst + im_price_forecast_noise[sample]        
       
        for idx, p in enumerate(im_price_forecast):
            if p < 0:
                im_price_forecast[idx] = 0
                                                        
                                                            
        for idx, p in enumerate(da_price_forecast):
            if p < 0:
                da_price_forecast[idx] = 0

        _, bids, _ = latest_deterministic_solver_ev_penetration_no_solar.main(seed, pv_gen_forecast, da_price_forecast, im_price_forecast, forecast_ev_dict, ratio_EV_discharge)
        bids_sample_paths.append(bids)
    
    
    # Javier: Change dirs
    #if(len(os.listdir('bids_snowball_no_solar/{}_perc'.format(ratio_EV_discharge))) >= 1):
    #    all_bids_sample_paths = util.load_result(('bids_snowball_no_solar/{}_perc'.format(ratio_EV_discharge)+'/bid_sample_path_'+str(ratio_EV_discharge)+'_perc'))
    if(len(os.listdir('bids_no_solar_2019_few_sample/{}_perc'.format(ratio_EV_discharge))) >= 1):
        all_bids_sample_paths = util.load_result(('bids_no_solar_2019_few_sample/{}_perc'.format(ratio_EV_discharge)+'/bid_sample_path_'+str(ratio_EV_discharge)+'_perc'))
            
    all_bids_sample_paths.append(bids_sample_paths)
    util.save_result('bids_no_solar_2019_few_sample/{}_perc'.format(ratio_EV_discharge)+'/bid_sample_path_'+str(ratio_EV_discharge)+'_perc', all_bids_sample_paths)
     
        
# #     # +++++++++++++++++++++++++++ For each of the sample paths' bid, solving STAGE - 2 +++++++++++++++++++++++++++++++++++   
#     revenue_sample_paths = []
#     im_buy_sample_paths = []
#     im_sell_sample_paths = []
#     time_to_soc_sample_paths = []

#     for b in (bids_sample_paths):

#         bids = b 

#         da_revenue_lst =  np.zeros(DAY_HRS)
#         imbalance_buy_lst = np.zeros(DAY_HRS)
#         imbalance_sell_lst = np.zeros(DAY_HRS)

#         revenue_lst = np.zeros(DAY_HRS)

#         #_, bids = perform_monte_carlo_simul(seed, date, unique_dates, ratio_EV_discharge, pv_gen_df, price_df)
#         ev_dict = update_ev_dict_allowed_discharge_evs(seed, orig_ev_dict, ratio_EV_discharge)

#         available_ev_lst = []
#         ev_history_lst = []
#         avg_time_to_full_soc = []

#         pv_gen_lst = np.array(list(pv_gen_df.loc[(pv_gen_df.index == current_date)]['PV_Vol']))
#         da_price_lst = np.array(list(price_df.loc[(price_df.index == current_date)]['price_da']))
#         im_price_lst = np.array(list(price_df.loc[(price_df.index == current_date)]['price_imbalance']))

#         for current_time in range(DAY_HRS):

#             revenue = 0
#             da_revenue = 0
#             im_buy_revenue = 0
#             im_sell_revenue = 0

#             # Get the EVs
#             found_evs_lst = get_connected_ev(ev_dict, current_time)

#             # Form available_ev_lst
#             if(found_evs_lst[0] != -1):
#                 for found_ev in found_evs_lst:
#                     ev_history_lst.append(found_ev)
#                     if(ev_history_lst.index(found_ev) in ev_dict['ev_allowed_discharge']):
#                         found_ev.allow_discharge = True
#                     else:
#                         found_ev.allow_discharge = False    
#                     available_ev_lst.append(found_ev)

#             # Find EVs with Low laxity
#             ac_lst = []
#             for ev in available_ev_lst:
#                 ev.bool_c_d = False #Reset for all EVs on every hour
#                 if(current_time >= ev.departure_time or ev.stay_time == 0):
#                     ev.bool_c_d = True
#                 if(ev.bool_c_d == False and ev.completed == False and (math.floor(ev.laxity)) <= 0 and (ev.soc_t != ev.soc_final)):
#                     ac_lst.append(charge_EV(ev, ev.alpha_c))

#             # Total charging demand
#             x_t = np.array(bids[current_time]) + np.sum(np.array(ac_lst))

#             bid_amt = np.array(bids[current_time])
#             total_chrg_demand =  np.sum(np.array(ac_lst))
#             da_revenue += bid_amt * da_price_lst[current_time]

#             # Get the surplus/deficit   
#             if(pv_gen_lst[current_time] >= x_t):
#                 # Surplus
#                 pv_diff = pv_gen_lst[current_time] - x_t
#                 im_sell_revenue += surplus_pv_gen(current_time, pv_diff, available_ev_lst, im_price_lst)


#             elif (x_t > pv_gen_lst[current_time]):
#                 # Deficit
#                 pv_diff = pv_gen_lst[current_time] - x_t
#                 e_short = shortage_pv_gen(current_time, pv_diff, available_ev_lst, da_price_lst, im_price_lst)
#                 if(e_short > 0):
#                     im_buy_revenue += -1 * e_short * im_price_lst[current_time]
#                 e_short = 0

#             revenue = da_revenue + im_sell_revenue + im_buy_revenue

#             revenue_lst[current_time] = revenue
#             da_revenue_lst[current_time] = da_revenue
#             imbalance_buy_lst[current_time] = im_buy_revenue
#             imbalance_sell_lst[current_time] = im_sell_revenue


#             for ev in available_ev_lst:                   
#                 if(ev.bool_c_d == False):
#                     # For EVs not charged or discharged, ONLY Laxity update
#                     e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
#                     ev.stay_time -= 1 
#                     ev.laxity = ev.stay_time - (e_required/(ev.alpha_c * ev.eta_c))    


#             for ev in available_ev_lst: 
#                 if(current_time == ev.departure_time or ev.stay_time == 0):
#                     if(ev.completed == False):
#                         avg_time_to_full_soc.append(1+(current_time - ev.arrival_time))
#                     ev.completed = True
#                     ev.bool_c_d = True
        
#         revenue_sample_paths.append(np.sum(revenue_lst))
#         im_buy_sample_paths.append(np.sum(imbalance_buy_lst))
#         im_sell_sample_paths.append(np.sum(imbalance_sell_lst))
#         time_to_soc_sample_paths.append(np.mean(avg_time_to_full_soc,axis=0))
        
    return  1, 1, 1, 1   
    #return np.average(revenue_sample_paths), np.average(im_buy_sample_paths), np.average(im_sell_sample_paths), np.average(time_to_soc_sample_paths)


# In[ ]:


sampling_pv_gen_df = pd.read_csv('sampling_pv_data.csv', index_col='Date') # Not used
sampling_pv_gen_df.index = pd.to_datetime(sampling_pv_gen_df.index)

# Javier chages to 2019
pv_gen_test_df = pd.read_csv('real_data/2019_test_data_pv.csv', index_col='Date')
pv_gen_test_df.index = pd.to_datetime(pv_gen_test_df.index)

sampling_price_df = pd.read_csv('sampling_price_data.csv', index_col='Date') # Not used
sampling_price_df.index = pd.to_datetime(sampling_price_df.index)

# Javier changes to 2019
price_test_df = pd.read_csv('real_data/2019_test_data_price.csv', index_col='Date')
price_test_df.index = pd.to_datetime(price_test_df.index)

# Javier df_ev
df_ev = pd.read_csv("real_data/df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])

unique_dates = pv_gen_test_df.index.unique()
sampling_unique_dates = sampling_pv_gen_df.index.unique()

baseline_revenue_dict = {}
online_algo_revenue_dict = {}

seed_lst = [777]
#perc_allow_EV_discharge = [100] # Was 75 before. Does it affect?


num_samples = 10 # Should we have 10^3?

for perc in (perc_allow_EV_discharge):
    
    baseline_revenue_dict = {}
    online_algo_revenue_dict = {}

    for run, seed in enumerate(seed_lst):
        baseline_revenue_lst = []
        
        total_revenue_lst_E_stage_1 = []
        im_buy_revenue_lst_E_stage_1 = []
        im_sell_revenue_lst_E_stage_1 = []
        time_to_full_soc_lst_E_stage_1 = []
        
        total_revenue_lst_R_stage_1 = []
        im_buy_revenue_lst_R_stage_1 = []
        im_sell_revenue_lst_R_stage_1 = []
        time_to_full_soc_lst_R_stage_1 = []
        
        #_, bids = perform_monte_carlo_simul(seed, unique_dates, perc, pv_gen_test_df, price_test_df)
                        
        for day_no, date in enumerate(tqdm(unique_dates, total=len(unique_dates))):

            current_pv_gen_lst = np.array(list(pv_gen_test_df.loc[(pv_gen_test_df.index == date)]['PV_Vol']))
            current_da_price_lst = np.array(list(price_test_df.loc[(price_test_df.index == date)]['price_da']))
            current_im_price_lst = (np.array(list(price_test_df.loc[(price_test_df.index == date)]['price_imbalance'])))

            #ev_dict = create_ev_dict(seed+run)
            ev_dict = create_ev_dict_from_df(df_ev, day_no)

            # Using Monte Carlo simulation results
            total, im_buy, im_sell, time_to_full_soc = get_online_alg_result_mc_simul(seed+run, day_no, date, unique_dates, sampling_unique_dates, ev_dict, perc, pv_gen_test_df, price_test_df, sampling_pv_gen_df, sampling_price_df, num_samples)
            total_revenue_lst_E_stage_1.append(total)
            im_buy_revenue_lst_E_stage_1.append(im_buy)
            im_sell_revenue_lst_E_stage_1.append(im_sell)
            time_to_full_soc_lst_E_stage_1.append(time_to_full_soc)
            
            seed += 10000
        
        baseline_revenue_dict[str(perc)+'_run'+str(run)] = baseline_revenue_lst
        online_algo_revenue_dict['total_revenue_'+str(perc)+'_run'+str(run)+'_E'] = total_revenue_lst_E_stage_1
        online_algo_revenue_dict['im_buy_'+str(perc)+'_run'+str(run)+'_E'] = im_buy_revenue_lst_E_stage_1
        online_algo_revenue_dict['im_sell_'+str(perc)+'_run'+str(run)+'_E'] = im_sell_revenue_lst_E_stage_1
        online_algo_revenue_dict['time_to_full_soc_'+str(perc)+'_run'+str(run)+'_E'] = time_to_full_soc_lst_E_stage_1

    #util.save_result('w_mc_results/baseline_revenue_'+str(perc),baseline_revenue_dict)
    #util.save_result('w_mc_results/1K/expected_online_algo_revenue_original_'+str(perc),online_algo_revenue_dict)


