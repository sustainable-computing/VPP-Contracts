#!/usr/bin/env python
# coding: utf-8
import traceback
import sys
import numpy as np
import pandas as pd
import yaml
import scipy
import os
import cvxpy as cp
import numpy.random as rand
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
import argparse
import ast

# Saidur's functions
from utility import utility as util
# import latest_deterministic_solver_ev_penetration # Not used
import ev_scheduler 
import generate_expected_EV_values as E_ev_generator
import ev_data_sampler

from real_funcOnlineAlgo import create_ev_dict, EV, get_connected_ev, charge_EV, charge_v2g_EV, discharge_v2g_EV, get_types_and_possible_discharge_e, get_online_alg_result_mc_simul, create_ev_dict_from_df

# # Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-T", "--Tau", help = "Time of V2G contracts", type=int, required=True)
parser.add_argument("-S", "--suffix", help = "Suffix to save the results", type=str, required=True)
parser.add_argument("-Y", "--types", help = "Types of contracts", type=str, required=True)
parser.add_argument("-P", "--perc", help = "percentage of vehicles that participate in v2g", type=int, required=True)
parser.add_argument("-K", "--Kappa", help = "Coefficent for the VPP's utility function", type=float, required=True)
parser.add_argument("-G", "--Gamma", help = "Coefficent for the EV's battery degradation", type=float, required=True)
parser.add_argument("-B", "--bat_deg", help = "Extra info to battery degradation", type=float, required=True)
parser.add_argument("-D", "--res_dir", help = "Results directory", type=str, required=False)
parser.add_argument("-I", "--bid_dir", help = "Bids directory", type=str, required=False)
parser.add_argument("-E", "--seed", help = "Seed for the rng", type=int, required=False)
parser.add_argument("-W", "--skewed", help = "Whether to skew owner types left (-1), uniform (0), or right (1)", type=int, required=False)
args = parser.parse_args()

n_ev = get_types_and_possible_discharge_e(util.load_result(r'data/new_expected_values/E_EV_dict'))

EV_TYPES = ast.literal_eval(args.types)
suffix = args.suffix
TAU =args.Tau
KAPPA = args.Kappa
GAMMA = args.Gamma
BAT_DEG = args.bat_deg
perc_allow_EV_discharge = [args.perc]
if args.res_dir is None:
    res_dir = "contract_no_solar/"
else:
    res_dir = args.res_dir

if args.bid_dir is None:
    bid_dir = "bids_no_solar_2019_few_sample/"
else:
    bid_dir = args.bid_dir

if args.seed is None:
    seed_lst = [790]
else:
    seed_lst = [args.seed]

num_types = len(EV_TYPES)

type_probs = np.ones(num_types) / num_types
if args.skewed is not None:
    assert (num_types == 5), "Skewedness is only supported for 5 owner types"
    if args.skewed == -1: 
        type_probs = [0.36, 0.28, 0.2 , 0.12, 0.04]
    elif args.skewed == 1:
        type_probs = [0.04, 0.12, 0.2 , 0.28, 0.36]

print(f"Running experiment with {EV_TYPES=}, {suffix=}, {TAU=}, {KAPPA=}, {GAMMA=}, {BAT_DEG=}, {perc_allow_EV_discharge=}, {res_dir=}, {bid_dir=}, {type_probs=}")

# Runtime: ~50 mins

sampling_pv_gen_df = pd.read_csv('data/sampling_pv_data.csv', index_col='Date')
sampling_pv_gen_df.index = pd.to_datetime(sampling_pv_gen_df.index)

# Load data from 2019
pv_gen_test_df = pd.read_csv('data/real_data/2019_test_data_pv.csv', index_col='Date')
pv_gen_test_df.index = pd.to_datetime(pv_gen_test_df.index)

sampling_price_df = pd.read_csv('data/sampling_price_data.csv', index_col='Date')
sampling_price_df.index = pd.to_datetime(sampling_price_df.index)

# Load data from 2019
price_test_df = pd.read_csv('data/real_data/2019_test_data_price.csv', index_col='Date')
price_test_df.index = pd.to_datetime(price_test_df.index)

# EV charging sessions
df_ev = pd.read_csv("data/real_data/df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])

unique_dates = pv_gen_test_df.index.unique()
sampling_unique_dates = sampling_pv_gen_df.index.unique()


#seed_lst = [777]
#seed_lst = [778]

#perc_allow_EV_discharge = [0, 25, 50, 75, 100]
#perc_allow_EV_discharge = [25]

num_samples = 100 #100
flag = 1
while(flag == 1):
    try: 
        flag = 0
        baseline_revenue_dict = {}
        online_algo_revenue_dict = {}

        for perc in (perc_allow_EV_discharge):
            
            baseline_revenue_dict = {}
            online_algo_revenue_dict = {}

            for run, seed in enumerate(seed_lst):
                ev_rng  = np.random.default_rng(seed)

                baseline_revenue_lst = []
                
                total_revenue_lst_E_stage_1 = []
                im_buy_revenue_lst_E_stage_1 = []
                im_sell_revenue_lst_E_stage_1 = []
                time_to_full_soc_lst_E_stage_1 = []
                num_v2g_evs_no_contract_lst = []
                num_v2g_evs_w_contract_lst = []
                time_to_full_soc_lst_E_stage_1 = []
                da_revenue_lst_E_stage1 = []
                retail_revenue_lst_E_stage1 = []
                owner_pay_lst_E_stage1 = []
                
                assigned_type_lst_E_stage_1 = []
                realized_type_lst_E_stage_1 = []

                
                total_revenue_lst_R_stage_1 = []
                im_buy_revenue_lst_R_stage_1 = []
                im_sell_revenue_lst_R_stage_1 = []
                time_to_full_soc_lst_R_stage_1 = []
                
                #_, bids = perform_monte_carlo_simul(seed, unique_dates, perc, pv_gen_test_df, price_test_df)
                                
                for day_no, date in enumerate(tqdm(unique_dates, total=len(unique_dates))):
                    current_pv_gen_lst = np.array(list(pv_gen_test_df.loc[(pv_gen_test_df.index == date)]['PV_Vol']))
                    current_da_price_lst = np.array(list(price_test_df.loc[(price_test_df.index == date)]['price_da']))
                    current_im_price_lst = np.array(list(price_test_df.loc[(price_test_df.index == date)]['price_imbalance']))

                    #ev_dict = create_ev_dict(seed+run)
                    ev_dict = create_ev_dict_from_df(df_ev, day_no)
                    
                    # ONLY IF YOU NEED TO RUN THE OFFLINE ALG
                    # JAVIER adds:
                    #pv_gen_lst = current_pv_gen_lst
                    #da_price_lst = current_da_price_lst 
                    #im_price_lst = current_im_price_lst 

                    # end
                    #revenue, _, _ = latest_deterministic_solver_ev_penetration.main(seed+run, pv_gen_lst, da_price_lst, im_price_lst, ev_dict, perc)
                    #baseline_revenue_lst.append(revenue)
                    #  Comment again
                    
                    # Using Monte Carlo simulation results
                    #all_bids_sample_paths = util.load_result(('bids_no_solar/{}_perc'.format(perc)+'/bid_sample_path_'+str(perc)+'_perc'))
                    bid_perc = perc
                    all_bids_sample_paths = util.load_result(('{}{}_perc'.format(bid_dir, bid_perc)+'/bid_sample_path_'+str(bid_perc)+'_perc'))
                    total, im_buy, im_sell, time_to_full_soc, num_v2g_evs_no_contract, num_v2g_evs_w_contract, da_revenue, retail_revenue, owner_pay, assigned_type, realized_type = get_online_alg_result_mc_simul(seed+run, day_no, date, unique_dates, sampling_unique_dates, ev_dict, perc, pv_gen_test_df, price_test_df, sampling_pv_gen_df, sampling_price_df, num_samples, all_bids_sample_paths, EV_TYPES = EV_TYPES, TAU = TAU , KAPPA = KAPPA, GAMMA = GAMMA, BAT_DEG = BAT_DEG, ev_rng = ev_rng, type_probs = type_probs)
                    total_revenue_lst_E_stage_1.append(total)
                    im_buy_revenue_lst_E_stage_1.append(im_buy)
                    im_sell_revenue_lst_E_stage_1.append(im_sell)
                    time_to_full_soc_lst_E_stage_1.append(time_to_full_soc)
                    da_revenue_lst_E_stage1.append(da_revenue)
                    retail_revenue_lst_E_stage1.append(retail_revenue)
                    owner_pay_lst_E_stage1.append(owner_pay)

                    assigned_type_lst_E_stage_1.append(assigned_type)
                    realized_type_lst_E_stage_1.append(realized_type)

                    num_v2g_evs_no_contract_lst.append(num_v2g_evs_no_contract)
                    num_v2g_evs_w_contract_lst.append(num_v2g_evs_w_contract)
                        
                    #print('done')
                                                          
                    # Using accurate predictions
        #             total, im_buy, im_sell, time_to_full_soc = get_online_alg_result_acc_preds(seed+run, day_no, date, ev_dict, perc, current_pv_gen_lst, current_da_price_lst, current_im_price_lst, False)
        #             total_revenue_lst_R_stage_1.append(total)
        #             im_buy_revenue_lst_R_stage_1.append(im_buy)
        #             im_sell_revenue_lst_R_stage_1.append(im_sell)
        #             time_to_full_soc_lst_R_stage_1.append(time_to_full_soc)

                    seed += 10000
                
                baseline_revenue_dict[str(perc)+'_run'+str(run)] = baseline_revenue_lst
                online_algo_revenue_dict['total_revenue_'+str(perc)+'_run'+str(run)+'_E'] = total_revenue_lst_E_stage_1
                online_algo_revenue_dict['im_buy_'+str(perc)+'_run'+str(run)+'_E'] = im_buy_revenue_lst_E_stage_1
                online_algo_revenue_dict['im_sell_'+str(perc)+'_run'+str(run)+'_E'] = im_sell_revenue_lst_E_stage_1
                online_algo_revenue_dict['time_to_full_soc_'+str(perc)+'_run'+str(run)+'_E'] = time_to_full_soc_lst_E_stage_1
                online_algo_revenue_dict['num_v2g_evs_no_contract_lst_'+str(perc)+'_run'+str(run)+'_E'] = num_v2g_evs_no_contract_lst
                online_algo_revenue_dict['num_v2g_evs_w_contract_lst_'+str(perc)+'_run'+str(run)+'_E'] = num_v2g_evs_w_contract_lst
                
                online_algo_revenue_dict["da_revenue"+str(perc)+"_run"+str(run)+"_E"] = da_revenue_lst_E_stage1
                online_algo_revenue_dict["retail_revenue"+str(perc)+"_run"+str(run)+"_E"] = retail_revenue_lst_E_stage1
                online_algo_revenue_dict["owner_pay"+str(perc)+"_run"+str(run)+"_E"] = owner_pay_lst_E_stage1

                online_algo_revenue_dict["assigned_type"+str(perc)+"_run"+str(run)+"_E"] = assigned_type_lst_E_stage_1
                online_algo_revenue_dict["realized_type"+str(perc)+"_run"+str(run)+"_E"] = realized_type_lst_E_stage_1

                online_algo_revenue_dict['total_revenue_'+str(perc)+'_run'+str(run)+'_R'] = total_revenue_lst_R_stage_1
                online_algo_revenue_dict['im_buy_'+str(perc)+'_run'+str(run)+'_R'] = im_buy_revenue_lst_R_stage_1
                online_algo_revenue_dict['im_sell_'+str(perc)+'_run'+str(run)+'_R'] = im_sell_revenue_lst_R_stage_1
                online_algo_revenue_dict['time_to_full_soc_'+str(perc)+'_run'+str(run)+'_R'] = time_to_full_soc_lst_R_stage_1

            #util.save_result('w_mc_results/baseline_revenue_'+str(perc),baseline_revenue_dict)
            util.save_result(f'w_mc_results/{res_dir}tau_1_contract_online_algo_revenue_original_'+str(perc)+str(suffix),online_algo_revenue_dict)
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(f"Experiment crashed {EV_TYPES=}, {suffix=}, {TAU=}, {KAPPA=}, {GAMMA=}, {BAT_DEG=}, {perc_allow_EV_discharge=}, trying different seed")
        flag = 1
        seed_lst[0] += 7
