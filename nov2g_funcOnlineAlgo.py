#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import scipy # type: ignore
import os
import cvxpy as cp # type: ignore
import numpy.random as rand
from scipy.stats import truncnorm # type: ignore
import sys
from utility import utility as util
import nov2g_ev_scheduler_v3 as new_ev_scheduler
import ev_data_sampler
from Contract_Reproduction.contracts import get_contract_customtypes

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
            stay_t = int(row.new_connected_time)
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

    FINAL_SOC = 0.97
    ALPHA_C = 11
    B_CAP = 80
    ETA_C = 0.98

    ev_dict = {}
    ev_outer_keys = ['init_soc','ev_stay_t','ev_laxity']

    for key in ev_outer_keys:
        ev_dict[key] = {}

    for key in ev_outer_keys:
        for h in range(24):
            ev_dict[key]['hour_{}'.format(str(h))] = []
                 
    for hour in range(24):
        num_arrived_ev = ev_data_sampler.sample_num_EV_arrivals(rng, hour)

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

class EV:
    def __init__(self, arrival_time, stay_time, soc_init, laxity, feasible_discharge_e):
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
        self.bool_c_d = False
        self.completed = False
        self.time_to_full_soc = 0
        self.remanining_contract_duration = 0
        self.remanining_discharge_e = 0
        self.feasible_discharge_e = feasible_discharge_e
        self.scheduled_profile = 0
        self.new_v2g_val = -1000

def get_connected_ev(ev_dict, hour): 
    ALPHA_C = 11
    ALPHA_D = 11
    
    ETA_C = 0.98
    ETA_D = 0.98
    
    # SHAI == PSI
    
    SHAI = (ALPHA_C * ETA_C * ETA_D)/(ALPHA_D + ALPHA_C * ETA_C * ETA_D)
    
    just_arrived_ev_lst = []
    ALPHA_D = 11
    ETA_D = 0.98

    if(len(ev_dict['init_soc']['hour_{}'.format(str(hour))]) != 0):
        for ev_num in range(len(ev_dict['init_soc']['hour_{}'.format(str(hour))])):
            just_arrived_ev_lst.append(EV(hour,
                                         ev_dict['ev_stay_t']['hour_{}'.format(str(hour))][ev_num],
                                         ev_dict['init_soc']['hour_{}'.format(str(hour))][ev_num],
                                          ev_dict['ev_laxity']['hour_{}'.format(str(hour))][ev_num],
                                          (ev_dict['ev_laxity']['hour_{}'.format(str(hour))][ev_num])*(SHAI*ALPHA_D))
                                      )
            
        return just_arrived_ev_lst
    else:
        return [-1]

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
    if(ev.remanining_contract_duration > 0):
        ev.remanining_contract_duration -= 1
    ev.laxity = ev.stay_time - ((ev.soc_final - ev.soc_t) * ev.battery_cap)/(ev.alpha_c * ETA_C)
    if(ev.laxity == 0):
        ev.laxity = 0
    ev.bool_c_d = True
    
    return e_charging

def charge_v2g_EV(ev, e_charging):
    ETA_C = 0.98
        
    ev.soc_t =  ev.soc_t + (e_charging*ETA_C)/ev.battery_cap
    ev.stay_time -= 1
    if(ev.remanining_contract_duration > 0):
        ev.remanining_contract_duration -= 1
    ev.laxity = ev.stay_time - ((ev.soc_final - ev.soc_t) * ev.battery_cap)/(ev.alpha_c * ETA_C)

    ev.bool_c_d = True
    
    return e_charging
 
def discharge_v2g_EV(ev, e_discharge_available, e_short):
    ETA_D = 0.98
    ETA_C = 0.98
        
    e_discharge_available = np.abs(e_discharge_available)
    
    ev.soc_t = ev.soc_t - (e_discharge_available/ETA_D)/ev.battery_cap
    ev.stay_time -= 1
    ev.laxity = ev.stay_time - ((ev.soc_final - ev.soc_t) * ev.battery_cap)/(ev.alpha_c * ETA_C)


    ev.bool_c_d = True
    
    ev.remaining_contract_duration -= 1
    
    
    #print('discharged_e ', -1 * e_discharge_available)

    ev.remanining_discharge_e += -1 * e_discharge_available

    # JS: Is this right?
    ev.new_v2g_val = ev.remanining_discharge_e

    #print('REMIANING DISCHARGE E after 1 ', ev)
    #print('REMIANING DISCHARGE E after 1E ', ev.remanining_discharge_e)


    return e_discharge_available

def get_types_and_possible_discharge_e(E_EV_dict):
    DAY_HRS = 24
    ALPHA_D = 11
    ETA_D = 0.98
    
    hourly_dsch_e_lst = []
    for h in range(DAY_HRS):
        for lax in E_EV_dict['ev_laxity']['hour_{}'.format(h)]:
            hourly_dsch_e_lst.append(ALPHA_D * ETA_D * (lax/2))

    total_num_evs = len(hourly_dsch_e_lst)
            
    # Generating more values for EV discharge in the contract as seen from historical data    
    
    for more_ed_vals in range(20,55,5):
        hourly_dsch_e_lst.append(more_ed_vals)

    # Now, getting the number of types and possible discharge E

    hourly_dsch_e_set = set(hourly_dsch_e_lst) # this set contains the unique values for discharge energy
    number_of_types = len(hourly_dsch_e_set)
    sum_possible_dsch_e = sum(hourly_dsch_e_set)
    
    # Now, getting the distribution of types
    
    mean, var  = scipy.stats.distributions.norm.fit(hourly_dsch_e_lst)
    x = np.linspace(0,np.ceil(max(hourly_dsch_e_set)),number_of_types)
    fitted_data = scipy.stats.distributions.norm.pdf(x, mean, var)
#     pi_m = np.sort(fitted_data/np.sum(fitted_data))
    pi_m = (fitted_data/np.sum(fitted_data))
    

#     plt.hist(hourly_dsch_e_lst, density=True)
#     plt.plot(x,fitted_data,'r-')

    #return number_of_types, sum_possible_dsch_e, pi_m, total_num_evs
    return total_num_evs

def get_ev_utility(theta, g, w, gamma, bat_deg):
    return g - (w * gamma) / (bat_deg * theta)


def get_contract2(num_types, n_ev, tau):   
    TAU = tau # contract duration
    ALPHA_D = 11
    DAY_HRS = 24

    GAMMA = 1
    KAPPA = 0.1

    M = num_types
    PI_M = np.ones(M) * (1/M)

    V_BATT = 150
    BAT_DEG = 2 * 80
    
    # VARS

    Y = cp.Variable(M, nonneg=True)
    Z = cp.Variable(M, nonneg=True)

    constraints = []
    
    for idx in range(M):
        theta_m = idx + 1
        
        if (idx == 0): # type-1
            constraints += [Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m)) == 0]
            constraints += [Y[idx] >= 0]
            constraints += [Z[idx] >= 0]

        elif (idx >= 1):
            constraints += [(Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m))) == (Y[idx-1] - ((Z[idx-1] * GAMMA)/(BAT_DEG * theta_m)))]
            constraints += [Y[idx] >= Y[idx-1]]
            constraints += [Z[idx] >= Z[idx-1]]
            constraints += [Z[idx] <= (ALPHA_D  * TAU)]          

    objective_func = np.zeros(M)

    for idx in range(M):
        objective_func += cp.sum((PI_M[idx]) * n_ev * (KAPPA*(cp.log(Z[idx]+1)) - (Y[idx])))

    obj = cp.Maximize(cp.sum(objective_func))
    prob = cp.Problem(obj, constraints)


    prob.solve(verbose=False)
    if prob.status != 'optimal':
        raise Exception("Optimal contracts not found")
    return Z.value, Y.value


def get_online_alg_result_mc_simul(seed, day_no, current_date, unique_dates, sampling_unique_dates, orig_ev_dict, ratio_EV_discharge, pv_gen_df, price_df, sampling_pv_gen_df, sampling_price_df, num_samples, all_bids_sample_paths, EV_TYPES=[1,2,3], TAU = 3, KAPPA = 0.1, GAMMA = 1, BAT_DEG = 1, ev_rng = None,type_probs = None):
    #E_im_price = util.load_result(r'J:\Thesis_code\thesis_code_saidur\thesis_code_new_22\new_expected_values\E_price_imbalance')
    E_im_price = util.load_result(r'new_expected_values/E_price_imbalance')
    do_print = False
    
    if ev_rng is None:
        rng = np.random.default_rng(seed)
    else:
        rng = ev_rng
    # +++++++++++++++++++++++++++ For each of the sample paths' bid, solving STAGE - 2 +++++++++++++++++++++++++++++++++++    
    DAY_HRS = 24
    revenue_sample_paths = []
    im_buy_sample_paths = []
    im_sell_sample_paths = []
    da_revenue_sample_paths = []
    retail_revenue_sample_paths = []
    owner_pay_sample_paths = []
    assigned_type = []
    realized_type = []
    time_to_soc_sample_paths = []
    m = current_date.month
    
    if(m >= 1 and m <= 3):
        season = 'Winter'
    elif(m >= 4 and m <= 6):
        season = 'Spring'
    elif(m >= 7 and m <= 9):
        season = 'Summer'  
    elif(m >= 10 and m <= 12):
        season = 'Autumn'

    for b in range(1):
    
        bids = np.mean(np.array(all_bids_sample_paths)[day_no][0:num_samples], axis=0)
        da_revenue_lst =  np.zeros(DAY_HRS)
        imbalance_buy_lst = np.zeros(DAY_HRS)
        imbalance_sell_lst = np.zeros(DAY_HRS)
        revenue_lst = np.zeros(DAY_HRS)
        retail_revenue_lst = np.zeros(DAY_HRS)
        owner_pay_lst = np.zeros(DAY_HRS)

        #_, bids = perform_monte_carlo_simul(seed, date, unique_dates, ratio_EV_discharge, pv_gen_df, price_df)
        ev_dict = update_ev_dict_allowed_discharge_evs(seed, orig_ev_dict, ratio_EV_discharge)

        available_ev_lst = []
        num_discharge_allowed_evs = 0 #just to keep track on numbers
        ev_history_lst = []
        avg_time_to_full_soc = []

        pv_gen_lst = np.array(list(pv_gen_df.loc[(pv_gen_df.index == current_date)]['PV_Vol']))
        
        # We do not consider solar gen. so setting it to ZERO
        # ==========================================================================
        #  FIRST CHANGE
        # ==========================================================================
        pv_gen_lst = pv_gen_lst * np.zeros(DAY_HRS)
        
        da_price_lst = np.array(list(price_df.loc[(price_df.index == current_date)]['price_da']))
        im_price_lst = np.array(list(price_df.loc[(price_df.index == current_date)]['price_imbalance']))

        for idx, p in enumerate(im_price_lst):
            if p < 0:
                im_price_lst[idx] = 0
    
    
        for idx, p in enumerate(da_price_lst):
            if p < 0:
                da_price_lst[idx] = 0
                
        
        CONTRACT_TAU = TAU
        NUM_EV_TYPES = len(EV_TYPES)

        #<!-->
        # EFFECTIVE_TYPES = 2
        EFFECTIVE_TYPES = len(EV_TYPES)

        #n_ev = get_types_and_possible_discharge_e(util.load_result(r'J:\Thesis_code\thesis_code_saidur\thesis_code_new_22\new_expected_values\E_EV_dict'))
        n_ev = get_types_and_possible_discharge_e(util.load_result(r'new_expected_values/E_EV_dict'))

        #whole_contract = get_contract(NUM_EV_TYPES, n_ev, CONTRACT_TAU)
        whole_contract = get_contract_customtypes(EV_TYPES, n_ev=n_ev,
                                                  TAU = TAU, KAPPA = KAPPA, GAMMA = GAMMA,
                                                  BAT_DEG = BAT_DEG )

        if do_print: print(f"{EV_TYPES=}, {n_ev=}, {TAU=}, {KAPPA=}, {GAMMA=}, {BAT_DEG=}, {whole_contract=}")
        # ============ TO DO: incorporate multiple possible \tau values ==============================
        for current_time in range(DAY_HRS):
            revenue = 0
            da_revenue = 0
            im_buy_revenue = 0
            im_sell_revenue = 0
            retail_revenue = 0
            ev_owner_payback = 0
            total_chrg_demand = 0
            
            E_ev_zero_lax = 0
            for ev in available_ev_lst:
                ev.bool_c_d == False
                if (ev.bool_c_d == False and ev.completed == False and (math.floor(ev.laxity)) <= 0 and (ev.soc_t != ev.soc_final)): # JS: Why floor? Also, never compare equality in float.
                    e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
                    if(e_required > ev.alpha_c):
                        e_required = ev.alpha_c                    
                    E_ev_zero_lax += e_required
            # Get the EVs
            found_evs_lst = get_connected_ev(ev_dict, current_time)

            # Form available_ev_lst
            if(found_evs_lst[0] != -1):
                for found_ev in found_evs_lst:
                    ev_history_lst.append(found_ev)
                    if do_print: print(f"{current_time} Found ev <{found_ev.stay_time=}>, energy_init={(found_ev.soc_init*found_ev.battery_cap):.2f}, {found_ev.feasible_discharge_e=:.2f}")
                                                                                          # nov2g: disable stay time condition
                    if(ev_history_lst.index(found_ev) in ev_dict['ev_allowed_discharge']):# and (found_ev.stay_time >= CONTRACT_TAU): # Condition 0 (would allow discharge), contition 1 
                        # Javier refactor conditions
                        found_ev.allow_discharge = False
                        if type_probs is None:
                            ev_true_type = rng.choice(list(range(NUM_EV_TYPES))) # Index of the type
                        else:
                            ev_true_type = rng.choice(list(range(NUM_EV_TYPES)), p=type_probs) # Index of the type

                        assigned_type.append(ev_true_type)
                        flag_accepted_contract = 0
                        gs, ws = whole_contract[0], whole_contract[1]
                        #if do_print: print(f"New ev enters {ev_true_type=}", end = ' ')
                        #print(f"{ev_true_type=}, {gs=}, {ws=}")
                        if do_print: print(f"--{ev_true_type=} ", end="")
                        while found_ev.allow_discharge == False and ev_true_type >= 0:
                            g, w = gs[ev_true_type], ws[ev_true_type]
                            #if do_print: print(f"{g=:.2f}, {w=:.2}", end = ' ')
                            # Condition 2, 3 and 4 (extra: if accepting contracts for a different type, make sure that you get positive utility)
                            #print(f"--{ev_true_type=} ({w=:.2f} <= {found_ev.soc_init * found_ev.battery_cap=:.2f}) and ({w=:.2f} <= {found_ev.feasible_discharge_e=:.2f}) and ({get_ev_utility(EV_TYPES[ev_true_type], g, w, GAMMA, BAT_DEG)=:.2f} >= 0)")

                            # Javier changes for nov2g
                            #if (w <= found_ev.soc_init * found_ev.battery_cap) and (w <= found_ev.feasible_discharge_e) and (get_ev_utility(EV_TYPES[ev_true_type], g, w, GAMMA, BAT_DEG) >= 0): 
                            if True: # nov2g: Every car gets a contract
                                w = 1 # nov2g: Disallow discharge
                                g = 0 # nov2g: No payback
                                found_ev.allow_discharge = True
                                found_ev.remaining_contract_duration = CONTRACT_TAU
                                found_ev.remaining_discharge_e = w 
                                ev_owner_payback += -1 * g
                                num_discharge_allowed_evs += 1
                                #if do_print: print(f"Contract accepted {ev_true_type=}, {g=}, {w=}", end= ' ')
                            else:
                                ev_true_type -= 1 # Pretend to be a lower type (higher perceived battery degradation cost)
                                #if do_print: print(f"- type decreased {ev_true_type=}, ", " ")
                        if do_print: print(f"({w=:.2f} <= {found_ev.soc_init * found_ev.battery_cap=:.2f}) and ({w=:.2f} <= {found_ev.feasible_discharge_e=:.2f}) and ({get_ev_utility(EV_TYPES[ev_true_type], g, w, GAMMA, BAT_DEG)=:.2f} >= 0)", end="")
                        if do_print: print(f"--{ev_true_type=}")
                        realized_type.append(ev_true_type)
                        #input()
                    #if do_print: input()
                    available_ev_lst.append(found_ev) # JS: 

                    #    # === squeeze in the contract ====

                    #    # <!--> Javier changes order of whole contract
                    #    offered_contract = [oc for oc in whole_contract[1] if ((oc <= found_ev.feasible_discharge_e) and (oc <= found_ev.soc_init * found_ev.battery_cap))] # JS: Condition 3 and 2
                    #    if(len(offered_contract) == 0): # This condition is true if all the offerings in the contract are infeasible for the arrived ev
                    #        found_ev.allow_discharge = False
                    #    else:
                    #        # Javier modifies <!-->
                    #        offered_contract = offered_contract[0:EFFECTIVE_TYPES]
                    #        # ====================== Condition (3) goes here ======================
                    #        if(CONTRACT_TAU > found_ev.stay_time): # JS: Condition 1
                    #            found_ev.allow_discharge = False
                    #        else:                                
                    #            # <!--> Javier's remove laxity
                    #            ev_true_type = rng.choice([1,2,3]) # Index of the type
                    #            if ((ev_true_type <= 0) or (ev_true_type > len(offered_contract))): # JS: Very hacky, condition 2 again. If the type's contract in infeasible, skip.
                    #                # First part of 'if' condition -> if EV owner type less outside the lowest offered contract
                    #                # Second part of 'if' condition -> if EV owner type greater than maximum offered contract
                    #                found_ev.allow_discharge = False                    
                    #            else:
                    #                found_ev.allow_discharge = True
                    #                found_ev.remaining_contract_duration = CONTRACT_TAU
                    #                found_ev.remaining_discharge_e = offered_contract[ev_true_type-1] # -1 because of list indexing. 
                    #                # <!--> Javier changes order of whole contract
                    #                ev_owner_payback += -1 * (whole_contract[0][list(whole_contract[1]).index(offered_contract[ev_true_type-1])])
                    #                num_discharge_allowed_evs += 1
                    #else:
                    #    found_ev.allow_discharge = False    
                    #available_ev_lst.append(found_ev) # JS: 
                    
#             # ====== Pre-step 1 for the V2G scheduling ======
#             available_v2g_ev_lst = []
#             for ev in available_ev_lst:
#                 if (ev.allow_discharge == True):
#                     available_v2g_ev_lst.append(ev)
                    
            # Find EVs with Low laxity
            ac_lst = []
            for ev in available_ev_lst:
                ev.bool_c_d = False #Reset for all EVs on every hour
                if(current_time >= ev.departure_time or ev.stay_time == 0):
                    ev.bool_c_d = True
                    

                elif(ev.bool_c_d == False and ev.completed == False and (math.floor(ev.laxity)) <= 0 and (ev.soc_t != ev.soc_final)): #JS: If EV's have completed charging, bad idea to compare equality in bools, but i'll let it slide :D
                    if(ev.allow_discharge == False): # this condition because only non-V2G EVs are charged in w/o using scheduler
                        e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
                        if(e_required > ev.alpha_c):
                            e_required = ev.alpha_c
                        # Total ammount of energy for non-V2G EV's at this timestep
                        ac_lst.append(charge_EV(ev, e_required))
                        
            # Total charging demand
            #     x_t                         + e_t
            x_t = np.array(bids[current_time] + np.sum(np.array(ac_lst)))
            
            total_chrg_demand = np.sum(np.array(ac_lst)) # e_t
    
            
            
            # ====== Pre-step for the V2G scheduling ======
            # get surplus/deficit -> accurate val for present time step, and expected vals for future time steps
#             updated_E_VPP = []
#             for t in range(DAY_HRS):
#                 if(t == current_time):
#                     updated_E_VPP.append(pv_gen_lst[t] - x_t)
#                 elif(t < current_time):
#                     updated_E_VPP.append(0)
#                 else:
#                     updated_E_VPP.append(pv_gen_lst[t])
            
             # ==========================================================================
            #  SECOND CHANGE
            # ==========================================================================
            updated_E_VPP = []
            for t in range(DAY_HRS):
                if(t == current_time):
                    updated_E_VPP.append(x_t)
                elif(t < current_time):
                    updated_E_VPP.append(0)
                else:
                    updated_E_VPP.append(bids[t])
            
            # =============== HERE, I will schedule the charging profile of the EVs that decided to participate in V2G ===========
            
            available_v2g_ev_lst = []
            available_v2g_energies_lst = []
            for ev in available_ev_lst:
                if (ev.allow_discharge == True and ev.completed == False):
                    available_v2g_ev_lst.append(ev) 
                    if(ev.new_v2g_val != -1000):
                        available_v2g_energies_lst.append(ev.new_v2g_val)
                        #print('ev id ', ev) 
                        #print('ev disch E', ev.new_v2g_val)
                    else:
                        available_v2g_energies_lst.append(ev.remaining_discharge_e)
                        #print('ev id ', ev) 
                        #print('ev disch E', ev.new_v2g_val)
                                    
            
                        
            if(len(available_v2g_ev_lst) > 0):
                                
                found_ev_schedule = new_ev_scheduler.main(seed, available_v2g_ev_lst, current_time, E_im_price, updated_E_VPP, CONTRACT_TAU)
                
                for ev_idx, my_ev in enumerate(available_v2g_ev_lst):

                    my_ev.scheduled_profile = found_ev_schedule[ev_idx][current_time]
                    
                    v2g_ev_idx = available_ev_lst.index(my_ev)
                    v2g_ev = available_ev_lst[v2g_ev_idx]
                    
                    v2g_ev.remanining_discharge_e = available_v2g_energies_lst[ev_idx]
                                   
                    if(v2g_ev.scheduled_profile >= 0):
                        e_chrg = charge_v2g_EV(v2g_ev, v2g_ev.scheduled_profile)
                        total_chrg_demand += e_chrg # charge the v2g
                    else:                                               
                        discharged_e = -1 * discharge_v2g_EV(v2g_ev, v2g_ev.scheduled_profile, 1000) 

                    # updating x_t value with the charge/discharge decision of V2G-participating EVs
                    x_t += my_ev.scheduled_profile

            da_revenue = bids[current_time] * da_price_lst[current_time]
            
            # Adding the revenue by getting money from EV owners for charging their batteries
            EV_PWR_RETAIL = 0.13
            retail_revenue += total_chrg_demand * EV_PWR_RETAIL
            #da_revenue += total_chrg_demand * EV_PWR_RETAIL
            
            # ========== Now, we calculate the remaining surplus/deficit and decide accordingly ==========

            # Get the surplus/deficit  
            # old for the case WITH SOLAR
#             if(pv_gen_lst[current_time] >= x_t):
#                 # Surplus
#                 pv_diff = pv_gen_lst[current_time] - x_t
#                 im_sell_revenue += pv_diff * im_price_lst[current_time] #surplus_pv_gen(current_time, pv_diff, available_ev_lst, im_price_lst)
#                 pv_diff = 0

#             elif (x_t > pv_gen_lst[current_time]):
#                 # Deficit
#                 pv_diff = pv_gen_lst[current_time] - x_t
#                 e_short = pv_diff #shortage_pv_gen(current_time, pv_diff, available_ev_lst, da_price_lst, im_price_lst)
#                 if(e_short > 0):
#                     im_buy_revenue += -1 * e_short * im_price_lst[current_time]
#                 e_short = 0

            # Get the surplus/deficit  
            # NEW for the case W/0 SOLAR
            # ==========================================================================
            #  THIRD CHANGE
            # ==========================================================================
            if (x_t < 0):
                # Surplus
                im_sell_revenue += -1 * x_t * im_price_lst[current_time] #surplus_pv_gen(current_time, pv_diff, available_ev_lst, im_price_lst)
                x_t = 0

            elif (x_t > 0):
                # Deficit
                im_buy_revenue += -1 * x_t * im_price_lst[current_time]
                x_t = 0
                

            revenue = da_revenue + im_sell_revenue + im_buy_revenue + retail_revenue + ev_owner_payback
            #print(f"{day_no=}, {current_time=}, {bids[current_time]=:.3f} {da_price_lst[current_time=:.3f]}, {da_revenue=:.3f}, {retail_revenue=:.3f}")
            #input()

            #one elem per hour
            revenue_lst[current_time] = revenue
            da_revenue_lst[current_time] = da_revenue
            imbalance_buy_lst[current_time] = im_buy_revenue
            imbalance_sell_lst[current_time] = im_sell_revenue
            retail_revenue_lst[current_time] = retail_revenue
            owner_pay_lst[current_time] = ev_owner_payback


            for ev in available_ev_lst:                   
                if(ev.bool_c_d == False):
                    if(ev.allow_discharge == False):
                        # For EVs not charged or discharged, ONLY Laxity update
                        e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
                        ev.stay_time -= 1 
                        if(ev.remanining_contract_duration > 0):
                            ev.remanining_contract_duration -= 1
                        ev.laxity = ev.stay_time - (e_required/(ev.alpha_c * ev.eta_c))    


            for ev in available_ev_lst: 
                if(current_time == ev.departure_time or ev.stay_time == 0):
                    if(ev.completed == False):
                        avg_time_to_full_soc.append(1+(current_time - ev.arrival_time))
                    ev.completed = True
                    ev.bool_c_d = True
              
        #all_revenue_sample_paths.append(np.sum(revenue_lst))
        #One elem per day
        revenue_sample_paths.append(np.sum(revenue_lst))
        im_buy_sample_paths.append(np.sum(imbalance_buy_lst))
        im_sell_sample_paths.append(np.sum(imbalance_sell_lst))
        da_revenue_sample_paths.append(np.sum(da_revenue_lst))
        retail_revenue_sample_paths.append(np.sum(retail_revenue_lst))
        owner_pay_sample_paths.append(np.sum(owner_pay_lst))
        time_to_soc_sample_paths.append(np.mean(avg_time_to_full_soc,axis=0))

    return np.average(revenue_sample_paths), np.average(im_buy_sample_paths), np.average(im_sell_sample_paths), np.average(time_to_soc_sample_paths), len(ev_dict['ev_allowed_discharge']), num_discharge_allowed_evs, np.average(da_revenue_sample_paths), np.average(retail_revenue_sample_paths), np.average(owner_pay_sample_paths), assigned_type, realized_type
    #return np.sum(revenue_lst), np.sum(imbalance_buy_lst), np.sum(imbalance_sell_lst), np.mean(avg_time_to_full_soc)

def update_ev_dict_allowed_discharge_evs(seed, ev_dict, ratio):
    rng = np.random.default_rng(seed)
    
    total_evs = 0
    for h in range(24):
        total_evs += len(ev_dict['init_soc']['hour_{}'.format(str(h))])
    
    ev_allowed_discharge = rng.choice(total_evs, size=round(total_evs*(ratio/100)), replace=False)
    
    ev_dict['ev_allowed_discharge'] = ev_allowed_discharge
    return ev_dict

# def get_pred_ev_type(lax, num_effective_types): # DEAD_CODE
#         
#     MAX_LAXITY = 12
#     
#     # 6 if we have 2 effective types and EV laxities range from 0 to 12 according to historical data
#     #EV OWNER TYPE SEGMENTS  >= 0 < 6 | >= 6 and above 
#         
#     LAX_AND_TYPE_BIN_DIVISOR = MAX_LAXITY/num_effective_types  
#     
#     ev_pred_type = (lax//(LAX_AND_TYPE_BIN_DIVISOR) + 1)
#     
#     if(ev_pred_type > num_effective_types):
#         return num_effective_types
#     else:
#         return int(lax//(LAX_AND_TYPE_BIN_DIVISOR) + 1)


# def solve_stage_1_w_expected_vals(seed, date, ratio_EV_discharge): # DEAD CODE
#     
#     m = date.month
#     
#     if(m >= 1 and m <= 3):
#         season = 'Winter'
#     elif(m >= 4 and m <= 6):
#         season = 'Spring'
#     elif(m >= 7 and m <= 9):
#         season = 'Summer'  
#     elif(m >= 10 and m <= 12):
#         season = 'Autumn' 
#          
#     E_PV_gen = util.load_result(r'J:\Thesis_code\thesis_code_saidur\thesis_code_new_22\new_expected_values\E_PV_{}'.format(season))
#     E_im_price = util.load_result(r'J:\Thesis_code\thesis_code_saidur\thesis_code_new_22\new_expected_values\E_price_imbalance')
#     E_da_price = util.load_result(r'J:\Thesis_code\thesis_code_saidur\thesis_code_new_22\new_expected_values\E_price_da')
#     E_EV_dict = util.load_result(r'J:\Thesis_code\thesis_code_saidur\thesis_code_new_22\new_expected_values\E_EV_dict')
#     
#     _, bids, _ = latest_deterministic_solver_ev_penetration.main(seed, E_PV_gen, E_da_price, E_im_price, E_EV_dict, ratio_EV_discharge)
#     
#     return bids

# def discharge_EV(ev, e_discharge_available, e_short):
        
#     ETA_D = 0.98
#     ETA_C = 0.98
    
#     e_discharge_available = np.abs(e_discharge_available)
    
#     if(e_discharge_available <= 0):
#         e_discharge_available = 0
    
#         return e_discharge_available

#     else: 
#         if(e_discharge_available > np.abs(ev.alpha_d)):
#             e_discharge_available = np.abs(ev.alpha_d)
#         elif(e_discharge_available > e_short):
#             e_discharge_available = e_short

#         ev.soc_t = ev.soc_t - (e_discharge_available/ETA_D)/ev.battery_cap
#         ev.stay_time -= 1
#         ev.laxity = ev.stay_time - ((ev.soc_final - ev.soc_t) * ev.battery_cap)/(ev.alpha_c * ETA_C)

#         if(ev.laxity == 0):
#             ev.laxity = 0
#         ev.bool_c_d = True

#         return e_discharge_available

# def get_available_discharge_energy(ev): # DEAD CODE
#     
#     if((ev.remaining_discharge_e == 0) or (ev.remanining_contract_duration == 0)):
#         e_discharge = 0
#     else:
#         e_required = (ev.soc_final - ev.soc_t) * ev.battery_cap
#         look_ahead_e_required = ev.alpha_c * ev.eta_c * (ev.stay_time - 1)
#         e_discharge = look_ahead_e_required - e_required
# 
#         if (e_discharge <= 0):
#             e_discharge = 0
# 
#         elif (e_discharge >= np.abs(ev.alpha_d)):
#             e_discharge = np.abs(ev.alpha_d)
#             
#         if (e_discharge > ev.remaining_discharge_e):
#             e_discharge = ev.remaining_discharge_e
#         
#         ev.remaining_discharge_e -= e_discharge
#                 
#     return e_discharge

# def shortage_pv_gen(current_time, pv_diff, available_ev_lst, da_price_lst, im_price_lst): # DEAD CODE
#     
#     imbalance_buy = 0
#     DISCHRG_THRESH = 0
#     
#     e_short = np.abs(pv_diff)
# 
#     discharge_e_dict = {}
#     sum_disch_E = 0
#     
#     #print("BEFOREEE DISCHRGG E-short ", e_short)
#     
#     # Obtain the EVs that have POSITIVE LAXITY and ALLOW DISCHARGE
#     for ev in available_ev_lst:
#         if(ev.laxity > DISCHRG_THRESH and ev.allow_discharge == True and ev.bool_c_d == False):
#             # storing the EV obj address in the dict as "key" and the available discharge energy as the "item"
#             discharge_e_dict[ev] = get_available_discharge_energy(ev)
#     #print('discharge_e_dict ', discharge_e_dict)
#     if(len(discharge_e_dict) != 0):
#         # Descending sort
#         discharge_e_dict = sorted(discharge_e_dict.items(), key=lambda x: x[1], reverse=True)
# 
#         for ev in discharge_e_dict:
#             sum_disch_E += ev[1]
#             
#         e_short_orig = e_short
#         for ev in discharge_e_dict:
#             if (e_short > 0 and ev[1] != 0):
#                 if(sum_disch_E > e_short):
#                     e_discharged = discharge_EV(ev[0], (ev[1]/sum_disch_E)*e_short_orig, e_short)
#                     #ev[0].dsch_revenue_contrib += e_discharged * im_price_lst[current_time]
#                 else:    
#                     e_discharged = discharge_EV(ev[0], ev[1], e_short)    
#                 e_short -= e_discharged
#                 #print('EEE_DISCCC ', e_discharged)
#                 #revenue_from_ev_dischrg += e_discharged * im_price_lst[current_time]
#         
#         #print('SSORRRTT ',e_short)
#     return e_short


# def surplus_pv_gen(current_time, pv_diff, available_ev_lst, im_price_lst): # DEAD_CODE
# 
#     imbalance_sell = 0
#     e_extra = pv_diff
#     
#     charge_e_dict = {}
# 
#     for ev in available_ev_lst:
#         if(ev.bool_c_d == False and ev.soc_t < ev.soc_final):
#             charge_e_dict[ev] = ev.laxity
# 
# 
#     # sort based on ascending order of Laxity. Smaller lax => more urgent to charge!
#     if(len(charge_e_dict) != 0):
#         charge_e_dict = sorted(charge_e_dict.items(), key=lambda x: x[1], reverse=False)
# 
#         for ev in charge_e_dict:
#             if(e_extra > 0):
#                 e_charging = charge_EV(ev[0], e_extra)
#                 e_extra -= e_charging
# #                 if(im_price_lst[current_time] > 0):
# #                     imbalance_sell += e_charging * im_price_lst[current_time]
#     
#     # Sell remaining surplus energy to imbalance market after EV charging complete!
#     if(e_extra > 0 and im_price_lst[current_time] > 0):
#         imbalance_sell += e_extra  * im_price_lst[current_time]
#     return imbalance_sell


# def create_forecast_ev_dict(seed, sample_no): # DEAD CODE x_x
#     
#     rng = np.random.default_rng(seed)
#     ev_rng = np.random.default_rng(seed+sample_no)
# 
#     
#     DAY_HRS = 24
#     FINAL_SOC = 0.97
#     ALPHA_C = 11
#     B_CAP = 80
#     ETA_C = 0.98
#     NOISE_STD = 0.1
# 
#     ev_dict = {}
#     ev_outer_keys = ['init_soc','ev_stay_t','ev_laxity']
# 
#     for key in ev_outer_keys:
#         ev_dict[key] = {}
# 
#     for key in ev_outer_keys:
#         for h in range(DAY_HRS):
#             ev_dict[key]['hour_{}'.format(str(h))] = []
#                  
#     for hour in range(DAY_HRS):
#         num_arrived_ev = ev_data_sampler.sample_num_EV_arrivals(rng, hour)
#         
# #         print('in forecast ', num_arrived_ev)
#         
#         num_arrived_ev_noise = ev_rng.normal(0, (num_arrived_ev*NOISE_STD))
#         
#         num_arrived_ev += num_arrived_ev_noise
# #         if(num_arrived_ev_noise > 0):
# #             num_arrived_ev = np.ceil(num_arrived_ev)
#         num_arrived_ev = int(num_arrived_ev)
#         
#         if(num_arrived_ev <= 0):
#             num_arrived_ev = 0
#         
#         for _ in range(num_arrived_ev):
#             stay_t = ev_data_sampler.sample_ev_stay_time(rng, hour)
#             stay_t_noise = ev_rng.normal(0, (stay_t*NOISE_STD))
#             stay_t += stay_t_noise
#             if(stay_t_noise > 0):
#                 stay_t = np.ceil(stay_t)
#             stay_t = int(stay_t)
# 
#             
#             if(stay_t >= (DAY_HRS-1)):
#                 stay_t = DAY_HRS-1
#             elif(stay_t < 0):
#                 stay_t = 0
#             
#             init_soc = ev_data_sampler.sample_init_soc(seed)
#             init_soc_noise = ev_rng.normal(0, (init_soc*NOISE_STD))
#             init_soc += init_soc_noise
#             seed += 1
# 
#             depart_time = hour + stay_t
#             if(depart_time > 23):
#                 depart_time = 23
#                 stay_t = depart_time - hour
#                 
#             if(init_soc >= FINAL_SOC):
#                 init_soc = FINAL_SOC
#             elif(init_soc < 0):
#                 init_soc = 0
#                                 
#             laxity = stay_t - (((FINAL_SOC - init_soc)*B_CAP)/(ALPHA_C * ETA_C))
#             if(laxity >= 0):
#                 ev_dict['ev_stay_t']['hour_{}'.format(hour)].append(stay_t)
#                 ev_dict['init_soc']['hour_{}'.format(hour)].append(init_soc)
#                 ev_dict['ev_laxity']['hour_{}'.format(hour)].append(laxity)
#                 
#     return ev_dict
