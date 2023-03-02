import cvxpy as cp
import numpy as np
import pandas as pd

     
def update_ev_dict_allowed_discharge_evs(seed, ev_dict, ratio):
        
    rng = np.random.default_rng(seed)
    
    total_evs = 0
    for h in range(24):
        total_evs += len(ev_dict['init_soc']['hour_{}'.format(str(h))])
    
    ev_allowed_discharge = rng.choice(total_evs, size=round(total_evs*(ratio/100)), replace=False)

    ev_dict['ev_allowed_discharge'] = ev_allowed_discharge
    return ev_dict, total_evs
    
    
def get_more_ev_info(seed, ev_dict, ratio):
    
    ev_dict, N_EV  = update_ev_dict_allowed_discharge_evs(seed, ev_dict, ratio)
            
    discharge_allowed_lst = ev_dict['ev_allowed_discharge']
    arrive_times_lst = []
    soc_init_lst = []
    stay_times_lst = []
    

    for hour in range(24):
        if(len(ev_dict['init_soc']['hour_{}'.format(str(hour))]) != 0):
            for ev_num in range(len(ev_dict['init_soc']['hour_{}'.format(str(hour))])):
                arrive_times_lst.append(hour)
                soc_init_lst.append(ev_dict['init_soc']['hour_{}'.format(str(hour))][ev_num])
                stay_times_lst.append(ev_dict['ev_stay_t']['hour_{}'.format(str(hour))][ev_num])
                
    FINAL_SOC = 0.97
    soc_final_lst = [FINAL_SOC] * N_EV
            
    I_EV_charging = np.zeros([N_EV, 24])
    D_EV_charging = np.zeros([N_EV, 24])
    
    for ev in range(N_EV):
        for time in range(arrive_times_lst[ev], arrive_times_lst[ev] + stay_times_lst[ev]):
            if(time >= 24):
                time = 23
            I_EV_charging[ev][time] = 1
            if ev in discharge_allowed_lst:
                D_EV_charging[ev][time] = 1
                    
                        
    return N_EV, I_EV_charging, D_EV_charging, soc_init_lst, soc_final_lst, discharge_allowed_lst, ev_dict
    
def get_bids_and_ev_charge_discharge_strategy(pv_gen_lst, da_price_lst, im_price_lst, active_chargers_lst, active_v2g_chargers_lst, I_EV_charging, D_EV_charging, soc_init_lst, soc_final_lst, discharge_allowed_lst):
   
    # Constant Values
    DAY_HRS = 24
    B_CAP = 80
    ALPHA_C = 11
    ALPHA_D = -11

    FINAL_SOC = 0.97
    MIN_SOC = 0.03

    MIN_BID = 0
    #MAX_BID = 40
    
    ETA_C = 0.98
    ETA_D = 0.98
    
    N_EV = I_EV_charging.shape[0]

    # Getting EV values and Forecast of PV gen and Prices
    s = pv_gen_lst
    p_da = da_price_lst
    p_im = im_price_lst
            
    # CVXPY Variables
    SELECT_M = cp.Variable((N_EV, DAY_HRS),boolean=True)
    
    AD = cp.Variable((N_EV, DAY_HRS),nonpos=True)
    AC = cp.Variable((N_EV, DAY_HRS),nonneg=True)
    
    y = cp.Variable((N_EV, DAY_HRS))

    x = cp.Variable(DAY_HRS)
    z = cp.Variable(DAY_HRS)
    
    # Constraints
    
    #print('III_EVVVV', I_EV_charging)
       
    
    constraints = []
    
            
    for ev in range(N_EV):
        if (ev in discharge_allowed_lst):
            for t in range(DAY_HRS):
                constraints += [AD[ev][t] >= I_EV_charging[ev][t] * ALPHA_D * (1-SELECT_M[ev][t])] 
                constraints += [AC[ev][t] <= I_EV_charging[ev][t] * ALPHA_C * SELECT_M[ev][t]]
                constraints += [y[ev][t] == (AC[ev][t] + AD[ev][t])]
                
        else:
            for t in range(DAY_HRS):            
                constraints += [AD[ev][t] == 0] 
                constraints += [AC[ev][t] <= I_EV_charging[ev][t] * ALPHA_C]
                constraints += [y[ev][t] == (AC[ev][t] + AD[ev][t])]
        
        constraints += [cp.sum(I_EV_charging[ev] @ (AC[ev] * ETA_C + AD[ev]/ETA_D)/B_CAP) + soc_init_lst[ev] == FINAL_SOC]
        #constraints += [cp.sum(I_EV_charging[ev] @ (AC[ev] * ETA_C + AD[ev]/ETA_D)/B_CAP) + soc_init[ev] >= MIN_SOC]
                        
       
    for t in range(DAY_HRS):
      

        constraints += [x[t] >=  -1 * active_chargers_lst[t] * ALPHA_C]
        constraints += [x[t] <=  s[t] + (-1 * active_v2g_chargers_lst[t] * ALPHA_D)]
        
        constraints += [z[t] ==  x[t]-s[t] + cp.sum(y, axis=0)[t]]
         

    Reward = x @ p_da.T - (z @ p_im.T)

    obj = cp.Maximize(Reward)
    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.MOSEK, verbose=False)
    prob.status

    # Print result.
    #print("The optimal value is", prob.value)
    #print("\nA solution x is ",x.value)
    #print("A solution y is ",y.value)
   
   
    #actual_bids = x.value
    #for t in times_discharge_for_day_ahead:
    #    actual_bids[t] -= A.value[t]  

        
    #print(y.value)
    
    #print("BIDSSSS ", x.value)
    
    return prob.value, x.value, z.value
    
    
def get_bids_only(pv_gen_lst, da_price_lst, im_price_lst, active_chargers_lst, active_v2g_chargers_lst, I_EV_charging, D_EV_charging, soc_init_lst, soc_final_lst, discharge_allowed_lst):
   
  
    # Constant Values
    DAY_HRS = 24
    B_CAP = 80
    ALPHA_C = 11
    ALPHA_D = -11

    FINAL_SOC = 0.97
    MIN_SOC = 0.03

    MIN_BID = 0
    #MAX_BID = 40
    
    ETA_C = 0.98
    ETA_D = 0.98
    
    N_EV = I_EV_charging.shape[0]

    # Getting EV values and Forecast of PV gen and Prices
    s = pv_gen_lst
    p_da = da_price_lst
    p_im = im_price_lst
            
    # CVXPY Variables    
    x = cp.Variable(DAY_HRS)
    z = cp.Variable(DAY_HRS)
    
    # Constraints
    
    #print('III_EVVVV', I_EV_charging)
       
    
    constraints = []
     
       
    for t in range(DAY_HRS):
      

        constraints += [x[t] >=  -1 * active_chargers_lst[t] * ALPHA_C]
        constraints += [x[t] <=  s[t] + (-1 * active_v2g_chargers_lst[t] * ALPHA_D)]
        
        constraints += [z[t] ==  x[t]-s[t]]
         

    Reward = x @ p_da.T - (z @ p_im.T)

    obj = cp.Maximize(Reward)
    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.GUROBI, verbose=False)
    prob.status

    # Print result.
    #print("The optimal value is", prob.value)
    #print("\nA solution x is ",x.value)
    #print("A solution y is ",y.value)
   
   
    #actual_bids = x.value
    #for t in times_discharge_for_day_ahead:
    #    actual_bids[t] -= A.value[t]  

        
    #print(y.value)
    
    #print("BIDSSSS ", x.value)
    
    return prob.value, x.value, z.value
    
    
    
def main(seed, pv_gen_lst, da_price_lst, im_price_lst, ev_dict, PERC_EV_PENETRATION):
    
    N_EV, I_EV_charging, D_EV_charging, soc_init_lst, soc_final_lst, discharge_allowed_lst, ev_dict = get_more_ev_info(seed, ev_dict, PERC_EV_PENETRATION)
        
    active_chargers_lst = np.sum(I_EV_charging, axis=0)
    active_v2g_chargers_lst = np.sum(D_EV_charging, axis=0)
    
    if(N_EV > 0):
        revenue, bids, z_val = get_bids_and_ev_charge_discharge_strategy(pv_gen_lst, da_price_lst, im_price_lst, active_chargers_lst, active_v2g_chargers_lst, I_EV_charging, D_EV_charging, soc_init_lst, soc_final_lst, discharge_allowed_lst)
    else:
        revenue, bids, z_val = get_bids_only(pv_gen_lst, da_price_lst, im_price_lst, active_chargers_lst, active_v2g_chargers_lst, I_EV_charging, D_EV_charging, soc_init_lst, soc_final_lst, discharge_allowed_lst)
   
    return revenue, bids, z_val
    
    
if __name__ == '__main__':
    main()