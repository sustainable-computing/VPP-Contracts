import numpy as np
import cvxpy as cp # type: ignore

def get_c_vpp(available_v2g_ev_lst, c_vpp_dept_times_lst, c_vpp_soc_init_lst, c_vpp_I_EV_charging, c_vpp_D_EV_charging,  c_vpp_D_EV_charging_count, c_vpp_pv_diff_lst, N_EV, p_im, current_time, tau):
    # Constants
    ALPHA_C = 11
    ALPHA_D = -11
    B_CAP = 80
    ETA_C = 0.98
    ETA_D = 0.98
    FINAL_SOC = 0.97
       
    DAY_HRS = 24

    # Input  
    I_EV_charging = c_vpp_I_EV_charging   
    D_EV_charging = c_vpp_D_EV_charging
    D_EV_charging_count = c_vpp_D_EV_charging_count
    
    # Variables    
    #SELECT_M = cp.Variable((N_EV, DAY_HRS),boolean=True)
    AD = cp.Variable((N_EV, DAY_HRS),nonpos=True)
    AC = cp.Variable((N_EV, DAY_HRS),nonneg=True)
    y = cp.Variable((N_EV, DAY_HRS))
    soc_t = cp.Variable((N_EV, DAY_HRS),nonneg=True)
    lax_t = cp.Variable((N_EV, DAY_HRS),nonneg=True)
    z = cp.Variable(DAY_HRS)
    
    constraints = []
    
    for ev in range(N_EV):
        for t in range(current_time):
            #constraints += [SELECT_M[ev][t] == 0]
            constraints += [AD[ev][t] == 0]
            constraints += [AC[ev][t] == 0]
            constraints += [y[ev][t] == 0]
            constraints += [soc_t[ev][t] == 0]
            constraints += [lax_t[ev][t] == 0]

    for ev_idx, ev in enumerate(available_v2g_ev_lst):
        for t in range(current_time, ev.departure_time):
            #print('TTT ', t)
            if (D_EV_charging[ev_idx][t] < 0):
                if(np.abs(D_EV_charging[ev_idx][t]) > np.abs(ALPHA_D)):
                    D_EV_charging[ev_idx][t] = ALPHA_D
                constraints += [AD[ev_idx][t] >= D_EV_charging[ev_idx][t] * D_EV_charging_count[ev_idx][t] ]
                #constraints += [AD[ev_idx][t] >= D_EV_charging[ev_idx][t] * SELECT_M[ev_idx][t]]
                constraints += [AC[ev_idx][t] <= I_EV_charging[ev_idx][t] * ALPHA_C ]
                #constraints += [AC[ev_idx][t] <= I_EV_charging[ev_idx][t] * (1-SELECT_M[ev_idx][t]) * ALPHA_C]
                #D_EV_charging[ev_idx][t] = 1
            else:
                constraints += [AD[ev_idx][t] == 0]
                constraints += [AC[ev_idx][t] <= I_EV_charging[ev_idx][t] * ALPHA_C]
                
            constraints += [y[ev_idx][t] == (AC[ev_idx][t] + AD[ev_idx][t])]
            
            if (t == current_time):
                constraints += [soc_t[ev_idx][t]  == c_vpp_soc_init_lst[ev_idx] + ((I_EV_charging[ev_idx][t] * AC[ev_idx][t] * ETA_C)/B_CAP + (D_EV_charging_count[ev_idx][t] * (AD[ev_idx][t]/ETA_D))/B_CAP)]
            else:
                constraints += [soc_t[ev_idx][t]  == soc_t[ev_idx][t-1]  + ((I_EV_charging[ev_idx][t] * AC[ev_idx][t] * ETA_C)/B_CAP + (D_EV_charging_count[ev_idx][t] * (AD[ev_idx][t]/ETA_D))/B_CAP)]
            
            constraints += [soc_t[ev_idx][t] >= (1-FINAL_SOC)]
            #constraints += [soc_t[ev_idx][t] <= (FINAL_SOC)]            
            
            constraints += [lax_t[ev_idx][t] == I_EV_charging[ev_idx][t] * (((c_vpp_dept_times_lst[ev_idx]-(t+1)) - (((FINAL_SOC - soc_t[ev_idx][t])*B_CAP)/(ALPHA_C*ETA_C))))]

            if ((t+1) < ev.departure_time):
                constraints += [lax_t[ev_idx][t] >= 0.001]
                #constraints += [lax_t[ev_idx][t] >= 0.0001]
                
            #if (t == (ev.departure_time - 1)):
            #    constraints += [soc_t[ev_idx][t] <= FINAL_SOC]
           

            
        #constraints += [cp.sum(AD[ev_idx]) <= tau]
            

    for ev_idx, ev in enumerate(available_v2g_ev_lst):
        for t in range(ev.departure_time, DAY_HRS):
            #constraints += [SELECT_M[ev_idx][t] == 0]
            constraints += [AD[ev_idx][t] == 0]
            constraints += [AC[ev_idx][t] == 0]
            constraints += [y[ev_idx][t] == 0]
            constraints += [soc_t[ev_idx][t] == 0]
            constraints += [lax_t[ev_idx][t] == 0]

    for t in range(DAY_HRS):
        constraints += [z[t] == cp.sum(y, axis=0)[t]]
            
        
    c_vpp = np.sum(c_vpp_pv_diff_lst - z) @ p_im.T
    
    obj = cp.Maximize(c_vpp)      
    problem = cp.Problem(obj, constraints)

    problem.solve(cp.MOSEK, mosek_params = {'MSK_IPAR_NUM_THREADS': 8, 'MSK_IPAR_BI_MAX_ITERATIONS': 2_000_000, "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800}, verbose=False)  

    if problem.status != 'optimal':
        #raise Exception("Optimal schedule not found")
        print("!!! Optimal schedule not found")
    
    #print('AD ', AD.value)
    #print("EV SCHEDD ", y.value)
    #print('SOCCC AFTER CHARGE ',soc_t.value)
    
    #print('=================================')
     
    
    return y.value
  
def get_c_vpp_input(available_ev_lst, E_im_price, E_VPP, current_time, tau):
    DAY_HRS = 24
            
    c_vpp_dept_times_lst = []
    c_vpp_soc_init_lst = []
    c_vpp_I_EV_charging = []
    c_vpp_D_EV_charging = []
    c_vpp_D_EV_charging_count = []
    
    p_im = []
    c_vpp_pv_diff_lst = []
    

    for idx, ev in enumerate(available_ev_lst):
        if (ev != 0 and ev.completed == False):

            temp_I_EV_charging = []
            temp_D_EV_charging = []
            temp_D_EV_charging_count = []
            
            c_vpp_dept_times_lst.append(ev.departure_time)
            c_vpp_soc_init_lst.append(ev.soc_t)

            for t in range(DAY_HRS):
                if (t >= 24):
                    t = 23
                #if((t >= ev.departure_time) or (t < ev.arrival_time)):
                if((t >= ev.departure_time) or (t < current_time)):
                    temp_I_EV_charging.append(0)
                    temp_D_EV_charging.append(0)
                    temp_D_EV_charging_count.append(0)
                else:
                    temp_I_EV_charging.append(1)

                    if (ev.allow_discharge == True):
                                               
                        if ( (t >= (ev.arrival_time)) and (t <= (ev.arrival_time + (tau - 1))) ):
                            # since contract duration is 1 so discharge MUST be done within first hour
                            if(ev.new_v2g_val != -1000):
                                temp_D_EV_charging.append(-1 * ev.new_v2g_val)
                                temp_D_EV_charging_count.append(1)
                            else:
                                temp_D_EV_charging.append(-1 * ev.remaining_discharge_e)
                                temp_D_EV_charging_count.append(1)
                        else:
                            temp_D_EV_charging.append(0)
                            temp_D_EV_charging_count.append(0)
                    else:
                        temp_D_EV_charging.append(0)
                        temp_D_EV_charging_count.append(0)
                        
            c_vpp_I_EV_charging.append(temp_I_EV_charging)
            c_vpp_D_EV_charging.append(temp_D_EV_charging)
            c_vpp_D_EV_charging_count.append(temp_D_EV_charging_count)
           
               
    for t in range(DAY_HRS): 
        if(t < current_time or t >= c_vpp_dept_times_lst[-1]):
            p_im.append(0)
            c_vpp_pv_diff_lst.append(0)
        else:
            p_im.append(E_im_price[t])
            c_vpp_pv_diff_lst.append(E_VPP[t])
            
    p_im = np.array(p_im)
    c_vpp_pv_diff_lst = np.array(c_vpp_pv_diff_lst)
                   
    c_vpp_I_EV_charging = np.vstack(c_vpp_I_EV_charging)    
    c_vpp_D_EV_charging = np.vstack(c_vpp_D_EV_charging)
    c_vpp_D_EV_charging_count = np.vstack(c_vpp_D_EV_charging_count)
    
    #print('current_time ', current_time)
    #print('I_EV ',c_vpp_I_EV_charging)
    #print('D_EV ',c_vpp_D_EV_charging)
    #print('D_EV_count ',c_vpp_D_EV_charging_count)
    #print('SOC BEFORE',c_vpp_soc_init_lst)
    #print('DEPT_T ',c_vpp_dept_times_lst)
                
    return c_vpp_dept_times_lst, c_vpp_soc_init_lst, c_vpp_I_EV_charging, c_vpp_D_EV_charging, c_vpp_D_EV_charging_count, p_im, c_vpp_pv_diff_lst


def offer_discharge_incentive(seed, available_v2g_ev_lst, current_time, E_im_price, E_VPP, tau):
    #tau = 3
    DAY_HRS = 24
    
    rng = np.random.default_rng(seed)
  
    c_vpp_dept_times_lst, c_vpp_soc_init_lst, c_vpp_I_EV_charging, c_vpp_D_EV_charging, c_vpp_D_EV_charging_count, p_im, c_vpp_pv_diff_lst = get_c_vpp_input(available_v2g_ev_lst, E_im_price, E_VPP, current_time, tau)
    N_EV = c_vpp_I_EV_charging.shape[0]
    new_EV_schedule = get_c_vpp(available_v2g_ev_lst, c_vpp_dept_times_lst, c_vpp_soc_init_lst, c_vpp_I_EV_charging, c_vpp_D_EV_charging, c_vpp_D_EV_charging_count, c_vpp_pv_diff_lst, N_EV, p_im, current_time, tau)
    
    return new_EV_schedule
    
    
def main(seed, available_v2g_ev_lst, current_time, E_im_price, E_VPP, tau):

    return offer_discharge_incentive(seed, available_v2g_ev_lst, current_time, E_im_price, E_VPP, tau)
