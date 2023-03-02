import numpy as np
from utility import utility as util

def sample_EV_arrival(seed, E_ev_arrival_probs):
    rng = np.random.default_rng(seed)
    ev_arrival_lst = []
    for hr in range(24): 
        ev_arrival_lst.append(rng.choice([1,0], p=[E_ev_arrival_probs[hr],1-E_ev_arrival_probs[hr]]))
        
    return ev_arrival_lst

def get_E_EV_values(seed, season):
    
    E_ev_arrival_probs = util.load_result('expected_values\E_ev_arrival_probs_'+season)
    E_ev_stay_t = util.load_result('expected_values\E_ev_stay_t_'+season)
    E_ev_soc_init = util.load_result('expected_values\E_ev_soc_init_'+season)
        
    E_ev_arrival_t = np.array(sample_EV_arrival(seed, E_ev_arrival_probs))
    
    E_ev_stay_t = [np.round(x) for x in E_ev_stay_t]
    E_ev_stay_t = E_ev_arrival_t * np.array(E_ev_stay_t)
    
    E_ev_soc_init = E_ev_arrival_t * np.array(E_ev_soc_init)

    
    E_required = E_ev_arrival_t * (0.97 - E_ev_soc_init) * 80
    E_ev_lax = E_ev_stay_t - E_required/11
    
    
    for idx in range(len(E_ev_arrival_t)):
        if(E_ev_arrival_t[idx] != 0):
            E_ev_arrival_t[idx] = idx
    
    E_ev_soc_init = E_ev_soc_init * 100

    ev_info_dict = {
    'ev_arrival':E_ev_arrival_t.tolist(),
    'ev_init_soc': E_ev_soc_init.tolist(),
    'ev_laxity' : E_ev_lax.tolist(),
    'ev_stay_time': E_ev_stay_t.astype(int).tolist()
    }
    
    return ev_info_dict