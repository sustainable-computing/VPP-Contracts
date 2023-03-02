import numpy as np
import cvxpy as cp

def oracle_flex(price, demand, max_storage = 0, alpha = 0, nonneg = True, verbose = False):
    assert len(price) == len(demand)
    steps = len(price)

    Z = cp.Variable(steps, nonneg=nonneg)
    constraints = []
    objective_vec = []

    storage = cp.cumsum(Z - demand)
    constraints += [storage[0] == 0]     
    constraints += [storage[1:] >= 0]
    #constraints += [storage >= 0]
    
    if max_storage != 0:
        constraints += [storage <= max_storage]

    if alpha != 0:
        #constraints += [cp.abs(Z - demand) <= alpha]
        constraints += [cp.abs(cp.diff(storage)) <= alpha]


    objective = cp.Minimize(cp.sum(cp.multiply(Z,price)) + 0.0001 * cp.sum((Z-demand)**2) )
    prob = cp.Problem(objective, constraints)

    #prob.solve(solver=cp.MOSEK, mosek_params = {'MSK_IPAR_NUM_THREADS': 8,
    #                                            'MSK_IPAR_BI_MAX_ITERATIONS': 600_000_000,
    #                                            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 100_000,
    #                                            "MSK_DPAR_PRESOLVE_TOL_X": 10**-8},
    #                                            verbose=verbose) #, max_iter = 500_000)
    #prob.solve(solver=cp.OSQP, verbose=verbose, max_iter = 2_000_000_000, eps_abs = 10**-5, eps_rel = 10**-5)
    prob.solve(solver=cp.MOSEK, verbose=verbose)
    if prob.status != 'optimal':
        #raise Exception("Optimal schedule not found")
        print("!!! Optimal solution not found")
    best_cost = prob.value
    Z_val = Z.value

    return best_cost, Z_val
