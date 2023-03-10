# VPP-Contracts

This is the accompanying repo for the (pre-print) paper **Making a Virtual Power Plant out of Privately Owned Electric Vehicles: From Contract Design to Scheduling** by Saidur Rahman, Javier Sales-Ortiz and Omid Ardakanian.

## Abstract
> With the rollout of bidirectional chargers, electric vehicle (EV) battery packs can be used in lieu of utility-scale energy storage systems to support the grid. These batteries, if aggregated and coordinated at scale, will act as a virtual power plant (VPP) that could offer flexibility and other services to the grid. To realize this vision, EV owners must be incentivized to let their battery be discharged before it is charged to the desired level. In this paper, we use contract theory to design incentive-compatible, fixed-term contracts between the VPP and EV owners. Each contract defines the maximum amount of energy that can be discharged from an EV battery and exported to the grid over a certain period of time, and the compensation paid to the EV owner upon successful execution of the contract, for reducing the cycle life of their battery. We then propose an algorithm for the optimal operation of this VPP that participates in day-ahead and balancing markets. This algorithm maximizes the expected VPP profit by taking advantage of the accepted contracts that are still valid, while honoring day-ahead commitments and fulfilling the charging demand of each EV by its deadline. We show through simulation that by offering a menu of fixed-term contracts to EVs that arrive at the charging station, trading energy and scheduling EV charging according to the proposed algorithm, the VPP profitability increases by up to 12.2%, while allow

## Overview

The code is divided into two parts:
* Simulation code for day ahead (DA) and imbalance markets (IM) operations
* Contract formulation and results analysis inside the `ResultAnalysis/` directory.

The requirements are in `requirements.txt`. Note, the convex optimization solver used was Mosek. Make sure to install the requirements and Mosek before running this code.

## Day Ahead :moneybag:
To run the day ahead, use: 

`$ python3 get_da_bids.py -P <percent of EV participation> -N [number of samples]`

For the baseline algorithm `-P` is set to 0 since no V2G participation is allowed, for all other bids `-P` is 100.

For `-N` either 10 or 100 works well.

## Imbalance market and Online Algorithm :car: :car: :car: :car: :car: :car:
This is were the bulk of the computation happens, the main program is `real_onlineAlgo.py`. However, as it takes a lot of parameters, it is best to use a script such as `run_realData_pilot.sh` or `run_realData_one.sh`.

Usage: 

`$ source run_realData_pilot.sh`

The parameters are:

| Parameter     | Description |
|---------------|-------------|
| -T, --Tau     | (aka $\ell_{V2G}$ in the paper) The length of time which a contract is valid |
| -S, --suffix  | Suffix for the results file. | 
| -Y, --types   | What types to consider, this argument is given as a list |
| -P, --perc    | Percentage of vehicles that get a contract menu, 100 in all of our experiments |
| -K, --Kappa   | Coefficient for the VPP's utility function (in $U_{VPP}$) | 
| -G, --Gamma   | Coefficient for the EV's battery degradation (in $U_{EV}$) | 
| -B, --bat_deg | Coefficient for battery degradation, set to 1 since this behaviour is already captured in gamma $\gamma$ (This is the equivalent of $\gamma \cdot c$ in the paper) | 
| -D, --res_dir | Results directory, the directory where the program writes |
| -I, --bid_dir | The directory from where to read the DA bids |
| -E, --seed    | Seed for the random number generator |
|-W, --skewed   | If the type distribution is skewed (or biased), only implemented for 5 types. Default is 0. Results not present in paper |

:hourglass: Running time is around 1 hour per experiment and only a single core is used.

### Other files :cloud_with_lightning:
* **deterministic_solver_ev_penetration_no_solar.py**: This is the solver used for Eqn. 13, the DA da bids.
* **ev_data_sampler.py**: This is used for sampling expected EV arrivals for the scenarios in Eqn. 13.
* **generate_expected_EV_values.py**: This is used for sampling other values for the scenarios in Eqn. 13.
* **ev_scheduler.py**: This is used to solve Eqn. 16.
* **real_funcOnlineAlgo.py**: Here are some functions for `real_onlineAlgo.py`

## Contract formulation and Results Analysis

### Contract :handshake:
`contracts.py` This is where the optimization problem for contracts is solved, that is Eqn. 8. 

### 1OptimizationRuntime :chart_with_upwards_trend:
This notebook measures and compares the runtime of solving Eqn. 7 & 8

### 2OptimalContracts
This notebook had previous drafts for the analysis of optimal contracts. This notebook can be ignored.

### 3ContractValues :balance_scale:
Notebook for Figure 2 and Figure 5. Also, investigates the behaviour of Eqn. 8 under different parameters. 

### 4ReloadFigSched
Previous draft for the results of the online algorithm. This notebook can be ignored.

### 5PriceElectricity
Short notebook on exploratory data analysis (EDA) of electricity prices. This notebook can be ignored.

### 6DatasetsEDA
A longer notebook on EDA, also including EV arrivals. This notebook can be ignored

### 7PreprocessElaad :cityscape:
Here the Elaad dataset is preprocessed as decribed on section 7.1 of the paper

### 8NewBids :bar_chart:
Here figures 3 and 4 are created. The results for many different runs of the online algorithm are analyzed. 


### Other files :owl:
* **sensitivity.py**: Helper functions for notebooks 2 and 3
* **analysis_tools.py**: Helper functions for notebook 8

