# VPP-Contracts

This is the accompanying repo for the (pre-print) paper **Making a Virtual Power Plant out of Privately Owned Electric Vehicles: From Contract Design to Scheduling** by Saidur Rahman, Javier Sales-Ortiz and Omid Ardakanian.

## Abstract
```
With the rollout of bidirectional chargers, electric vehicle (EV) battery packs can be used in lieu of utility-scale energy storage systems to support the grid. These batteries, if aggregated and coordinated at scale, will act as a virtual power plant (VPP) that could offer flexibility and other services to the grid. To realize this vision, EV owners must be incentivized to let their battery be discharged before it is charged to the desired level. In this paper, we use contract theory to design incentive-compatible, fixed-term contracts between the VPP and EV owners. Each contract defines the maximum amount of energy that can be discharged from an EV battery and exported to the grid over a certain period of time, and the compensation paid to the EV owner upon successful execution of the contract, for reducing the cycle life of their battery. We then propose an algorithm for the optimal operation of this VPP that participates in day-ahead and balancing markets. This algorithm maximizes the expected VPP profit by taking advantage of the accepted contracts that are still valid, while honoring day-ahead commitments and fulfilling the charging demand of each EV by its deadline. We show through simulation that by offering a menu of fixed-term contracts to EVs that arrive at the charging station, trading energy and scheduling EV charging according to the proposed algorithm, the VPP profitability increases by up to 12.2%, while allow
```

## Overview

The code is divided into two parts:
* Simulation code for day ahead and imbalance operations
* Contract formulation and results analysis inside the `ResultAnalysis/` directory.

The requirements are in `requirements.txt`. Note, the convex optimization solver used was Mosek. Make sure to install the requirements and Mosek before running this code.

## Day Ahead
To run the day ahead, use `python3 u`
