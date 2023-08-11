#!/bin/sh
# python program            suffix         types                  skewed    seed   gamma batt  kappa  tau   perc   dir
python3 nov2g_onlineAlgo.py -S _nov2g_disAlphaD_tau1  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 122 -G 0.01 -B 1 -K 0.2 -T 1 -P 100 -I bids_no_solar_2019/ -D noV2G/ &&  echo "nov2g Tau1 100 done"  &
python3 nov2g_onlineAlgo.py -S _nov2g_disAlphaD_tau2  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 104 -G 0.01 -B 1 -K 0.2 -T 2 -P 100 -I bids_no_solar_2019/ -D noV2G/ &&  echo "nov2g Tau2 100 done"  &
python3 nov2g_onlineAlgo.py -S _nov2g_disAlphaD_tau3  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 131 -G 0.01 -B 1 -K 0.2 -T 3 -P 100 -I bids_no_solar_2019/ -D noV2G/ &&  echo "nov2g Tau3 100 done"  &
