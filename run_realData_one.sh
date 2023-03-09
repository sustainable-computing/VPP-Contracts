#!/bin/sh
# python program            suffix               types                  seed   gamma   batt kappa tau   perc   dir
python3 real_onlineAlgo.py -S _zero_utl_unifEVsameBid_tau1  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 122 -G 0.01 -B 1 -K 0.2 -T 1 -P 75 -I bids_no_solar_2019_few_sample/ -D sameBid_realVisits/ &&  echo "unifFewBid Tau1 75 done"
