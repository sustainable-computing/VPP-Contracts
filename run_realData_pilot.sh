#!/bin/sh
# python program            suffix               types                  seed   gamma   batt kappa tau   perc   dir
#python3 real_onlineAlgo.py -S _zero_utl_newEVfewBid_tau1_v2   -Y [0.5,0.75,1,1.25,1.5]       -E 100 -G 0.01 -B 1 -K 0.2 -T 1 -P   0 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "FewBid Tau1 0 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_leftEVfewBid_tau1  -Y [0.5,0.75,1,1.25,1.5] -W -1 -E 130 -G 0.01 -B 1 -K 0.2 -T 1 -P 100 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "leftFewBid Tau1 100 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_leftEVfewBid_tau2  -Y [0.5,0.75,1,1.25,1.5] -W -1 -E 140 -G 0.01 -B 1 -K 0.2 -T 2 -P 100 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "leftFewBid Tau2 100 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_leftEVfewBid_tau3  -Y [0.5,0.75,1,1.25,1.5] -W -1 -E 151 -G 0.01 -B 1 -K 0.2 -T 3 -P 100 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "leftFewBid Tau3 100 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_unifEVallBid_tau1  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 122 -G 0.01 -B 1 -K 0.2 -T 1 -P 75 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "unifFewBid Tau1 75 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_unifEVallBid_tau2  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 104 -G 0.01 -B 1 -K 0.2 -T 2 -P 75 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "unifFewBid Tau2 75 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_unifEVallBid_tau3  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 131 -G 0.01 -B 1 -K 0.2 -T 3 -P 75 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "unifFewBid Tau3 75 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_rightEVfewBid_tau1 -Y [0.5,0.75,1,1.25,1.5] -W  1 -E 129 -G 0.01 -B 1 -K 0.2 -T 1 -P 100 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "rightFewBid Tau1 100 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_rightEVfewBid_tau2 -Y [0.5,0.75,1,1.25,1.5] -W  1 -E 109 -G 0.01 -B 1 -K 0.2 -T 2 -P 100 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "rightFewBid Tau2 100 done"  &
#python3 real_onlineAlgo.py -S _zero_utl_rightEVfewBid_tau3 -Y [0.5,0.75,1,1.25,1.5] -W  1 -E 131 -G 0.01 -B 1 -K 0.2 -T 3 -P 100 -I bids_no_solar_2019_few_sample/ -D realVisits/ &&  echo "rightFewBid Tau3 100 done"  &

python3 real_onlineAlgo.py -S _zero_utl_unifEVallBid_tau1  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 122 -G 0.01 -B 1 -K 0.2 -T 1 -P 100 -I bids_no_solar_2019/ -D realVisits/ &&  echo "unifAllBid Tau1 100 done"  &
python3 real_onlineAlgo.py -S _zero_utl_unifEVallBid_tau2  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 104 -G 0.01 -B 1 -K 0.2 -T 2 -P 100 -I bids_no_solar_2019/ -D realVisits/ &&  echo "unifAllBid Tau2 100 done"  &
python3 real_onlineAlgo.py -S _zero_utl_unifEVallBid_tau3  -Y [0.5,0.75,1,1.25,1.5] -W  0 -E 131 -G 0.01 -B 1 -K 0.2 -T 3 -P 100 -I bids_no_solar_2019/ -D realVisits/ &&  echo "unifAllBid Tau3 100 done"  &
