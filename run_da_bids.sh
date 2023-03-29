#!/bin/sh
python3 get_da_bids.py -P 0   && echo "Bids 100 sample  00 done" &
python3 get_da_bids.py -P 25  && echo "Bids 100 sample  25 done" &
python3 get_da_bids.py -P 50  && echo "Bids 100 sample  50 done" &
python3 get_da_bids.py -P 75  && echo "Bids 100 sample  75 done" &
python3 get_da_bids.py -P 100 && echo "Bids 100 sample 100 done" &

