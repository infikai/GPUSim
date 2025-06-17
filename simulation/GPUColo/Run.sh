#!/bin/bash

cd data
sh clean_data.sh
cd ..

source /mydata/Data/gpucolo/venv/bin/activate

python3 run.py -g 50000 -slo 0.9 -d /mydata/Data/gpucolo/csv/dfas_burst_D70000_U20.csv
