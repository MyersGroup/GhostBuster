#!/bin/bash
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/new_venv/bin/activate

for chr in {1..22}
do
  python simulate.py ${chr} &
done
