# GhostBuster

## About the code
* `ghost_buster.py`: Main script to run the EM and find ghost populations
* `plotting/`: Supporting plotting scripts to get population size plots, PC, local ancestry accuracy

## Installation
```
conda create -n gb python=3.11 -y
conda activate gb
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## List of options
```
python ghost_buster.py --help
```

## Examples
* Running GhostBuster:

```
python ghost_buster.py \
    --genome_build hg37 \
    --trees example/stdpopsim_homsap_chr \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 51 \
    --chr 22 \
    --output example/stdpopsim_homsap
```