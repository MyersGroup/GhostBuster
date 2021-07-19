# RelateLocalAncestry
2nd rotation with Prof. Simon Myers: An unsupervised algorithm to find ghost populations using genome-wide genealogies.

## About the code
* `sgdp_no_groundtruth.py` to generate coalescence rate plots on SGDP samples - for example understanding the history for Mbuti samples
* `sgdp_true_likelihood.py` to run the simulated AFR/EAS and EUR/EAS split cases using SGDP 
* `em_true_likelihood.py` has the code for the first simulation (6 populations and an admixed population with 1:1:2 admixture proportions from population A, B and C)
* `em_true_ghost.py` the first simulation but running with ghost groups or proxy groups
* `em_true_ancient.py` is the code for the second simulation on western eurasians

## Demo code
* `python em_true_ancient_sim_subsampled.py --chr 1,2 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas True --window_size 0 --sample_id 0 --path /well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sim_debug/transfer/transfer/input/ --trees SGDP_archaic_v1_EAS`