# RelateLocalAncestry
2nd rotation with Prof. Simon Myers: An unsupervised algorithm to find ghost populations using genome-wide genealogies.

## About the code
* `sgdp_no_groundtruth.py` to generate coalescence rate plots on SGDP samples - for example understanding the history for Mbuti samples
* `sgdp_true_likelihood.py` to run the simulated AFR/EAS and EUR/EAS split cases using SGDP 
* `em_true_likelihood.py` has the code for the first simulation (6 populations and an admixed population with 1:1:2 admixture proportions from population A, B and C)
* `em_true_ghost.py` the first simulation but running with ghost groups or proxy groups
* `em_true_ancient.py` is the code for the second simulation on western eurasians

## Demo code
* `python em_true_ancient_sim_subsampled.py --chr 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas True --window_size 0 --sample_id 24 --path /well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sim_debug/transfer/transfer/input/ --rec /well/myers/users/ooz218/workspace/msprime_maps/genetic_map_GRCh37 --trees SGDP_archaic_v1_EAS --init_at_truth False --num_iters 10`

* `python em_true_ancient_sim_subsampled.py --chr 1,2 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas True --window_size 0 --sample_id 8 --init_at_truth 0 --path /well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sim_debug/test/ --trees SGDP_archaic -k 4 -i 1000 --load_gamma gamma_8_iter41.npy --load_props 0.03973251 0.00391422 0.7170701 0.23928318 --rec /well/myers/users/ooz218/workspace/msprime_maps/genetic_map_GRCh37 

* `python em_true_ancient_sim_subsampled.py --chr 20 --relate_trees True --masking_thresh 0.5 --plot_intermediate_gammas False --window_size 0 --sample_id 8 9 10 --init_at_truth 0 --path /well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/input_Papuan_th0.5_mult/ --trees SGDP_archaic -k 4 -i 10 --rec /well/myers/users/ooz218/workspace/msprime_maps/genetic_map_GRCh37 -start_time 4.7 -end_time 6 -ignore_first_epoch True -o Papuan > log.txt `