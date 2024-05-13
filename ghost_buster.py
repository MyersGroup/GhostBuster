import pandas as pd
import numpy as np
import math
import time
import tskit
import argparse
import copy
from tqdm import tqdm
import random
from functools import partial
from scipy.stats import hmean
import scipy.stats as stats
from joblib import Parallel, delayed
import gc

from calc_tree_stats import load_tree_stats
from calc_fixed_params import fixed_parameters, load_fixed_params
from calc_ground_truth import get_groundtruth_reference, make_ground_truth
from utils import (
    make_one_hot,
    filter_recomb_rate,
    filter_bstat,
    load_mask_csv,
    write_coal,
    write_calibration,
    calculate_accuracy,
    boolean,
    compute_gamma_num,
    compute_gamma_denom,
    load_gamma,
    load_props,
    get_target_branch_length,
    make_numba_nested_list,
)
import pdb
import warnings
from hmm_decode import Decode_grid
from numba import jit
import numba as nb
import networkx as nx 
import glob
import os 
import msprime
import bisect
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def input_assertions(args):
    if not args.evaluate_local_ancestry and not args.evaluate_gamma:
        raise ValueError(
            "If you don't want to evaluate the population sizes or local ancestry, what are you here for ?"
        )
    if args.init_at_truth and args.mode == "real":
        raise ValueError(
            "How can you initialize your local ancestry in real-world data ?"
        )
    if args.props_per_chrs and args.load_props:
        raise ValueError(
            "Propotions per chromosomes not supported when you load proportions"
        )
    if args.mode == "sim" and args.ground_truth_path is None:
        raise ValueError(
            "Supply the location of ground_truth file or run in mode = real"
        )
    if args.opportunity_filter and (args.mutden is None or args.allmuts is None):
        raise ValueError(
            "Supply the location for mutden and allmuts file to filter trees based on opportunity"
        )


def load_trees(args):
    chrs = list(map(int, args.chrs.split(",")))
    ts_list = []
    if args.trees != None:
        for chr in chrs:
            ts = tskit.load(args.trees + str(chr) + ".trees")  ## relate trees
            ts_list.append(ts)

    return ts_list


def update_membership_eventwise_denom(gamma_arr, denom, ignore_first_epoch, ignore_last_epoch, n_epochs):
    log_denom_em = np.zeros((gamma_arr.shape[0], denom.shape[0]), dtype="float64")
    start_end_mask = np.ones(n_epochs - 1, dtype=bool)
    start_end_mask[0] = False if ignore_first_epoch else True
    start_end_mask[-1] = False if ignore_last_epoch else True
    for j in range(gamma_arr.shape[0]):
        ### Caution: This hack removes epochs which have any one reference population without coal. rate
        ### We instead want to remove the reference population not the entire epoch
        nan_mask = np.isnan(gamma_arr[j]).sum(axis=0) == 0
        combined_mask = np.logical_and(start_end_mask, nan_mask)
        gamma_arr_nan_removed = gamma_arr[j][:, combined_mask]
        denom_nan_removed = denom[:, :,  combined_mask]
        log_denom_em[j] = -np.sum(gamma_arr_nan_removed * denom_nan_removed, axis=(1,2))
    return log_denom_em
   

@jit(nopython=True, fastmath=True)
def update_membership_eventwise_numba(
    proportion_of_coalescing_all,
    epoch_index_all,
    gamma_arr,
    ignore_first_epoch,
    ignore_last_epoch,
    n_epochs,
    target_branch_length,
    num_clusters,
):
    log_num_em = np.zeros((num_clusters, len(proportion_of_coalescing_all)), dtype="float64")
    for n_site in range(len(proportion_of_coalescing_all)):
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[n_site]
        epoch_index_in_tree = epoch_index_all[n_site]
        target_branch_length_in_tree = target_branch_length[n_site]
        for j in range(num_clusters):
            log_num_em_j = 0.0
            for i in range(len(proportion_of_coalescing_in_tree)):
                if (not ignore_first_epoch) or epoch_index_in_tree[i] >= 1:
                    if (not ignore_last_epoch) or epoch_index_in_tree[i] < n_epochs - 2:
                        if not (gamma_arr[j][:, epoch_index_in_tree[i]] == -9223372036854775808).any():
                            log_num_em_j += (
                                np.log(
                                    sum(
                                        gamma_arr[j][:, epoch_index_in_tree[i]]
                                        * proportion_of_coalescing_in_tree[i]
                                    )
                                    / sum(proportion_of_coalescing_in_tree[i]),
                                )
                            ) / target_branch_length_in_tree[i]
            log_num_em[j, n_site] = log_num_em_j
    return log_num_em

def update_membership_eventwise_numpy(
    proportion_of_coalescing_all,
    epoch_index_all,
    gamma_arr,
    ignore_first_epoch,
    ignore_last_epoch,
    n_epochs,
    target_branch_length,
    num_clusters,
):
    log_num_em = np.zeros((num_clusters, len(proportion_of_coalescing_all)), dtype="float64")
    
    for n_site in range(len(proportion_of_coalescing_all)):
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[n_site]
        epoch_index_in_tree = epoch_index_all[n_site]
        target_branch_length_in_tree = target_branch_length[n_site]
        for j in range(num_clusters):
            log_num_em_j = 0.0
            for i in range(len(proportion_of_coalescing_in_tree)):
                if (not ignore_first_epoch) or epoch_index_in_tree[i] >= 1:
                    if (not ignore_last_epoch) or epoch_index_in_tree[i] < n_epochs - 2:
                        if np.isnan(gamma_arr[j][:, epoch_index_in_tree[i]]).any():
                            continue
                        log_num_em_j += (
                            np.log(
                                np.sum(
                                    gamma_arr[j][:, epoch_index_in_tree[i]]
                                    * proportion_of_coalescing_in_tree[i]
                                )
                                / sum(proportion_of_coalescing_in_tree[i]),
                            )
                        ) / target_branch_length_in_tree[i]
            log_num_em[j, n_site] = log_num_em_j
    return log_num_em


def combine_local_ancestry(arr, n):
    ## combines the arr values every n elements
    ## Input: arr of shape c x N => output of shape c x N/n
    out = np.zeros((arr.shape[0], arr.shape[1] // n))
    for i in range(n):
        out += arr[:, i::n]
    return out / n


def regress_out_mean(y):
    y_out = np.zeros_like(y)
    sum_all = np.sum(y)
    for i in range(len(y)):
        y_out[i] = y[i] - (sum_all - y[i]) / (len(y) - 1)

    return y_out
    # corr_bt = np.array(C > 0.3, dtype='int')
    # g = nx.from_numpy_array(corr_bt)
    # mean_lowvar = 0.0
    # count = 0
    # for _ in range(10):
    #     indep_set = nx.maximal_independent_set(g)
    #     if len(indep_set) > 1:
    #         count += len(indep_set)
    #         mean_lowvar += np.sum(y[indep_set])
    # if count > 0:
    #     mean_lowvar /= count
    # print(np.linalg.norm(mean_lowvar))
    # return (y - mean_lowvar)

    # y = y.reshape(-1, 1)
    # X = np.ones_like(y)
    # try:
    #     cov_inv = np.linalg.inv(C + 1e-8*np.eye(len(C)))
    # except:
    #     pdb.set_trace()
    # X_transpose_cov_inv = np.dot(X.T, cov_inv)
    # X_cov_X_transpose_cov_inv = np.dot(X_transpose_cov_inv, X)
    # X_cov_X_transpose_cov_inv_inv = np.linalg.inv(X_cov_X_transpose_cov_inv)
    # g_mle = np.dot(X_cov_X_transpose_cov_inv_inv, np.dot(X_transpose_cov_inv, y))
    # return (y - np.dot(X, g_mle)).flatten()


def e_m_step(
    args,
    own_membership,  ##now is n_clusters x n_sites
    trans_prop,
    tau,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    unique_groups,
    epoch_intervals,
    n_sites,
    n_samples,
    epoch,
    target_branch_length_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    loglikehood_cov=None,
    regressors=None
):
    st = time.time()
    n_epochs = len(epoch_intervals)
    n_unique_groups = len(unique_groups)
    
    if args.regress_out:
        for i in range(len(n_sites)):
            assert n_sites[i] == n_sites[0]
        n_sites_common = n_sites[0]

    if args.load_gamma != None:
        print("Using initial gamma specified in file: " + str(args.load_gamma))
        gamma_arr = load_gamma(args.load_gamma, args.groups, unique_groups)
        ## CAUTION: recheck this, removing gamma_arr outside the range
        for epoch in range(gamma_arr.shape[2]):
            if args.ignore_first_epoch and (args.start_time - math.log(args.ypg, 10)) > epoch_intervals[epoch]:
                gamma_arr[:,:,epoch] = np.nan
            if args.ignore_last_epoch and (args.end_time - math.log(args.ypg, 10)) < epoch_intervals[epoch+1]:
                gamma_arr[:,:,epoch] = np.nan
    if args.load_gamma is None or epoch == args.num_iters - 1:
        n = np.zeros(
            (args.num_clusters, n_unique_groups, n_epochs - 1),
            dtype="float64",
        )
        d = np.zeros(
            (args.num_clusters, n_unique_groups, n_epochs - 1),
            dtype="float64",
        )
        if prev_gamma is None:
            prev_gamma = np.ones_like(n)
        prev_gamma = np.nan_to_num(prev_gamma, nan=1)    
        for sample_no in range(n_samples):
            start = int(np.sum(n_sites[0:sample_no]))
            end = int(np.sum(n_sites[0:sample_no+1]))
            own_membership_sample = own_membership[
                :, start:end
            ]
            for j in range(len(own_membership_sample)):
                n_j = compute_gamma_num(
                    own_membership_sample[j],
                    prev_gamma[j],
                    proportion_of_coalescing_all[sample_no],
                    epoch_index_all[sample_no],
                    n_unique_groups,
                    n_epochs,
                    target_branch_length_masked[sample_no],
                    args.ignore_first_epoch,
                    args.ignore_last_epoch,
                )
                d_j = compute_gamma_denom(
                    own_membership_sample[j],
                    denom[sample_no],
                )
                n[j] += n_j
                d[j] += d_j
        if args.load_gamma is None:
            gamma_arr = n / d
    if args.load_props != None:
        print("Using initial props specified in file: " + str(args.load_props))        
        tau = load_props(
            args.load_props
        )  ### load taus only works for not(props_per_chrs)

    print("M-step time: " + str(time.time() - st))
    st = time.time()

    if args.t_admix_guess is not None:
        trans_prop = args.t_admix_guess

    gamma_arr = np.maximum(gamma_arr, 0)
    prev_gamma = copy.deepcopy(gamma_arr)
    if np.isnan(gamma_arr).any():
        print(gamma_arr)

    log_num_em = np.zeros((args.num_clusters, np.sum(n_sites)), dtype="float64")
    log_denom_em = np.zeros((args.num_clusters, np.sum(n_sites)), dtype="float64")
    count_masked_trees = 0

    update_membership_eventwise = update_membership_eventwise_numba
    for sample_no in range(n_samples):
        log_num_em_sam = update_membership_eventwise(
            proportion_of_coalescing_all[sample_no],
            epoch_index_all[sample_no],
            gamma_arr,
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
            target_branch_length_masked[sample_no],
            args.num_clusters,
        )
        log_denom_em_sam = update_membership_eventwise_denom(
            gamma_arr,
            denom[sample_no],
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
        )
        start = int(np.sum(n_sites[0:sample_no]))
        end = int(np.sum(n_sites[0:sample_no+1]))
        log_num_em[:, start:end] = log_num_em_sam
        log_denom_em[
            :, start:end
        ] = log_denom_em_sam
    log_num_em = 1 * combine_local_ancestry(log_num_em, args.num_subtrees)
    log_denom_em = 1 * combine_local_ancestry(log_denom_em, args.num_subtrees)

    loglikelihood_per_comp = log_num_em + log_denom_em
    print("log-like time: " + str(time.time() - st))

    ## no regress out on last epochs
    if args.regress_out:
        st = time.time()
        loglikehood_base = loglikelihood_per_comp[0].sum().copy()
        loglikelihood_per_comp_copy = loglikelihood_per_comp.copy()
        for k in range(1, args.num_clusters):
            loglikelihood_per_comp_diff = (
                loglikelihood_per_comp_copy[k] - loglikelihood_per_comp_copy[0]
            )
            loglikelihood_per_comp_diff = loglikelihood_per_comp_diff.reshape(
                n_samples, n_sites_common
            )
            loglikelihood_per_comp_diff_before = loglikelihood_per_comp_diff.copy()
            ## regress-out the mean of the loglikelihood
            for sam in range(n_samples):
                mean_logl_minus_sample = np.mean(loglikelihood_per_comp_diff_before[np.arange(n_samples) != sam], axis=0)
                mean_logl_minus_sample = mean_logl_minus_sample.reshape(-1, 1)

                if regressors is not None:
                    mean_logl_minus_sample = regressors[sam] #np.hstack([mean_logl_minus_sample, regressors[sam]])
                else:
                    mean_logl_minus_sample = mean_logl_minus_sample

                mean_logl_minus_sample_meansub = sm.add_constant(mean_logl_minus_sample-np.mean(mean_logl_minus_sample, axis=0))
                model = sm.OLS(loglikelihood_per_comp_diff[sam],mean_logl_minus_sample_meansub)
                results = model.fit()
                if sam == 0:
                    print(results.summary())
                beta = results.params
                loglikelihood_per_comp_diff[sam] = loglikelihood_per_comp_diff[sam] - mean_logl_minus_sample_meansub[:,1:]@beta[1:] ##ignore the intercept

            loglikelihood_per_comp_diff = loglikelihood_per_comp_diff.reshape(
                n_sites_common * n_samples
            )
            
            loglikelihood_per_comp[k] = loglikelihood_per_comp_diff
        loglikelihood_per_comp[0] = 0
        print("Time taken for regressing out: " + str(time.time() - st))

    ### HMM smoothing
    st = time.time()
    own_membership_hmm = np.zeros((args.num_clusters, np.sum(n_sites)), dtype="float64")
    for sample_no in range(n_samples):
        start = int(np.sum(n_sites[0:sample_no]))
        end = int(np.sum(n_sites[0:sample_no+1]))
        own_membership_sam, trans_num_sam, trans_denom_sam, tau_sam, log_likelihood_sam = Decode_grid(
            tree_left_bp[sample_no],
            tree_right_bp[sample_no],
            gen_grid_all[sample_no],
            trans_prop,
            loglikelihood_per_comp[:, start:end],
            tau,
            window_size=args.force_build,
        )
        own_membership_hmm[:, start:end] = own_membership_sam 
        if sample_no == 0:
            trans_num = trans_num_sam
            trans_denom = trans_denom_sam
            tau_update = tau_sam
            log_likelihood_hmm = log_likelihood_sam
        else:
            trans_num += trans_num_sam
            trans_denom += trans_denom_sam
            tau_update += tau_sam 
            log_likelihood_hmm += log_likelihood_sam

    trans_prop = trans_num/trans_denom
    tau = tau_update/np.sum(tau_update)
    tau = np.minimum(np.maximum(tau, 1e-11), 1-1e-11)

    own_membership_hmm = np.repeat(own_membership_hmm, args.num_subtrees, axis=1)
    if args.regress_out:
        log_likelihood_hmm += loglikehood_base
    
    print((tau, np.mean(own_membership_hmm, axis= 1)))
    print("HMM time: " + str(time.time() - st))
    if np.isnan(log_likelihood_hmm):
        pdb.set_trace()
    
    if epoch == args.num_iters - 1:
        gamma_arr = n/d

    return own_membership_hmm, trans_prop, gamma_arr, tau, log_likelihood_hmm


def estimate_gt_ref(
    gamma_arr,
    tau,
    args,
    ts_list,
    poplabels,
    n_trees,
    n_epochs,
    mask_dodgy,
    epoch_intervals_pow,
    target_branch_length_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all
):
    ## random init gt_ref & unique_groups

    poplabels_included = poplabels[poplabels.INCLUDE == 1]
    unique_groups, i = {}, 0
    for group in np.unique(poplabels_included.GROUP):
        if poplabels_included.GROUP.iloc[args.sample_id[0]] == group:
            for c in range(args.num_clusters):
                unique_groups[group + str(c + 1)] = i
                i += 1
        else:
            unique_groups[group] = i
            i += 1

    gt_ref = np.zeros((len(poplabels.GROUP), n_trees), dtype="object")
    for sample_no, sample in enumerate(poplabels.index):
        group = poplabels.GROUP.loc[sample]
        if group == poplabels.GROUP.loc[args.sample_id[0]]:
            gt_ref[sample_no] = {
                poplabels.GROUP.iloc[sample] + str(c + 1): tau[c]
                for c in range(args.num_clusters)
            }
        else:
            try:
                gt_ref[sample_no] = unique_groups[group]
            except:
                gt_ref[sample_no] = "NA"

    chrs = list(map(int, args.chrs.split(",")))

    gt_ref = np.array(gt_ref, dtype="object")
    gt_ref_update = np.zeros_like(gt_ref, dtype="object")
    for outer_iter in range(1):
        r2 = 0.0
        for sample_no, sample in enumerate(
            poplabels[
                (poplabels.GROUP == poplabels.GROUP.iloc[args.sample_id[0]])
                & poplabels.INCLUDE
                == 1
            ].index
        ):
            ## Calc fixed params
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                n_trees,
                mask_dodgy,
                sample,
                args.sample_id,
                epoch_intervals_pow,
                target_branch_length_masked[sample_no],
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=gt_ref,
                ignore_first_epoch=args.ignore_first_epoch,
            )
            ## E-step to infer local ancestry
            own_membership_trial = np.ones(
                (args.num_clusters, n_trees), dtype="float64"
            )
            # update_membership_eventwise = update_membership_eventwise_numpy if np.isnan(gamma_arr).any() else update_membership_eventwise_numba
            update_membership_eventwise = update_membership_eventwise_numba 
            log_num_em, log_denom_em = update_membership_eventwise(
                proportion_of_coalescing_all1,
                epoch_index_all1,
                denom1,
                gamma_arr,
                args.ignore_first_epoch,
                args.ignore_last_epoch,
                n_epochs,
                target_branch_length_masked[sample_no],
                np.arange(n_trees),
                args.num_clusters,
                n_trees,
            )

            loglikelihood_per_comp = log_num_em + log_denom_em
            own_membership_trial = np.exp(
                log_num_em
                + log_denom_em
                - np.repeat(
                    np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
                    args.num_clusters,
                    axis=1,
                ).T
            )
            own_membership_trial = np.nan_to_num(own_membership_trial, nan=1)
            for j in range(args.num_clusters):
                own_membership_trial[j] *= tau[j]

            own_membership_trial = own_membership_trial / (
                np.sum(own_membership_trial, axis=0)
            )

            ## pass through HMM
            own_membership_trial, trans_prop, tau, _ = Decode_grid(
                tree_left_bp,
                tree_right_bp,
                gen_grid_all,
                t_admix,
                loglikelihood_per_comp,
                tau,
                window_size=args.force_build,
                per_tree_output=True,
            )

            # Sample from the posteriors
            for n_t in range(n_trees):
                gt_ref_update[sample, n_t] = {
                    poplabels.GROUP.iloc[sample]
                    + str(c + 1): own_membership_trial[c, n_t]
                    for c in range(args.num_clusters)
                }
        for sample in tqdm(
            np.random.permutation(
                poplabels[
                    (poplabels.GROUP == poplabels.GROUP.loc[args.sample_id[0]])
                ].index
            )
        ):
            gt_ref[sample] = copy.deepcopy(gt_ref_update[sample])

    return gt_ref, unique_groups


def random_sweep_iter(
    args,
    n_clusters,
    unique_groups,
    n_epochs,
    num_trees_per_sample,
    n_iters,
    ts_list,
    poplabels,
    mask_dodgy,
    epoch_intervals_pow,
    num,
    denom,
    proportion_of_coalescing_all,
    epoch_index_all,
    target_branch_length_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    chr_map,
    loglikehood_cov=None,
    regressors_all=None
):
    n_sites = []
    for sample_no in range(len(args.sample_id)):
        n_sites_sam = 0
        for i, (l, r) in enumerate(zip(tree_left_bp[sample_no], tree_right_bp[sample_no])):
            n_sites_sam += int(r/args.force_build) - int(l/args.force_build)
        n_sites.append(copy.deepcopy(n_sites_sam))

    epoch_intervals = np.log10(epoch_intervals_pow)
    
    n_unique_groups = len(unique_groups)
    if args.load_gamma is not None:
        gamma_arr = load_gamma(args.load_gamma, args.groups, unique_groups)
        ## CAUTION: recheck this, removing gamma_arr outside the range
        for epoch in range(gamma_arr.shape[2]):
            if args.ignore_first_epoch and (args.start_time - math.log(args.ypg, 10)) > epoch_intervals[epoch]:
                gamma_arr[:,:,epoch] = np.nan
            if args.ignore_last_epoch and (args.end_time - math.log(args.ypg, 10)) < epoch_intervals[epoch+1]:
                gamma_arr[:,:,epoch] = np.nan
    else:
        if args.joint_fit:
            n_unique_groups = n_unique_groups + 1*(n_clusters - 1)

        gamma_arr = np.power(
            np.e,
            np.random.uniform(
                -16.11, -5.3, (n_clusters, n_unique_groups, n_epochs - 1)
            ),
        )
        gamma_arr = np.array(gamma_arr, dtype="float64")

    if args.load_props is not None:  
        tau = load_props(args.load_props)

    else:
        tau = np.random.uniform(0.01, 0.99, n_clusters)
        tau /= np.sum(tau)

    trans_prop = (
        np.power(10, np.random.uniform(np.log10(20), np.log10(2000)))
        if args.t_admix_guess is None
        else args.t_admix_guess
    )
    print((tau, trans_prop))
    n_samples = len(args.sample_id)
    if args.joint_fit:
        # gt_ref, unique_groups = estimate_gt_ref(
        #     gamma_arr,
        #     tau,
        #     args,
        #     ts_list,
        #     poplabels,
        #     n_trees,
        #     n_epochs,
        #     mask_dodgy,
        #     epoch_intervals_pow,
        #     target_branch_length_masked,
        #     tree_left_bp,
        #     tree_right_bp,
        #     tree_left_bp_gen,
        #     tree_right_bp_gen,
        # )
        unique_groups, i = {}, 0
        poplabels_included = poplabels[poplabels.INCLUDE == 1]
        for group in np.unique(poplabels_included.GROUP):
            # if group == 'WestEurasia' or group == 'Papuan' or group == 'Cambodian' or group == 'Han' or group == 'Korean' or group == 'Japanese' or group == 'Burmese' or group == 'Thai':
            if group == 'GBR':
                for c in range(args.num_clusters):
                    unique_groups[group + str(c + 1)] = i
                    i += 1
            else:
                unique_groups[group] = i
                i += 1

        assert (num_trees_per_sample == np.mean(num_trees_per_sample)).all()
        gt_ref = np.zeros((len(poplabels.GROUP), int(np.mean(num_trees_per_sample))), dtype="object")
        for sample_no, sample in enumerate(poplabels.index):
            group = poplabels.GROUP.loc[sample]
            # if group == 'WestEurasia' or group == 'Papuan' or group == 'Cambodian' or group == 'Han' or group == 'Korean' or group == 'Japanese' or group == 'Burmese' or group == 'Thai':
            if group == 'GBR':
                gt_ref[sample_no] = {
                    poplabels.GROUP.iloc[sample] + str(c + 1): 1/(args.num_clusters)
                    for c in range(args.num_clusters)
                }
            else:
                try:
                    gt_ref[sample_no] = unique_groups[group]
                except:
                    gt_ref[sample_no] = "NA"
        
        
        # for file_name in glob.glob('output_detailed1/eas_regressed_overall_membership*csv'):
        #     print(file_name)
        #     sample = int(file_name.split('output_detailed1/eas_regressed_overall_membership')[1].split('sample_id_')[1].split('.csv')[0])
        #     local_anc = pd.read_csv(file_name, sep='\s+')
        #     for n_t in range(n_trees):
        #         chr_ = chr_map_masked[n_t]
        #         local_anc_chr = local_anc[local_anc['chr'] == chr_]
        #         tree_genpos = 100*(tree_left_bp_gen[n_t] + tree_right_bp_gen[n_t])/2
        #         local_anc_nt = local_anc_chr.iloc[(local_anc_chr['genpos']-tree_genpos).abs().argsort()[:1]].values[:, 3:].flatten()
        #         gt_ref[sample, n_t] = {
        #             poplabels.GROUP.iloc[sample]
        #             + str(c + 1): local_anc_nt[c]
        #             for c in range(args.num_clusters)
        #         }

        # for file_name in glob.glob('output_detailed1/eur_regressed_overall_membership*csv'):
        #     print(file_name)
        #     sample = int(file_name.split('output_detailed1/eur_regressed_overall_membership')[1].split('sample_id_')[1].split('.csv')[0])
        #     local_anc = pd.read_csv(file_name, sep='\s+')
        #     for n_t in range(n_trees):
        #         chr_ = chr_map_masked[n_t]
        #         local_anc_chr = local_anc[local_anc['chr'] == chr_]
        #         tree_genpos = 100*(tree_left_bp_gen[n_t] + tree_right_bp_gen[n_t])/2
        #         local_anc_nt = local_anc_chr.iloc[(local_anc_chr['genpos']-tree_genpos).abs().argsort()[:1]].values[:, 3:].flatten()
        #         gt_ref[sample, n_t] = {
        #             poplabels.GROUP.iloc[sample]
        #             + str(c + 1): local_anc_nt[c]
        #             for c in range(args.num_clusters)
        #         }

        # for file_name in glob.glob('output_nea/nea_debug_overall_membership*csv'):
        #     print(file_name)
        #     sample = int(file_name.split('output_nea/nea_debug_overall_membership')[1].split('sample_id_')[1].split('.csv')[0])
        #     local_anc = pd.read_csv(file_name, sep='\s+')
        #     for n_t in range(n_trees):
        #         chr_ = chr_map_masked[n_t]
        #         local_anc_chr = local_anc[local_anc['chr'] == chr_]
        #         tree_genpos = 100*(tree_left_bp_gen[n_t] + tree_right_bp_gen[n_t])/2
        #         local_anc_nt = local_anc_chr.iloc[(local_anc_chr['genpos']-tree_genpos).abs().argsort()[:1]].values[:, 3:].flatten()
        #         gt_ref[sample, n_t] = {
        #             poplabels.GROUP.iloc[sample]
        #             + str(c + 1): local_anc_nt[c]
        #             for c in range(args.num_clusters)
        #         }
    else:
        gt_ref = None
    own_membership_trial = [[] for _ in range(n_clusters)]
    log_num_em = np.zeros((n_clusters, int(np.sum(n_sites))), dtype="float64")
    log_denom_em = np.zeros((n_clusters, int(np.sum(n_sites))), dtype="float64")
    count_masked_trees = 0

    # update_membership_eventwise = update_membership_eventwise_numpy if np.isnan(gamma_arr).any() else update_membership_eventwise_numba
    update_membership_eventwise = update_membership_eventwise_numba
    for sample_no, sample in enumerate(args.sample_id):
        if args.joint_fit:
            # (
            #     num1,
            #     denom1,
            #     proportion_of_coalescing_all1,
            #     epoch_index_all1,
            # ) = fixed_parameters(
            #     ts_list,
            #     poplabels,
            #     unique_groups,
            #     n_trees,
            #     mask_dodgy,
            #     sample,
            #     args.sample_id,
            #     epoch_intervals_pow,
            #     target_branch_length_masked[sample_no],
            #     args.force_build,
            #     args.num_subtrees,
            #     args.max_per_group,
            #     gt_ref=gt_ref,
            #     ignore_first_epoch=args.ignore_first_epoch,
            # )
            # num.append(num1)
            # denom.append(denom1)
            # proportion_of_coalescing_all.append(make_numba_nested_list(proportion_of_coalescing_all1))
            # epoch_index_all.append(make_numba_nested_list(epoch_index_all1))
            
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = load_fixed_params(args, ts_list, sample, poplabels, mask_dodgy[sample_no], chr_map, epoch_intervals, target_branch_length_masked[sample_no], gt_ref=gt_ref, unique_groups=unique_groups)
            num.append(num1)
            denom.append(denom1)
            proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
            epoch_index_all.append(epoch_index_all1)

        log_num_em_sam = update_membership_eventwise(
            proportion_of_coalescing_all[sample_no],
            epoch_index_all[sample_no],
            gamma_arr,
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
            target_branch_length_masked[sample_no],
            args.num_clusters,
        )
        log_denom_em_sam = update_membership_eventwise_denom(
            gamma_arr,
            denom[sample_no],
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
        )
        start = int(np.sum(n_sites[0:sample_no]))
        end = int(np.sum(n_sites[0:sample_no+1]))
        log_num_em[:, start:end] = log_num_em_sam
        log_denom_em[
            :, start:end
        ] = log_denom_em_sam
        
        own_membership_sam, trans_num_sam, trans_denom_sam, tau_sam, log_likelihood_sam = Decode_grid(
            tree_left_bp[sample_no],
            tree_right_bp[sample_no],
            gen_grid_all[sample_no],
            trans_prop,
            log_num_em_sam + log_denom_em_sam,
            tau,
            window_size=args.force_build,
        )
        for clust_j in range(n_clusters):
            own_membership_trial[clust_j].extend(own_membership_sam[clust_j].tolist())
        if sample_no == 0:
            trans_num = trans_num_sam
            trans_denom = trans_denom_sam
            tau_update = tau_sam
            log_likelihood = log_likelihood_sam
        else:
            trans_num += trans_num_sam
            trans_denom += trans_denom_sam
            tau_update += tau_sam 
            log_likelihood += log_likelihood_sam

    trans_prop = trans_num/trans_denom
    tau = tau_update/np.sum(tau_update)
    loglikelihood_per_comp = log_num_em + log_denom_em
    own_membership_trial = np.array(own_membership_trial, dtype='float64')
    own_membership_trial = np.repeat(own_membership_trial, args.num_subtrees, axis=1)
    if args.regress_out:
        loglikelihood_per_comp = loglikelihood_per_comp[0] - loglikelihood_per_comp[1]
    
    for epoch in range(args.sweep_num_iters):
        own_membership_trial, trans_prop, gamma_arr, tau, log_likelihood = e_m_step(
            args,
            own_membership_trial,
            trans_prop,
            tau,
            gamma_arr,
            proportion_of_coalescing_all,
            epoch_index_all,
            denom,
            unique_groups,
            epoch_intervals,
            n_sites,
            n_samples,
            epoch,
            target_branch_length_masked,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
            regressors=regressors_all
            # loglikehood_cov=loglikehood_cov
        )
    
    ### caution ########
    # if args.joint_fit:
    #     for sam_no, sam in enumerate(args.sample_id):
    #         file = glob.glob('output_nea/nea_debug_overall_membership*sample_id_' + str(sam) + '.csv')[0]
    #         own_membership_trial[:, sam_no*n_sites:(sam_no+1)*n_sites] = pd.read_csv(file, sep='\s+').values[:, 3:].T

    if args.load_membership != None:
        own_membership_trial = np.load(args.load_membership)
        trans_prop = args.t_admix_guess
        tau = np.load(args.load_props)
    

    if args.joint_fit:
        return (
            log_likelihood,
            own_membership_trial,
            trans_prop,
            tau,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
            unique_groups,
            loglikelihood_per_comp,
        )
    else:
        return (
            log_likelihood,
            own_membership_trial,
            trans_prop,
            tau,
            unique_groups,
            loglikelihood_per_comp,
        )


def random_sweep(
    args,
    n_clusters,
    unique_groups,
    n_epochs,
    num_trees_per_sample,
    n_repeats,
    ts_list,
    poplabels,
    mask_dodgy,
    epoch_intervals,
    target_branch_length_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    chr_map,
    loglikehood_cov=None,
    regressors_all=None
):
    print("Performing a random sweep for better initialization")
    best_loglikelihood = -np.inf
    epoch_intervals_pow = np.power(10, epoch_intervals)

    st = time.time()
    if not args.joint_fit:
        num, denom, proportion_of_coalescing_all, epoch_index_all = [], [], [], []
        for sample_no, sample in enumerate(args.sample_id):
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = load_fixed_params(args, ts_list, sample, poplabels, mask_dodgy[sample_no], chr_map, epoch_intervals, target_branch_length_masked[sample_no])
            num.append(num1)
            denom.append(denom1)
            proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
            epoch_index_all.append(epoch_index_all1)
        del ts_list
        gc.collect()
        ts_list = []
    else:
        num, denom, proportion_of_coalescing_all, epoch_index_all = (
            [],
            [],
            [],
            [],
        )
    print("fixed params:" + str(time.time() - st))
    st = time.time()
    out = Parallel(n_jobs=args.n_jobs)(
        delayed(random_sweep_iter)(
            args,
            n_clusters,
            unique_groups,
            n_epochs,
            num_trees_per_sample,
            n_iters,
            ts_list,
            poplabels,
            mask_dodgy,
            epoch_intervals_pow,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
            target_branch_length_masked,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
            chr_map,
            # loglikehood_cov
            regressors_all=regressors_all
        )
        for n_iters in range(n_repeats)
    )
    print("random sweep: " + str(time.time() - st))
    loglikelihood_per_comp_arr = []
    for i in range(len(out)):
        if not args.joint_fit:
            (
                log_likelihood,
                own_membership_trial,
                trans_prop_trial,
                tau_trial,
                unique_groups,
                loglikelihood_per_comp,
            ) = out[i]
        else:
            (
                log_likelihood,
                own_membership_trial,
                trans_prop_trial,
                tau_trial,
                num,
                denom,
                proportion_of_coalescing_all,
                epoch_index_all,
                unique_groups,
                loglikelihood_per_comp,
            ) = out[i]
        loglikelihood_per_comp_arr.append(loglikelihood_per_comp)
        if log_likelihood > best_loglikelihood or i == 0:
            best_loglikelihood = log_likelihood
            own_membership = own_membership_trial
            trans_prop = trans_prop_trial
            tau = tau_trial
            (
                best_num,
                best_denom,
                best_proportion_of_coalescing_all,
                best_epoch_index_all,
            ) = (num, denom, proportion_of_coalescing_all, epoch_index_all)
    return (
        own_membership,
        trans_prop,
        tau,
        best_num,
        best_denom,
        best_proportion_of_coalescing_all,
        best_epoch_index_all,
        unique_groups,
        loglikelihood_per_comp_arr,
    )


def write_membership_grid(
    args,
    own_membership,
    chr_map,
    mask_dodgy,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    n_clusters,
    sample_name_list,
    sample_id_list,
    output,
    window_size=1e3,
):
    assert len(tree_left_bp) == len(tree_right_bp)
    count_i = 0
    for k, (sample_name, sample_id) in enumerate(zip(sample_name_list, sample_id_list)):
        chr_map_ = chr_map[mask_dodgy[k]]
        res = []
        for i, (c, l, r) in enumerate(zip(chr_map_, tree_left_bp[k], tree_right_bp[k])):
            for j in range(int(l / window_size), int(r / window_size)):
                res.append(
                    [c, j * window_size] +  own_membership[:, count_i].tolist()
                )
                count_i += 1
        df = pd.DataFrame(
            data=np.array(res),
            columns=["chr", "pos"]
            + ["prob_" + str(i) for i in range(n_clusters)],
        )

        df['genpos'] = gen_grid_all[k] ## unit in morgans 
        # ## correct the genpos
        # for chr in df.chr.unique():
        #     if os.path.isfile(args.rec + str(int(chr)) + ".txt"):
        #         recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(int(chr)) + ".txt")
        #     elif os.path.isfile(args.rec + str(int(chr)) + ".txt.gz"):
        #         recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(int(chr)) + ".txt.gz")
        #     else:
        #         raise "Recomb map format not identified"   
        #     df.loc[df.chr == chr, 'genpos'] = 100*recomb_map_msprime.get_cumulative_mass(df.loc[df.chr==chr,'pos'].values)

        df.to_csv(
            output + "_overall_membership_" + str(sample_name) + "_sample_id_" + str(sample_id) + ".csv",
            index=False,
            sep="\t",
        )


def write_membership_gamma(
    args,
    own_membership,
    gamma_arr,
    tau,
    mask_dodgy,
    chr_map,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    epoch_intervals,
    unique_groups,
    sample_id_label,
    sample_name_list
):
    ## gamma and membership plots
    filename = (
        "overall_membership_" + sample_id_label + ".npy"
    )  ## this saves membership for all the trees (without the filtering)
    filename = args.output + "_" + filename
    with open(filename, "wb") as f:
        np.save(f, own_membership)

    write_membership_grid(
        args,
        own_membership,
        np.array(chr_map),
        mask_dodgy,
        tree_left_bp,
        tree_right_bp,
        gen_grid_all,
        args.num_clusters,
        sample_name_list,
        args.sample_id,
        args.output,
        args.force_build,
    )

    write_coal(
        gamma_arr,
        sample_id_label + ".coal",
        unique_groups,
        args.output,
        epoch_intervals,
    )

    with open(
        args.output + "_gamma_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, gamma_arr)

    with open(
        args.output + "_props_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, tau)

def main(args):
    ### Initialize some global variables
    # if args.joint_fit:
    #     raise ValueError("Not yet implemented the changes, please wait!")

    if args.masking_threshold_bstat is not None and args.bmap is None:
        raise ValueError("Please provide a bmap file for masking")

    if args.groups is None:
        args.groups = np.arange(args.num_clusters)
    epoch_intervals = np.array(
        [-np.inf]
        + np.linspace(
            args.start_time - math.log(args.ypg, 10),
            args.end_time - math.log(args.ypg, 10),
            args.num_epochs - 1,
        ).tolist()
        + [np.inf],
        dtype="float64",
    )
    if args.load_gamma is not None:
        if ".coal" in args.load_gamma:
            second_line = (
                open(args.load_gamma).readlines()[1].strip("\n").split(" ")[:-1]
            )
            epoch_intervals = np.array(np.log10(np.array(second_line, dtype="float64")).tolist() + [np.inf], dtype='float64')
            args.num_epochs = len(epoch_intervals) - 1

    poplabels = pd.read_csv(args.poplabels, sep="\s+")
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))

    ### Load all the trees in a list
    ts_list = load_trees(args)
    if len(ts_list) > 0:
        if len(poplabels) == ts_list[0].num_samples // 2:
            poplabels = pd.DataFrame(
                np.repeat(poplabels.values, 2, axis=0), columns=poplabels.columns
            )
        if len(poplabels) != ts_list[0].num_samples:
            raise ValueError(
                "Number of samples in trees doesnt match number of samples in poplabels.txt"
            )
    else:
        ### TODO: check if poplabels need to repeated or not, if no trees specified
        poplabels = pd.DataFrame(
            np.repeat(poplabels.values, 2, axis=0), columns=poplabels.columns
        )

    sample_id = []
    for i in range(len(args.sample_id)):
        if "-" in args.sample_id[i]:
            sample_id.extend(
                np.arange(
                    int(args.sample_id[i].split("-")[0]),
                    int(args.sample_id[i].split("-")[1]) + 1,
                ).tolist()
            )
        else:
            try:
                sample_id.append(int(args.sample_id[i]))
            except:
                ## mention the group
                sample_id.extend(poplabels[((poplabels.INCLUDE == 1)&(poplabels.GROUP == args.sample_id[i]))].index.tolist())
    sample_id = np.unique(sample_id).tolist()
    print(sample_id)
    args.sample_id = sample_id
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    if len(sample_id_label) > 100:
        print("Truncating sample_id_label to 100 characters to avoid long file names")
        sample_id_label = sample_id_label[0:100]
    sample_name_list = poplabels.loc[args.sample_id].ID.tolist(); print(sample_name_list)

    poplabels.loc[args.sample_id, "INCLUDE"] = 1
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)

    num_samples = len(poplabels)
    for sample in args.sample_id:
        if sample >= num_samples or sample < 0:
            raise ValueError("The sample ids are out of range") 
        else:
            print(str(sample) + " is: " + str(poplabels.GROUP.iloc[sample]))
    
    ### Load tree stats
    (
        recomb_rates,
        mutrate_opportunity_target,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        chr_map,
        frac_branches_with_snp_target,
        mutrate_logpmf_target,
        num_snps_on_lineage,
        b_stat
    ) = load_tree_stats(args, ts_list, poplabels, args.tree_stats_file_prefix)

    ### Filter based on recombination rates
    if args.spurious_run is not None:
        mask_dodgy = []
        for sample_id in args.sample_id:
            spurious_filename_sample = glob.glob(args.spurious_run + '*_sample_id_' + str(sample_id) + '.csv')[0]
            membership = pd.read_csv(spurious_filename_sample, sep="\t")
            smaller_cluster = np.argmin(
                membership[membership.columns[3:]].values.mean(axis=0)
            )
            ## also remove the ones that are close to the ones that are in the smaller cluster
            for i in range(len(membership)):
                if membership.loc[i, "prob_" + str(smaller_cluster)] >= 0.001:
                    mask_for_i = (membership['chr'] == membership.loc[i, 'chr']) & (np.abs(membership['genpos'] - membership.loc[i, 'genpos']) < 0.2)
                    membership.loc[mask_for_i, "prob_" + str(smaller_cluster)] = 0.0011
            
            mask_df = membership[membership["prob_" + str(smaller_cluster)] < 0.001]
            
            mask_df = mask_df[mask_df.columns[:2]]            
            mask_dodgy_sam = load_mask_csv(
                args, mask_df, tree_left_bp, tree_right_bp, chr_map
            )
            mask_dodgy_sam2 = filter_recomb_rate(
                args.masking_threshold,
                tree_left_bp,
                recomb_rates,
                chr_map
            )
            mask_dodgy_sam3 = filter_bstat(
                args.masking_threshold_bstat,
                tree_left_bp,
                b_stat,
                chr_map
            )
            print("Number of locations removed = " + str(1 - sum(mask_dodgy_sam*mask_dodgy_sam2*mask_dodgy_sam3)/len(mask_dodgy_sam)))
            mask_dodgy.append(mask_dodgy_sam*mask_dodgy_sam2*mask_dodgy_sam3)

        # mask_dodgy = mask_dodgy.sum(axis=0) == len(mask_dodgy)
        # print("Number of locations removed total = " + str(1 - mask_dodgy.sum()/len(mask_dodgy)))

    elif args.load_mask is None:
        mask_dodgy = []
        for sample_id in args.sample_id:
            mask_dodgy_sam = filter_recomb_rate(
                args.masking_threshold,
                tree_left_bp,
                recomb_rates,
                chr_map
            )
            mask_dodgy_sam2 = filter_bstat(
                args.masking_threshold_bstat,
                tree_left_bp,
                b_stat,
                chr_map
            )
            mask_dodgy.append(mask_dodgy_sam*mask_dodgy_sam2)
    else:
        raise ValueError("Either specify spurious_run or load_mask")
        mask_dodgy = np.zeros(len(recomb_rates), dtype="bool")
        mask_df = pd.read_csv(args.load_mask, sep="\s+")
        mask_dodgy = load_mask_csv(
            args, mask_df, args.sample_id, ts_list, mask_dodgy, chrs
        )

    for sample_no in range(len(args.sample_id)):
        mask_dodgy[sample_no] = np.repeat(mask_dodgy[sample_no], args.num_subtrees)

    ### Use ground-truth local ancestry for reference samples
    num_trees_per_sample = [np.sum(mask_dodgy[sam]) for sam in range(len(args.sample_id))]
    poplabels_included = poplabels[poplabels.INCLUDE == 1]
    tree_left_bp = np.array(
        [np.array(tree_left_bp)[mask_dodgy[sam]].tolist() for sam in range(len(args.sample_id))]
    )
    tree_right_bp = np.array(
        [np.array(tree_right_bp)[mask_dodgy[sam]].tolist() for sam in range(len(args.sample_id))]
    )
    tree_left_bp_gen = np.array(
        [np.array(tree_left_bp_gen)[mask_dodgy[sam]].tolist() for sam in range(len(args.sample_id))]
    )
    tree_right_bp_gen = np.array(
        [np.array(tree_right_bp_gen)[mask_dodgy[sam]].tolist() for sam in range(len(args.sample_id))]
    )

    ## generate gen_grid for each sample 
    gen_grid_kb = {}
    for chr in list(map(int, args.chrs.split(","))):
        if os.path.isfile(args.rec + str(chr) + ".txt"):
            recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(chr) + ".txt")
        elif os.path.isfile(args.rec + str(chr) + ".txt.gz"):
            recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(chr) + ".txt.gz")
        else:
            raise "Recomb map format not identified"   
        
        gen_grid_kb[chr] = recomb_map_msprime.get_cumulative_mass(np.arange(0, int(tree_right_bp[0][np.array(chr_map)[mask_dodgy[0]] == chr].max()), args.force_build))


    gen_grid_all = []
    for sample_no in range(len(args.sample_id)):
        gen_grid_sam = []
        for i, (c, l, r) in enumerate(zip(np.array(chr_map)[mask_dodgy[sample_no]], tree_left_bp[sample_no], tree_right_bp[sample_no])):
            for j in range(int(l/args.force_build), int(r/args.force_build)):
                gen_grid_sam.append(gen_grid_kb[c][j])
        gen_grid_all.append(np.array(gen_grid_sam))

    ### record the sites per sample 
    ### Compute recombination rate and b-statistcs at that locations 
    ### Provide them as regressors to regress-out in the EM

    ### Compute recomb rate and b stats every 10kb
    regressors_all = []
    if args.regress_out:
        regressors_kb = {}
        for chr in list(map(int, args.chrs.split(","))):
            if os.path.isfile(args.rec + str(chr) + ".txt"):
                recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(chr) + ".txt")
            elif os.path.isfile(args.rec + str(chr) + ".txt.gz"):
                recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(chr) + ".txt.gz")
            else:
                raise "Recomb map format not identified"   
            
            if args.bmap is not None:
                bmap = pd.read_csv(args.bmap + '{0}.bmap.txt'.format(int(chr)), '\s+')
            
            rec_rate_kb = []
            b_stat_kb = []

            for pos in np.arange(0, int(tree_right_bp[0][np.array(chr_map)[mask_dodgy[0]] == chr].max()), args.force_build):
                if np.isnan(recomb_map_msprime.get_rate(pos)):
                    rec_rate_kb.append(1e9*recomb_map_msprime.mean_rate)
                else:
                    rec_rate_kb.append(1e9*recomb_map_msprime.get_rate(pos))
                idx1 = bisect.bisect_left(bmap['start'].values, pos)
                idx2 = bisect.bisect_left(bmap['end'].values, pos)
                if idx2 == idx1 - 1:
                    b_stat_kb.append(bmap['b-value'].values[idx2])
                else:
                    b_stat_kb.append(np.mean(bmap['b-value'].values))
            
            if args.bmap is not None:
                regressors_kb[chr] = np.vstack([rec_rate_kb, b_stat_kb]).T
            else:
                regressors_kb[chr] = np.array(rec_rate_kb).reshape(-1, 1)
                
        for sample_no in range(len(args.sample_id)):
            regressors_sam = []
            for i, (c, l, r) in enumerate(zip(np.array(chr_map)[mask_dodgy[sample_no]], tree_left_bp[sample_no], tree_right_bp[sample_no])):
                for j in range(int(l/args.force_build), int(r/args.force_build)):
                    regressors_sam.append(regressors_kb[c][j])
            regressors_all.append(np.array(regressors_sam))

    st = time.time()
    target_branch_length_masked = get_target_branch_length(
        args, poplabels, ts_list, chrs, mask_dodgy, args.sample_id
    )
    print("target branch length: " + str(time.time() - st))
    ### Calculate ground truth local ancestry
    if args.mode == "sim":
        ground_truth_membership = []
        for sample in args.sample_id:
            ground_truth_membership_sample = make_ground_truth(
                ts_list,
                mask_dodgy=mask_dodgy[:: args.num_subtrees],
                path=args.ground_truth_path,
                sample=[sample],
                chrs=chrs,
                force_build=args.force_build,
                tree_left_bp=tree_left_bp,
                tree_right_bp=tree_right_bp,
            )
            ground_truth_membership_sample = np.repeat(
                ground_truth_membership_sample, args.num_subtrees, axis=1
            )
            ground_truth_membership.append(ground_truth_membership_sample)
        ground_truth_membership = (
            np.array(ground_truth_membership)
            .transpose(1, 0, 2)
            .reshape(ground_truth_membership_sample.shape[0], -1)
        )

    ### Initialize local ancestry
    if args.init_at_truth and args.joint_fit:
        unique_groups, i = {}, 0
        for group in np.unique(poplabels_included.GROUP):
            if poplabels_included.GROUP.iloc[args.sample_id[0]] == group:
                for c in range(args.num_clusters):
                    unique_groups[group + str(c + 1)] = i
                    i += 1
            else:
                unique_groups[group] = i
                i += 1

        gt_ref = np.zeros((len(poplabels.GROUP), num_trees), dtype="object")
        for sample_no, sample in enumerate(poplabels.index):
            group = poplabels.GROUP.loc[sample]
            if group == poplabels.GROUP.loc[args.sample_id[0]]:
                gt_ref[sample_no] = {
                    poplabels.GROUP.iloc[sample] + str(c + 1): 1
                    for c in range(args.num_clusters)
                }
            else:
                try:
                    gt_ref[sample_no] = unique_groups[group]
                except:
                    gt_ref[sample_no] = "NA"

        gt_ref = np.array(gt_ref, dtype="object")

        print("Calculating ground-truth ancestry of the reference...")
        gt_ref_orig, _ = get_groundtruth_reference(
            ts_list,
            poplabels_included,
            np.sum(mask_dodgy) // args.num_subtrees,
            mask_dodgy[:: args.num_subtrees],
            args.ground_truth_path,
            chrs,
            poplabels.GROUP.loc[args.sample_id[0]],
            args.force_build,
        )
        for sample_no, sample in enumerate(
            poplabels_included[
                poplabels_included.GROUP == poplabels.GROUP.loc[args.sample_id[0]]
            ].index
        ):
            for n_t in range(gt_ref.shape[1]):
                gt_ref[sample, n_t] = unique_groups[
                    poplabels.GROUP.iloc[args.sample_id[0]]
                    + str(gt_ref_orig[sample_no, n_t] + 1)
                ]

        n = np.zeros(
            (args.num_clusters, len(unique_groups), len(epoch_intervals) - 1),
            dtype="float64",
        )
        d = np.zeros(
            (args.num_clusters, len(unique_groups), len(epoch_intervals) - 1),
            dtype="float64",
        )
        tau = np.zeros(args.num_clusters, dtype="float64")
        for sample_no, sample in enumerate(args.sample_id):
            own_membership_sample = make_one_hot(gt_ref[sample])
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                num_trees,
                mask_dodgy,
                sample,
                args.sample_id,
                np.power(10, epoch_intervals),
                target_branch_length_masked[sample_no],
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=gt_ref,
                ignore_first_epoch=args.ignore_first_epoch,
            )
            for j in range(len(own_membership_sample)):
                n_j, d_j = compute_gamma_num_denom(
                    own_membership_sample[j],
                    np.ones_like(n)[j],
                    proportion_of_coalescing_all[sample_no],
                    denom[sample_no],
                    epoch_index_all[sample_no],
                    len(unique_groups),
                    np.arange(0, num_trees),
                    len(epoch_intervals),
                    target_branch_length_masked[sample_no],
                    args.ignore_first_epoch,
                    args.ignore_last_epoch,
                    tree_left_bp,
                    tree_right_bp,
                    args.force_build,
                )
                n[j] += n_j
                d[j] += d_j
            print(np.mean(own_membership_sample, axis=1))

            tau += np.mean(own_membership_sample, axis=1)

        tau /= len(args.sample_id)
        gamma_arr = n / d

        print(tau)
        print(gamma_arr)

        with open(
            args.output + "_gamma_all.npy",
            "wb",
        ) as f:
            np.save(f, gamma_arr)

        with open(
            args.output + "_props_all.npy",
            "wb",
        ) as f:
            np.save(f, tau)

        return

    elif args.init_at_truth and not args.joint_fit:
        own_membership = ground_truth_membership
        num, denom, proportion_of_coalescing_all, epoch_index_all = [], [], [], []
        for sample_no, sample in enumerate(args.sample_id):
            ## TODO: use load_fixed_params instead
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                num_trees,
                mask_dodgy,
                sample,
                args.sample_id,
                np.power(10, epoch_intervals),
                target_branch_length_masked[sample_no],
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=None,
                ignore_first_epoch=args.ignore_first_epoch,
            )
            num.append(num1)
            denom.append(denom1)
            proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
            epoch_index_all.append(epoch_index_all1)
        ### initial guess for trans_prop = t_admix
        date_guess = (
            np.power(10, np.random.uniform(np.log10(20), np.log10(2000)))
            if args.t_admix_guess is None
            else args.t_admix_guess
        )
        # print("Admixture time guess = ", date_guess)
        trans_prop = date_guess
        tau = np.mean(own_membership, axis=1)

    else:
        # sweep_num_iters = copy.deepcopy(args.sweep_num_iters)
        # args.sweep_num_iters = 0 if args.regress_out else args.sweep_num_iters
        (
            own_membership,
            trans_prop,
            tau,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
            unique_groups,
            loglikelihood_per_comp_arr,
        ) = random_sweep(
            args,
            args.num_clusters,
            unique_groups,
            len(epoch_intervals),
            num_trees_per_sample,
            args.n_repeats,
            ts_list,
            poplabels,
            mask_dodgy,
            epoch_intervals,
            target_branch_length_masked,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
            chr_map,
            regressors_all=regressors_all
        )
        # args.sweep_num_iters = sweep_num_iters
        # if args.regress_out:
        #     raise ValueError ("Not yet implemented the changes, please wait!")
        #     st = time.time()
        #     n_sites = own_membership.shape[1] // len(args.sample_id)
        #     loglikelihood_per_comp_arr = np.array(loglikelihood_per_comp_arr)
        #     loglikelihood_per_comp_arr = loglikelihood_per_comp_arr.reshape(
        #         args.n_repeats, len(args.sample_id), n_sites
        #     )
        #     ## calc covariance per tree
        #     loglikehood_cov = np.zeros(
        #         (n_sites, len(args.sample_id), len(args.sample_id))
        #     )
        #     for i in range(n_sites):
        #         loglikehood_cov[i] = np.cov(loglikelihood_per_comp_arr[:, :, i].T)
        #     print("Covariance calculation time = ", time.time() - st)

        #     loglikehood_corr = np.zeros(
        #         (n_sites, len(args.sample_id), len(args.sample_id))
        #     )
        #     for i in range(n_sites):
        #         loglikehood_corr[i] = np.corrcoef(loglikelihood_per_comp_arr[:, :, i].T)
        #         for j in range(len(args.sample_id)):
        #             for k in range(len(args.sample_id)):
        #                 if np.isnan(loglikehood_corr[i][j,k]):
        #                     if np.std(loglikelihood_per_comp_arr[:,j,i]) == 0 and np.std(loglikelihood_per_comp_arr[:,k,i]) == 0:
        #                         loglikehood_corr[i, j, k] = 1
        #                     else:
        #                         loglikehood_corr[i, j, k] = 0
        #         for j in range(len(args.sample_id)):
        #             loglikehood_corr[i, j, j] = 0

        #     mean_loglikelihood_corr = np.mean(loglikehood_corr, axis=0)
        #     import seaborn as sns 
        #     import matplotlib.pyplot as plt 
        #     plt.Figure((6,6))
        #     sns.heatmap(mean_loglikelihood_corr, center=0)
        #     mean_ll_corr = np.sum(mean_loglikelihood_corr) / (len(mean_loglikelihood_corr)**2-len(mean_loglikelihood_corr))
        #     plt.title('Expected Corr. matrix (Avg. corr. = {0})'.format(np.round(mean_ll_corr,2)))
        #     plt.savefig("corr1.png", dpi=300)
        #     (
        #         own_membership,
        #         trans_prop,
        #         tau,
        #         num,
        #         denom,
        #         proportion_of_coalescing_all,
        #         epoch_index_all,
        #         unique_groups,
        #         loglikelihood_per_comp_arr,
        #     ) = random_sweep(
        #         args,
        #         args.num_clusters,
        #         unique_groups,
        #         len(epoch_intervals),
        #         num_trees_per_sample,
        #         args.n_repeats,
        #         ts_list,
        #         poplabels,
        #         mask_dodgy,
        #         epoch_intervals,
        #         target_branch_length_masked,
        #         tree_left_bp,
        #         tree_right_bp,
        #         tree_left_bp_gen,
        #         tree_right_bp_gen,
        #         chr_map,
        #         loglikehood_cov=loglikehood_cov,
        #     )

    if args.load_gamma:
        gamma_arr = load_gamma(args.load_gamma, args.groups, unique_groups)
        ## CAUTION: recheck this, removing gamma_arr outside the range
        for epoch in range(gamma_arr.shape[2]):
            if args.ignore_first_epoch and (args.start_time - math.log(args.ypg, 10)) > epoch_intervals[epoch]:
                gamma_arr[:,:,epoch] = np.nan
            if args.ignore_last_epoch and (args.end_time - math.log(args.ypg, 10)) < epoch_intervals[epoch+1]:
                gamma_arr[:,:,epoch] = np.nan
        
        print(gamma_arr)
    else:
        gamma_arr = None
    if args.load_props:
        tau = load_props(args.load_props)

    ### EM
    if args.evaluate_gamma:
        filename = (
            "overall_membership_iter0_" + sample_id_label + ".npy"
        )  ## this saves membership for all the trees (without the filtering)
        filename = args.output + "_" + filename
        with open(filename, "wb") as f:
            np.save(f, own_membership)

        log_likelihood_arr = []
        # print("Starting the EM..")

        filename_logl = args.output + "_" + sample_id_label + ".logl"
        filename_tau = args.output + "_" + sample_id_label + ".tau"
        f_logl = open(filename_logl, "w")
        f_tau = open(filename_tau, "w")

        st = time.time()
        n_sites = []
        for sample_no in range(len(args.sample_id)):
            n_sites_sam = 0
            for i, (l, r) in enumerate(zip(tree_left_bp[sample_no], tree_right_bp[sample_no])):
                n_sites_sam += int(r/args.force_build) - int(l/args.force_build)
            n_sites.append(n_sites_sam)
        for epoch in range(args.num_iters):
            own_membership, trans_prop, gamma_arr, tau, log_likelihood = e_m_step(
                args,
                own_membership,
                trans_prop,
                tau,
                gamma_arr,
                proportion_of_coalescing_all,
                epoch_index_all,
                denom,
                unique_groups,
                epoch_intervals,
                n_sites,
                len(args.sample_id),
                epoch,
                target_branch_length_masked,
                tree_left_bp,
                tree_right_bp,
                gen_grid_all,
                regressors=regressors_all
                # loglikehood_cov=loglikehood_cov if args.regress_out else None,
            )
            # print(tau)

            if epoch == 0:
                write_coal(
                    gamma_arr,
                    sample_id_label + "_iter0.coal",
                    unique_groups,
                    args.output,
                    epoch_intervals,
                )
            for i in range(np.shape(tau)[0]):
                f_tau.write(str(tau[i]) + " ")
            f_tau.write("\n")
            log_likelihood_arr.append(log_likelihood)

            if args.mode == "sim":
                calculate_accuracy(own_membership, ground_truth_membership)

            ## Early-stopping
            # print("log-likelihood = " + str(log_likelihood_arr[-1]), flush=True)
            f_logl.write(str(log_likelihood_arr[-1]) + "\n")

        print("HMM admix date = " + str(trans_prop))
        print("em iters: " + str(time.time() - st))
        print("Final log-likelihood = " + str(log_likelihood_arr[-1]))

        if args.mode == "sim":
            write_calibration(args, own_membership, ground_truth_membership)
            filename = (
                "ground_truth_membership_" + sample_id_label + ".npy"
            )  ## this saves membership for all the trees (without the filtering)
            filename = args.output + "_" + filename
            with open(filename, "wb") as f:
                np.save(f, ground_truth_membership)

        write_membership_gamma(
            args,
            own_membership,
            gamma_arr,
            tau,
            mask_dodgy,
            chr_map,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
            epoch_intervals,
            unique_groups,
            sample_id_label,
            sample_name_list
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-fb",
        "--force_build",
        help="force build size to subsample the trees in bp",
        type=float,
        default=10000,
    )
    parser.add_argument("-r", "--rec", help="Filename of rec maps.", type=str)
    parser.add_argument("--bmap", help="Filename of Bmap file of the format name_{chr}.bmap.txt", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="Output prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "-chrs",
        "--chrs",
        help="Comma-seperated list of chromosomes to be considered",
        type=str,
        default="1,2",
    )
    parser.add_argument(
        "-masking_thresh",
        "--masking_threshold",
        help="Remove top x cent of high recombination regions",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-masking_thresh_bstat",
        "--masking_threshold_bstat",
        help="Remove top x cent of high b-stat regions",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-init_at_truth",
        "--init_at_truth",
        help="Do you wish to initialize at ground-truth",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "--n_repeats",
        help="Number of restarts when randomly initializing local ancestry, we choose best out of n_trails",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-load_gamma",
        "--load_gamma",
        help="Starting gamma values written in a file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-load_mask",
        "--load_mask",
        help="Load mask csv file with chr, tree_position_left//force_build",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-load_props",
        "--load_props",
        help="Starting taU values written in a file or space seperated list",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--t_admix_guess",
        help="Guess for the time of admixture",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-trees",
        "--trees",
        help="Location to trees in tskit format",
        type=str,
        default=None,
    )
    parser.add_argument("--poplabels", help="Location to poplabels file", type=str)
    parser.add_argument(
        "--mutden",
        help="Location prefix to mutation density file from Relate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--allmuts",
        help="Location prefix to allmuts file from Relate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ground_truth_path",
        help="Location prefix to groundtruth file generated from tskit in simulations",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-i",
        "--num_iters",
        help="Number of iterations for EM",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-k",
        "--num_clusters",
        help="Number of clusters to find using the EM",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-mode",
        "--mode",
        help="Which mode do you want to run in? Simulation or Real-world ?",
        type=str,
        default="real",
        choices=("sim", "real"),
    )
    parser.add_argument(
        "-evaluate_gamma",
        "--evaluate_gamma",
        help="Run EM to calculate the gammas (population sizes)",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "-start_time",
        "--start_time",
        help="Starting time for the population size plots, measured in log-scale",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "-end_time",
        "--end_time",
        help="Ending time for the population size plots, measured in log-scale",
        type=float,
        default=6,
    )
    parser.add_argument(
        "-num_epochs",
        "--num_epochs",
        help="Num epochs",
        type=int,
        default=9,
    )
    parser.add_argument(
        "-ignore_first_epoch",
        "--ignore_first_epoch",
        help="Ignore first epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "-ignore_last_epoch",
        "--ignore_last_epoch",
        help="Ignore last epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "--opportunity_filter",
        help="check mutations on target lineage: criterion to choose trees",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "--num_subtrees",
        help="Number of subtrees in the composite likelihood",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_per_group",
        help="Maximum number of individuals in a subtrees for the composite likelihood",
        type=int,
        default=-1,
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--joint_fit",
        action="store_const",
        const=True,
        default=False,
        help="Joint fitting of multiple samples",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of threads")
    parser.add_argument(
        "--sweep_num_iters",
        type=int,
        default=10,
        help="Number of iterations to run EM for in random sweep",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        nargs="+",
        help="space seperated list of source group, example Nea CHB where Nea and CHB are population names in poplabels file",
    )
    parser.add_argument(
        "--spurious_run",
        type=str,
        default=None,
        help="Name of the spurious run to remove trees from minor component",
    )
    parser.add_argument(
        "--hmm",
        help="Run HMM or treat each window as independent",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "--regress_out",
        help="Regress out the mean posterior across samples to control for background selection",
        type=boolean,
        default=False,
    )
    parser.add_argument("--ypg", type=float, default=28, help="years per generation, 28 years default")
    parser.add_argument("--tree_stats_file_prefix", type=str, default=None, help="file prefix for the tree stats file")
    parser.add_argument("--branch_persistence_file_prefix", type=str, default=None, help="file prefix for the branch persistence file")
    parser.add_argument("--fixed_params_file_prefix", type=str, default=None, help="file prefix for the fixed params file")
    parser.add_argument("--load_membership", help = "Load the membership from a .npy file", type=str, default=None)
    args = parser.parse_args()
    if not args.hmm:
        args.t_admix_guess = 10.0**30
    np.random.seed(args.seed)  ## fix the random seed
    random.seed(args.seed)
    main(args)
