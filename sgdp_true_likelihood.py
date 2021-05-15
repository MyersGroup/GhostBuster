from numpy.core.fromnumeric import prod
import pandas as pd 
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import seaborn as sns
from collections import Counter
import scipy.sparse
import tskit
from tqdm import tqdm
import h5py
from sklearn.calibration import calibration_curve

epoch_intervals = np.array([-np.inf] + np.linspace(3 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])
epoch_intervals_pow = np.power(10, epoch_intervals)

def make_one_hot(X, max_X):
    X = np.array(X, dtype='int')
    classes = np.arange(0, max_X,1)
    if len(X.shape) == 2:
        Y = np.zeros((len(classes), X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        Y = np.zeros((len(classes), X.shape[0]))
    for c in classes:
        Y[c] = np.array(X == c, dtype='int')
    return Y

def make_ground_truth(tree_per_chr, chrs):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")
    ground_truth_membership_one_hot = np.zeros((7, tree_per_chr[-1]))
    count = 0
    for chr in chrs:
        if chr%2 == 0:
            ground_truth_membership_one_hot[6, tree_per_chr[count]:tree_per_chr[count+1]] = 1
        else:
            ground_truth_membership_one_hot[3, tree_per_chr[count]:tree_per_chr[count+1]] = 1
        count +=1 
    print("Done in " + str(time.time() - start_time))
    return ground_truth_membership_one_hot

def fixed_parameters(base_path, poplabels, unique_groups, num_trees, target_seq_, subset_ids, chrs):
    eps = 1e-20
    opportunity = np.zeros((len(unique_groups), len(epoch_intervals_pow)-1, num_trees))
    proportion_of_coalescing_all = []
    epoch_index_all = []
    count_mut_trees = -1
    group_id = {}
    for u in range(len(unique_groups)):
        group_id[unique_groups[u]] = u
    for chr in chrs:
        print(chr)
        ts = tskit.load(base_path + str(chr) + '.trees')
        # ts = ts.simplify(subset_ids)
        tree = ts.first()
        tree.next ## we ignore the first tree in SGDP trees
        num_samples = len(list(tree.samples()))
        for tid in tqdm(range(ts.num_trees - 1)): #len(list(ts.trees()))
            if tree.num_sites == 0:
                tree.next()
                continue
            count_mut_trees += 1
            if chr%2 ==  0:
                target_seq = int(target_seq_[0])
            else:
                target_seq = int(target_seq_[1])
            ## Make the coalescene table and sort it
            coal_events_matrix = []
            mapping = {}
            count = num_samples
            for s in tree.nodes():
                if s < num_samples:
                    mapping[s] = s
                else:
                    mapping[s] = count
                    count += 1
            for s in tree.nodes():
                if tree.children(s) != ():
                    a = tree.children(s)[0]
                    b = tree.children(s)[1]
                    c = s
                    t = tree.time(c)
                    coal_events_matrix.append([int(mapping[a]), int(mapping[b]), int(mapping[c]), t])
            coal_events_matrix = np.array(coal_events_matrix)
            coal_events_matrix = coal_events_matrix[coal_events_matrix[:, 3].argsort()] ## sorting based on coalescene times
            lineage_content = np.zeros((2*num_samples-1, len(unique_groups)))
            for m in range(len(poplabels)):
                lineage_content[2*m:2*m + 2, group_id[poplabels.GROUP.loc[m]]] = 1
            lineage_content[target_seq] = 0 ## setting lineage content of target sequence = 0 
            prev_branch_length = np.sum(lineage_content, axis = 0) #np.sum(lineage_content[:,1])
            ## lets calculate for group-1 for one tree
            proportion_of_coalescing_in_tree = []
            epoch_index_in_tree = []
            event_count = 0
            for epoch in range(len(epoch_intervals_pow) -1):
                coal_events_submatrix = coal_events_matrix[(coal_events_matrix[:, 3] >= epoch_intervals_pow[epoch]) & (coal_events_matrix[:, 3] < epoch_intervals_pow[epoch+1])]
                tprev = epoch_intervals_pow[epoch]
                for (a,b,c,t) in coal_events_submatrix:
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    event_count += 1
                    # opportunity[epoch] += (t-tprev)*np.sum(lineage_content[:,1]/(np.sum(lineage_content, axis = 1) + eps))
                    opportunity[:,epoch,count_mut_trees] += (t-tprev)*(prev_branch_length)
                    if a == target_seq:
                        proportion_of_coalescing = lineage_content[b]/(sum(lineage_content[b]))
                        # coal_count[:,epoch,count_mut_trees] += proportion_of_coalescing 
                        target_seq = c
                        lineage_content[c] = 0
                        proportion_of_coalescing_in_tree.append(proportion_of_coalescing)
                        epoch_index_in_tree.append(epoch)
                        prev_branch_length = prev_branch_length - lineage_content[b]/(sum(lineage_content[b]))
                    elif b == target_seq:
                        proportion_of_coalescing = lineage_content[a]/(sum(lineage_content[a])) ## sum() faster than np.sum()
                        # coal_count[:,epoch,count_mut_trees] += proportion_of_coalescing  
                        target_seq = c
                        lineage_content[c] = 0
                        proportion_of_coalescing_in_tree.append(proportion_of_coalescing)
                        epoch_index_in_tree.append(epoch)
                        prev_branch_length = prev_branch_length - lineage_content[a]/(sum(lineage_content[a]))
                    else:
                        lineage_content[c] = lineage_content[a] + lineage_content[b]
                        prev_branch_length = prev_branch_length - lineage_content[a]/(sum(lineage_content[a])) - lineage_content[b]/(sum(lineage_content[b])) + lineage_content[c]/(sum(lineage_content[c]))
                    lineage_content[a] = 0 
                    lineage_content[b] = 0
                    tprev = t
                if epoch < len(epoch_intervals_pow) -2:
                    opportunity[:,epoch,count_mut_trees] += (epoch_intervals_pow[epoch+1]-tprev)*(prev_branch_length)
                if (event_count == num_samples - 1) and epoch < len(epoch_intervals_pow) -2:
                    opportunity[:,epoch+1:,count_mut_trees] = 0.
                    break
            proportion_of_coalescing_all.append(proportion_of_coalescing_in_tree)
            epoch_index_all.append(epoch_index_in_tree)
            tree.next()
        del(tree)
        del(ts)
    return opportunity, proportion_of_coalescing_all, epoch_index_all

def compute_gamma_num(own_membership, prev_gamma, proportion_of_coalescing_all, epoch_index_all, num_ref_groups):
    eps = 1e-40
    num_full_tree = np.zeros((num_ref_groups, len(epoch_intervals) - 1))
    if not(isinstance(prev_gamma, np.ndarray)):
        for tid in range(len(proportion_of_coalescing_all)):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                epoch = epoch_index_in_tree[i]
                num = proportion_of_coalescing_in_tree[i]
                num = num/np.sum(num)
                num_full_tree[:,epoch] += own_membership[tid]*num 
    else:
        for tid in range(len(proportion_of_coalescing_all)):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                epoch = epoch_index_in_tree[i]
                prev_gamma_e = prev_gamma[:, epoch] ##adding eps for shielding purposes
                num = prev_gamma_e*proportion_of_coalescing_in_tree[i]
                if np.sum(num) > 0:
                    num = num/np.sum(num)
                num_full_tree[:,epoch] += own_membership[tid]*num 
    return num_full_tree

def compute_gamma_denom(own_membership, denom):
    eps = 1e-20
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    for epoch in range(len(epoch_intervals) - 1):#
        denom_1[epoch] = sum(denom[epoch]*own_membership)
    return denom_1 + eps

def count_num_trees(base_path, subset_ids, chrs):
    num_trees = 0
    tree_per_chr = [0]
    for chr in chrs:
        print(chr)
        ts = tskit.load(base_path + str(chr) + '.trees')
        # ts = ts.simplify(subset_ids)
        tree = ts.first()
        tree.next()
        for tid in range(ts.num_trees - 1): #len(list(ts.trees()))
            if tree.num_sites > 0:  
                num_trees += 1
            tree.next()
        tree_per_chr.append(num_trees)
        del(tree)
        del(ts)
    return num_trees, tree_per_chr

def anamoly_trees(df_arr, position, poplabels):
    df_arr = df_arr[position]
    x_all = []
    y_all = []
    for m in range(len(poplabels)):
        x = df_arr[:,2*m:2*m + 2].flatten()
        x = x[x > 0] ## removing the target sequence
        y = np.repeat(poplabels.GROUP.loc[m], len(x))
        x_all.extend(x)
        y_all.extend(y)
    return x_all, y_all

def calibration_plots(own_membership, membership_thresh, ground_truth_membership, num_bins = 40):
    prob_range = np.linspace(0,1,num_bins+1)
    calibration_arr = np.zeros(num_bins)
    print(np.sum(membership_thresh))
    print(own_membership)
    print(ground_truth_membership)
    own_membership = own_membership[ground_truth_membership] ## only consider the GT region
    matches = membership_thresh[ground_truth_membership] == 1
    print(np.sort(own_membership))
    print(prob_range)
    for bin in range(num_bins):
        own_membership_in_range = ((own_membership < prob_range[bin+1]) & (own_membership >= prob_range[bin]))
        acc = np.mean(matches[own_membership_in_range])
        # freq = np.sum(own_membership_in_range)
        calibration_arr[bin] = acc
    plt.clf()
    print(calibration_arr)
    plt.scatter(0.5*prob_range[0:-1] + 0.5*prob_range[1:], calibration_arr, marker = 'o')
    plt.ylabel('Accuracy')
    plt.xlabel('Probability')

def compute_tree_stats(base_path, subset_ids, chrs):
    num_trees = 0
    tree_size = []
    no_of_mutations = []
    tmrca = []
    recomb_rates = []
    frac_branches_with_snp = []
    num_snps_on_tree = [] 
    fraction_snps_not_mapping = []
    for chr in chrs:
        print(chr)
        ts = tskit.load(base_path + str(chr) + '.trees')
        recomb_map = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/recomb_maps/msprime_maps/genetic_map_GRCh37_chr'+ str(chr) +'.txt.gz', sep = '\t')
        relate_quality_output = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/SGDP_whole_genome/SGDP_relate/SGDP_v1_annot_ne_chr' + str(chr) + '.qual', sep = ' ')
        # ts = ts.simplify(subset_ids)
        tree = ts.first()
        tree.next()
        for tid in range(ts.num_trees - 1): #len(list(ts.trees()))
            if tree.num_sites > 0:
                recomb_events = recomb_map[(recomb_map['Position(bp)'] < tree.interval[1]) & (recomb_map['Position(bp)'] >= tree.interval[0])]
                    # recomb_rates.append(1e6*(recomb_events['Map(cM)'].iloc[-1] - recomb_events['Map(cM)'].iloc[0])/(tree.interval[1] - tree.interval[0]))
                recomb_rates.append(np.mean(recomb_events['Rate(cM/Mb)']))
                num_trees += 1
                tree_size.append(tree.interval[1] - tree.interval[0])
                no_of_mutations.append(tree.num_sites)
                tmrca.append(tree.time(tree.root))
                relate_quality = relate_quality_output[(relate_quality_output.pos < tree.interval[1]) & (relate_quality_output.pos >= tree.interval[0])].iloc[0]
                frac_branches_with_snp.append(relate_quality['frac_branches_with_snp'])
                num_snps_on_tree.append(relate_quality['num_snps_on_tree'])
                fraction_snps_not_mapping.append(relate_quality['fraction_snps_not_mapping'])
            tree.next()
        del(tree)
        del(ts)
    return tree_size, no_of_mutations, tmrca, recomb_rates, frac_branches_with_snp, num_snps_on_tree, fraction_snps_not_mapping

def mask_for_dodgy_trees(frac_branches_with_snp, num_snps_on_tree):
    mask = (frac_branches_with_snp > np.percentile(frac_branches_with_snp, 20)) & (num_snps_on_tree > np.percentile(num_snps_on_tree, 20))
    return mask

def main(sample_id, plot = False, gamma_arr = None):
    start_time = time.time()
    num_clusters = 3
    poplabels = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/SGDP_whole_genome/SGDP_relate/data/SGDP.poplabels', sep =' ')
    subset_ids = (2*poplabels.index).tolist() + (2*poplabels.index + 1).tolist()
    unique_groups = np.unique(poplabels.POP)#np.unique(poplabels.GROUP)
    poplabels = poplabels.reset_index()

    unique_groups = np.unique(poplabels.GROUP)
    print(unique_groups)

    base_path = '/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/SGDP_whole_genome/SGDP_relate/SGDP_v1_annot_ne_chr'
    chrs = range(15,23)

    num_trees, tree_per_chr = count_num_trees(base_path, subset_ids, chrs = chrs)
    tree_size, no_of_mutations, tmrca, recomb_rates, frac_branches_with_snp, num_snps_on_tree, fraction_snps_not_mapping = compute_tree_stats(base_path, subset_ids, chrs = chrs)
    mask_dodgy = mask_for_dodgy_trees(frac_branches_with_snp, num_snps_on_tree)
    print("Trees with high certainty = " + str(np.sum(mask_dodgy)))
    eps = 1e-300
    print("Total number of trees = " + str(num_trees))
    ground_truth_membership = make_ground_truth(tree_per_chr, chrs = chrs) 
    ground_truth_membership = ground_truth_membership[[6,3,0]] ## adding 0 and 1 as increasing a cluster
    print(ground_truth_membership)
    own_membership = np.random.dirichlet(np.ones(num_clusters), num_trees).T # #ground_truth_membership  #[[0,3]]#
    denom, proportion_of_coalescing_all, epoch_index_all = fixed_parameters(base_path, poplabels, unique_groups, num_trees, sample_id, subset_ids, chrs = chrs)

    log_likelihood_arr = []
    start_time_em = time.time()
    print("Starting the EM..")
    prev_gamma = None ## start with None
    for epoch in range(40):  ## max-iters = 40
        gamma_arr = np.zeros((len(own_membership), len(unique_groups), len(epoch_intervals) - 1))        
        ## M-step
        for j in range(len(own_membership)):
            if epoch == 0:
                n = compute_gamma_num(own_membership[j]*mask_dodgy, None, proportion_of_coalescing_all, epoch_index_all, len(unique_groups))
            else:
                n = compute_gamma_num(own_membership[j]*mask_dodgy, prev_gamma[j] , proportion_of_coalescing_all, epoch_index_all, len(unique_groups)) #compute_gamma_num(own_membership[j], prev_gamma[j], proportion_of_coalescing_all, epoch_index_all, len(unique_groups))
            for i in range(len(unique_groups)):
                d = compute_gamma_denom(own_membership[j]*mask_dodgy, denom[i])
                gamma_arr[j][i] = n[i]/d #n/d #
        prev_gamma = gamma_arr
        print(gamma_arr)
        tau = np.ones(len(own_membership))/len(own_membership)
        for j in range(len(own_membership)):
            tau[j] = np.clip(np.sum(own_membership[j])/own_membership[j].shape[0],1e-10, 1-1e-10)
        print(tau)

        ## E-step
        log_num_em = np.zeros((len(own_membership), num_trees))
        log_denom_em = np.zeros((len(own_membership), num_trees))
        for tid in range(num_trees):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                for j in range(len(own_membership)):
                    log_num_em[j, tid] += np.log(np.sum(gamma_arr[j,:,epoch_index_in_tree[i]]*proportion_of_coalescing_in_tree[i]))
            max_epoch_index = int(np.minimum(epoch_index_in_tree[i]+1, len(epoch_intervals_pow) - 1))
            for j in range(len(own_membership)):
                log_denom_em[j, tid] = -np.sum(gamma_arr[j,:,0:max_epoch_index]*denom[:,0:max_epoch_index,tid])  ##-np.sum(gamma_arr[j,:,:]*denom[:,:,tid])  ## summing only till the maximum epoch in that tree
        own_membership_update = np.exp(log_num_em + log_denom_em - np.repeat(np.max(log_num_em + log_denom_em, axis = 0).reshape(-1,1), len(own_membership), axis = 1).T)#np.exp(log_num_em + log_denom_em)
        own_membership_update = np.nan_to_num(own_membership_update, nan = 0.5) ##recent_addition
        for j in range(len(own_membership)):
            own_membership_update[j] *= tau[j]
        log_likelihood = np.sum(np.log(np.sum(own_membership_update, axis = 0)) + np.max(log_num_em + log_denom_em, axis = 0))
        log_likelihood_arr.append(log_likelihood)
        own_membership_update = own_membership_update/(np.sum(own_membership_update, axis = 0))
        own_membership = own_membership_update
        membership_thresh = make_one_hot(np.argmax(own_membership, axis = 0), len(own_membership))

        proportion_of_coalescing_top2 = np.zeros((num_trees, 2, len(unique_groups)))
        for tid in range(num_trees):
            proportion_of_coalescing_top2[tid,:,:] = proportion_of_coalescing_all[tid][0:2]
        plt.clf()
        for i in range(len(own_membership)):
            tree_size_i = np.array(tree_size)[np.argmax(own_membership, axis = 0) == i]
            num_mutations_i = np.array(no_of_mutations)[np.argmax(own_membership, axis = 0) == i]
            recomb_rates_i = np.array(recomb_rates)[np.argmax(own_membership, axis = 0) == i]
            recomb_rates_i = recomb_rates_i[~ np.isnan(np.array(recomb_rates_i))]
            proportion_of_coalescing_i = proportion_of_coalescing_top2[np.argmax(own_membership, axis = 0) == i]
            print("Cluster: " + str(i) + " Median tree size: " + str(np.median(tree_size_i)) + " Median num of muts: " + str(np.median(num_mutations_i)) + " Median recomb rate: " + str(np.median(recomb_rates_i)) + " Mean 1st coal. proportion: " + str(np.mean(proportion_of_coalescing_i[:,0,:], axis = 0)) + " Mean 2nd coal. proportion: " + str(np.mean(proportion_of_coalescing_i[:,1,:], axis = 0)))
            plt.hist(recomb_rates_i)
        
        plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
        plt.savefig('recomb_rates_hist.png')
        plt.show()

        ## Evaluate accuracy
        acc_arr = np.zeros((len(own_membership), len(ground_truth_membership)))
        for i in range(len(own_membership)):
            for j in range(0,len(ground_truth_membership)): ###hard-coding: because onle first three rows have ground-truth information
                acc = np.sum(membership_thresh[i] == ground_truth_membership[j])
                acc_arr[i][j] = acc
        overall_acc = np.sum(np.max(acc_arr, axis=1))/len(membership_thresh)/len(membership_thresh[0])
        print("Sample = " + str(sample_id) + " Accuracy = " + str(overall_acc))
        
        ## Gamma plots
        for i in range(gamma_arr.shape[0]):
            plt.clf()
            for j in range(gamma_arr.shape[1]):
                plt.plot(gamma_arr[i][j], marker = 'o')        
            plt.legend(unique_groups, fontsize = 14)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Gamma', fontsize = 14)
            plt.ylim(0,2e-4)
            plt.show()
            plt.savefig('gamma_' + str(i) + '_iter_' + str(epoch) + '.png')
            plt.close()  
        print("log-likelihood = " + str(log_likelihood_arr[-1]))

        # Early-stopping
        if epoch > 20: ##min-iters = 10
            if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.000001:
                break ## stop if log-likelihood isn't changing much
    
    print("Sample = " + str(sample_id) + " Epochs = " + str(epoch) + " Total time = " + str(time.time() - start_time) + " EM time = " + str(time.time() - start_time_em))

    ## Calibration plots
    mapping = np.argmax(acc_arr, axis = 1)
    y, x = calibration_curve(ground_truth_membership[mapping[0]], own_membership[0], n_bins = 20)
    plt.clf()
    plt.scatter(x, y, marker = 'o')
    plt.ylabel('True Probability')
    plt.xlabel('Predicted Probability')
    plt.savefig('calibration_plot.png')
    plt.close()
    plt.clf()      
    plt.plot(log_likelihood_arr)
    plt.savefig('sgdp_log_likelihood.png')
    plt.close()
    plt.clf()
    plt.figure(figsize=(40,4))
    sns.heatmap(own_membership)
    plt.savefig('own_membership_' + str(sample_id) + '.png')
    plt.close()
    plt.clf()
    plt.figure(figsize=(40,4))
    sns.heatmap(ground_truth_membership)
    plt.savefig('ground_truth_membership_' + str(sample_id) + '.png')
    plt.close()
    plt.clf()
    plt.figure(figsize=(40,4))
    sns.heatmap(np.array(tree_size).reshape(1,-1))
    plt.savefig('tree_size_' + str(sample_id) + '.png')
    plt.close()
    plt.clf()
    plt.figure(figsize=(40,4))
    sns.heatmap(np.array(no_of_mutations).reshape(1,-1))
    plt.savefig('no_of_mutations_' + str(sample_id) + '.png')
    plt.close()
    plt.clf()
    plt.figure(figsize=(40,4))
    sns.heatmap(np.array(tmrca).reshape(1,-1))
    plt.savefig('tmrca_' + str(sample_id) + '.png')
    plt.close()
    return overall_acc

acc = 0
count = 0
# sample_id = 253 ## S_Maori-1
sample_id = [399, 173]  ##S_Mbuti-3 (21)/S_English-1(399/401) and S_Han-1  ### 401 for SGDP_NEA
acc += main(sample_id, plot=False, gamma_arr =  None)
count += 1   
print("Average accuracy = " + str(acc/count)) 





