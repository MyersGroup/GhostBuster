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

def make_ground_truth(ts, num_trees, sample = None, chrs = None):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")
    ground_truth_membership_one_hot = np.zeros((6, num_trees))
    last_end = 0
    for chr in chrs:
        ground_truth = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/output/local_ancestry_chr' + str(chr) +'_' +str(sample)+'.csv', names = ['startpos', 'endpos', 'dest'])
        tree = ts.first()
        num_tree = 0
        for tid in range(len(list(ts.trees()))): #len(list(ts.trees()))
            if tree.num_sites > 0:
                for j in range(len(ground_truth)):
                    if tree.interval[0] >= ground_truth['startpos'].loc[j] and tree.interval[1] <= ground_truth['endpos'].loc[j]:
                        ground_truth_membership_one_hot[int(ground_truth['dest'].loc[j]) - 1, num_tree] = 1
                num_tree += 1
            tree.next()
    print("Done in " + str(time.time() - start_time))
    return ground_truth_membership_one_hot

def fixed_parameters(ts, membership, num_trees, target_seq_):
    eps = 1e-20
    tree = ts.first()
    num_samples = len(list(tree.samples()))
    coal_count = np.zeros((len(membership), len(epoch_intervals_pow)-1, num_trees))
    opportunity = np.zeros((len(membership), len(epoch_intervals_pow)-1, num_trees))
    proportion_of_coalescing_all = []
    coalescene_times_all = []
    epoch_index_all = []
    count_mut_trees = -1
    for tid in tqdm(range(len(list(ts.trees())))): #len(list(ts.trees()))
        if tree.num_sites == 0:
            tree.next()
            continue
        count_mut_trees += 1
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

        lineage_content = np.zeros((2*num_samples -1, len(membership)))
        target_seq = target_seq_
        for m in membership:
            lineage_content[m[0]:m[1], m[2]] = 1
        lineage_content[target_seq] = 0 ## setting lineage content of target sequence = 0 
        prev_branch_length = np.sum(lineage_content, axis = 0) #np.sum(lineage_content[:,1])
        ## lets calculate for group-1 for one tree
        proportion_of_coalescing_in_tree = []
        coalescene_times_in_tree = []
        epoch_index_in_tree = []
        event_count = 0
        for epoch in range(len(epoch_intervals_pow) -1):
            coal_events_submatrix = coal_events_matrix[(coal_events_matrix[:, 3] >= epoch_intervals_pow[epoch]) & (coal_events_matrix[:, 3] < epoch_intervals_pow[epoch+1])]
            tprev = epoch_intervals_pow[epoch]
            for (a,b,c,t) in coal_events_submatrix:
                event_count += 1
                a = int(a)
                b = int(b)
                c = int(c)
                # opportunity[epoch] += (t-tprev)*np.sum(lineage_content[:,1]/(np.sum(lineage_content, axis = 1) + eps))
                opportunity[:,epoch,count_mut_trees] += (t-tprev)*(prev_branch_length) 
                if a == target_seq:
                    proportion_of_coalescing = lineage_content[b]/(sum(lineage_content[b]))
                    coal_count[:,epoch,count_mut_trees] += proportion_of_coalescing 
                    target_seq = c
                    lineage_content[c] = 0
                    proportion_of_coalescing_in_tree.append(proportion_of_coalescing)
                    epoch_index_in_tree.append(epoch)
                    prev_branch_length = prev_branch_length - lineage_content[b]/(sum(lineage_content[b]))
                elif b == target_seq:
                    proportion_of_coalescing = lineage_content[a]/(sum(lineage_content[a])) ## sum() faster than np.sum()
                    coal_count[:,epoch,count_mut_trees] += proportion_of_coalescing  
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
    return coal_count, opportunity, proportion_of_coalescing_all, epoch_index_all


def estimate_gamma_matrix(own_membership, num, denom):
    ## M-step for MLE estimation
    eps = 1e-20
    ne_arr = np.zeros(len(epoch_intervals) - 1)
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    denom_2 = np.zeros(len(epoch_intervals) - 1)
    for epoch in range(len(epoch_intervals) - 1):
        ne_arr[epoch] = sum(num[epoch]*own_membership)#[membership[0]: membership[1]].sum()
    for epoch in range(len(epoch_intervals) - 1):
        denom_1[epoch] = sum(denom[epoch]*own_membership)#[membership[0]: membership[1]].sum()
    for epoch in range(len(epoch_intervals) -1):
        for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
            denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
    return ne_arr, denom_2 + denom_1 + eps

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

def compute_tree_stats(ts, chrs):
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
        recomb_map = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/recomb_maps/msprime_maps/genetic_map_GRCh37_chr'+ str(chr) +'.txt.gz', sep = '\t')
        relate_quality_output = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/relate_ne_chr' + str(chr) + '.qual', sep = ' ')
        tree = ts.first()
        for tid in range(ts.num_trees): #len(list(ts.trees()))
            if tree.num_sites > 0:
                recomb_events = recomb_map[(recomb_map['Position(bp)'] < tree.interval[1]) & (recomb_map['Position(bp)'] >= tree.interval[0])]
                recomb_rates.append(np.mean(recomb_events['Rate(cM/Mb)']))
                num_trees += 1
                tree_size.append(tree.interval[1] - tree.interval[0])
                no_of_mutations.append(tree.num_mutations)  ###changed to mutations from sites
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
    # membership = [(0,10,0), (10,110,1), (110,210,2), (210,310,3), (310,410,4), (410,510,5), (510,610,6)]   ## (startpos, endpos, groupid)
    # membership = [(0,1,0), (1,101,1), (101,201,2), (201,301,3)]   ## (startpos, endpos, groupid) ## only considering D + E+ F
    membership = [(0,1,0), (1,101,1), (101,201,2)]   ## (startpos, endpos, groupid) ## only considering A + B
    ts = tskit.load('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/relate_ne_chr5.trees')
    ts = ts.simplify([sample_id] + np.arange(10,210).tolist())  ## only considering A + B
    # ts = ts.simplify([sample_id] + np.arange(310,610).tolist())  ## only considering D + E + F
    tree = ts.first()
    num_trees = 0
    for tid in range(len(list(ts.trees()))): #len(list(ts.trees()))
        if tree.num_sites > 0:
            num_trees += 1
        tree.next()
    print("Total number of trees = " + str(num_trees))
    ground_truth_membership = make_ground_truth(ts, num_trees, sample = sample_id, chrs = [5])
    own_membership = np.random.dirichlet(np.ones(num_clusters), num_trees).T # ground_truth_membership[0:3] #
    tree_size, no_of_mutations, tmrca, recomb_rates, frac_branches_with_snp, num_snps_on_tree, fraction_snps_not_mapping = compute_tree_stats(ts, chrs = [5])
    mask_dodgy = mask_for_dodgy_trees(frac_branches_with_snp, num_snps_on_tree)
    print("Trees with high certainty = " + str(np.sum(mask_dodgy)))
    num, denom, proportion_of_coalescing_all, epoch_index_all = fixed_parameters(ts, membership, num_trees, sample_id)
    log_likelihood_arr = []
    start_time_em = time.time()
    eps = 1e-100
    print("Starting the EM..")
    for epoch in range(40):  ## max-iters = 40
        gamma_arr = np.zeros((len(own_membership), len(membership), len(epoch_intervals) - 1))        
        for j in range(len(own_membership)):
            if epoch == 0:
                n = compute_gamma_num(own_membership[j]*mask_dodgy, None, proportion_of_coalescing_all, epoch_index_all, len(membership))
            else:
                n = compute_gamma_num(own_membership[j]*mask_dodgy, prev_gamma[j], proportion_of_coalescing_all, epoch_index_all, len(membership)) #compute_gamma_num(own_membership[j], prev_gamma[j], proportion_of_coalescing_all, epoch_index_all, len(unique_groups))
            for i in range(len(membership)):
                d = compute_gamma_denom(own_membership[j]*mask_dodgy, denom[i])
                gamma_arr[j][i] = n[i]/d #n/d #
        prev_gamma = gamma_arr
        print(gamma_arr)
        tau = np.ones(len(own_membership))/len(own_membership)
        for j in range(len(own_membership)):
            tau[j] = np.clip(np.sum(own_membership[j])/own_membership[j].shape[0],1e-10, 1-1e-10)
        print(tau)

        ## E-step
        prod_term = np.zeros((len(own_membership), len(membership), len(epoch_intervals) - 1))
        for k in range(len(membership)):
            for j in range(len(own_membership)):
                prod_term[j][k] = get_prod_term(gamma_arr[j][k])

        own_membership_update = np.ones((len(own_membership), num_trees))

        log_num_em = np.zeros((len(own_membership), num_trees))
        log_denom_em = np.zeros((len(own_membership), num_trees))
        for tid in range(num_trees):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                for j in range(len(own_membership)):
                    log_num_em[j, tid] += np.log(np.sum(gamma_arr[j,:,epoch_index_in_tree[i]]*proportion_of_coalescing_in_tree[i]))
                    # log_likelihood += np.sum(np.log(gamma_arr[j,:,epoch_index_in_tree[i]] + eps)*proportion_of_coalescing_in_tree[i])  ## n_ig*log(\gamma)
            max_epoch_index = int(np.minimum(epoch_index_in_tree[i]+1, len(epoch_intervals_pow) - 1))
            for j in range(len(own_membership)):
                log_denom_em[j, tid] = -np.sum(gamma_arr[j,:,0:max_epoch_index]*denom[:,0:max_epoch_index,tid])  ##-np.sum(gamma_arr[j,:,:]*denom[:,:,tid])  ## summing only till the maximum epoch in that tree
                # log_likelihood += log_denom_em[j, tid]
        own_membership_update = np.exp(log_num_em + log_denom_em - np.repeat(np.max(log_num_em + log_denom_em, axis = 0).reshape(-1,1), len(own_membership), axis = 1).T)#np.exp(log_num_em + log_denom_em)
        for j in range(len(own_membership)):
            own_membership_update[j] *= tau[j]
        log_likelihood = np.sum(np.log(np.sum(own_membership_update, axis = 0)) + np.max(log_num_em + log_denom_em, axis = 0))
        log_likelihood_arr.append(log_likelihood)
        # print("log-like = " + str(np.sum(np.log(np.sum(own_membership_update, axis = 0)) + np.max(log_num_em + log_denom_em, axis = 0))))
        own_membership_update = own_membership_update/(np.sum(own_membership_update, axis = 0))
        own_membership = own_membership_update
        membership_thresh = make_one_hot(np.argmax(own_membership, axis = 0), len(own_membership))

        ## Evaluate accuracy
        acc_arr = np.zeros((len(own_membership), 3))
        for i in range(len(own_membership)):
            for j in range(0,3): ###hard-coding: because onle first three rows have ground-truth information
                acc = np.sum(membership_thresh[i] == ground_truth_membership[j])
                acc_arr[i][j] = acc
        overall_acc = np.sum(np.max(acc_arr, axis=1))/len(membership_thresh)/len(membership_thresh[0])
        print("Sample = " + str(sample_id) + " Accuracy = " + str(overall_acc))
        proportion_of_coalescing_top2 = np.zeros((num_trees, 2, len(membership)))
        for tid in range(num_trees):
            proportion_of_coalescing_top2[tid,:,:] = proportion_of_coalescing_all[tid][0:2]
        recomb_x_all = []
        recomb_y_all = []
        size_x_all = []
        size_y_all = []
        muts_x_all = []
        muts_y_all = []
        frac_branch_x_all = []
        num_snps_x_all = []
        frac_snps_x_all = []
        for i in range(len(own_membership)):
            tree_size_i = np.array(tree_size)[np.argmax(own_membership, axis = 0) == i]
            num_mutations_i = np.array(no_of_mutations)[np.argmax(own_membership, axis = 0) == i]
            recomb_rates_i = np.array(recomb_rates)[np.argmax(own_membership, axis = 0) == i]
            recomb_rates_i = recomb_rates_i[~ np.isnan(np.array(recomb_rates_i))]
            frac_branches_with_snp_i = np.array(frac_branches_with_snp)[np.argmax(own_membership, axis = 0) == i]
            num_snps_on_tree_i = np.array(num_snps_on_tree)[np.argmax(own_membership, axis = 0) == i]
            fraction_snps_not_mapping_i = np.array(fraction_snps_not_mapping)[np.argmax(own_membership, axis = 0) == i]
            frac_branch_x_all.extend(frac_branches_with_snp_i)
            num_snps_x_all.extend(num_snps_on_tree_i)
            frac_snps_x_all.extend(fraction_snps_not_mapping_i)
            recomb_x_all.extend(recomb_rates_i)
            size_x_all.extend(tree_size_i)
            muts_x_all.extend(num_mutations_i)
            recomb_y_all.extend(np.repeat(i, len(recomb_rates_i)))
            size_y_all.extend(np.repeat(i, len(tree_size_i)))
            muts_y_all.extend(np.repeat(i, len(num_mutations_i)))
            proportion_of_coalescing_i = proportion_of_coalescing_top2[np.argmax(own_membership, axis = 0) == i]
            print("Cluster: " + str(i) + " Median tree size: " + str(np.median(tree_size_i)) + " Median num of muts: " + str(np.median(num_mutations_i)) + " Median recomb rate: " + str(np.median(recomb_rates_i)) + " Median frac_branches_with_snp: " + str(np.median(frac_branches_with_snp_i)) + " Median num_snps_on_tree: " + str(np.median(num_snps_on_tree_i)) + " Median fraction_snps_not_mapping: " + str(np.median(fraction_snps_not_mapping_i)) )
            print( " Mean 1st coal. proportion: " + str(np.mean(proportion_of_coalescing_i[:,0,:], axis = 0)) + " Mean 2nd coal. proportion: " + str(np.mean(proportion_of_coalescing_i[:,1,:], axis = 0)))
        
        plt.clf()
        sns.boxplot(recomb_y_all, recomb_x_all)
        plt.ylim(0, 5*np.median(recomb_x_all))
        plt.savefig('recomb_rates_box.png')
        plt.clf()
        sns.boxplot(size_y_all, size_x_all)
        plt.ylim(0, 5*np.median(size_x_all))
        plt.savefig('tree_size_box.png')
        plt.clf()
        sns.boxplot(muts_y_all, muts_x_all)
        plt.ylim(0, 5*np.median(muts_x_all))
        plt.savefig('num_muts_box.png')
        plt.clf()
        sns.boxplot(muts_y_all, frac_branch_x_all)
        plt.ylim(0, 5*np.median(frac_branch_x_all))
        plt.savefig('frac_branches_with_snp_box.png')
        plt.clf()
        sns.boxplot(muts_y_all, num_snps_x_all)
        plt.ylim(0, 5*np.median(num_snps_x_all))
        plt.savefig('num_snps_on_tree_box.png')
        plt.clf()
        sns.boxplot(muts_y_all, frac_snps_x_all)
        plt.ylim(0, 5*np.median(frac_snps_x_all))
        plt.savefig('fraction_snps_not_mapping_box.png')

        ## Early-stopping
        print("log-likelihood = " + str(log_likelihood_arr[-1]))
        if epoch > 20: ##min-iters = 10
            if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.00001:
                break ## stop if log-likelihood isn't changing much
    
    print("Sample = " + str(sample_id) + " Epochs = " + str(epoch) + " Total time = " + str(time.time() - start_time) + " EM time = " + str(time.time() - start_time_em))

    return gamma_arr, log_likelihood_arr, own_membership, ground_truth_membership

acc = 0
sample_id = 0
gamma_arr, log_likelihood_arr, own_membership, ground_truth_membership = main(sample_id, plot=False, gamma_arr =  None)
for i in range(gamma_arr.shape[0]):
    plt.clf()
    for j in range(gamma_arr.shape[1]):
        plt.plot(gamma_arr[i][j], marker = 'o')        
    plt.legend(['Admix', 'pop A', 'pop B'], fontsize = 14)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Gamma', fontsize = 14)
    plt.ylim(0,2e-4)
    plt.show()
    plt.savefig('ghost_relate_sim_true_gamma_' + str(i) + '.png')
    plt.close()  
plt.clf()      
plt.plot(log_likelihood_arr)
plt.savefig('ghost_relate_sim_true_log_likelihood.png')
plt.close()
plt.clf()
plt.figure(figsize=(40,4))
sns.heatmap(own_membership)
plt.savefig('ghost_relate_sim_true_own_membership_' + str(sample_id) + '.png')
plt.close()
plt.clf()
plt.figure(figsize=(40,4))
sns.heatmap(ground_truth_membership)
plt.savefig('ghost_relate_sim_true_ground_truth_membership_' + str(sample_id) + '.png')
plt.close()