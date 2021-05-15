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
from sklearn.calibration import calibration_curve


epoch_intervals = np.array([-np.inf] + np.linspace(3 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])
epoch_intervals_pow = np.power(10, epoch_intervals)

def make_one_hot(X, max_X):
    X = np.array(X, dtype='int')
    classes = np.arange(0, max_X,1)
    # Y = []
    if len(X.shape) == 2:
        Y = np.zeros((len(classes), X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        Y = np.zeros((len(classes), X.shape[0]))
    for c in classes:
        # Y.append(scipy.sparse.csr_matrix(np.array(X == c, dtype='int')))
        Y[c] = np.array(X == c, dtype='int')
    return Y

def make_ground_truth(num_trees, sample = None, chrs = None):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")
    if sample != None:
        ground_truth_membership_one_hot = np.zeros((6, num_trees))
        if sample >= 10 and sample < 110:
            ground_truth_membership_one_hot[0] = 1
        if sample >= 110 and sample < 210:
            ground_truth_membership_one_hot[1] = 1
        if sample >= 210 and sample < 310:
            ground_truth_membership_one_hot[2] = 1
        if sample >= 310 and sample < 410:
            ground_truth_membership_one_hot[3] = 1
        if sample >= 410 and sample < 510:
            ground_truth_membership_one_hot[4] = 1
        if sample >= 510 and sample < 610:
            ground_truth_membership_one_hot[5] = 1
        if sample < 10:
            last_end = 0
            for chr in chrs:
                ground_truth = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/output/local_ancestry_chr' + str(chr) +'_' +str(sample)+'.csv', names = ['startpos', 'endpos', 'dest'])
                df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/input_files/relate_ne_chr' + str(chr) + '_'+str(sample) +'.coaltimes', sep = ' ', usecols=['startpos', 'endpos'])
                for j in range(len(ground_truth)):
                    indices = np.array(df[(df['startpos'] >= ground_truth['startpos'].loc[j]) & (df['endpos']  <= ground_truth['endpos'].loc[j])].index.tolist(), dtype=np.int64)
                    ground_truth_membership_one_hot[int(ground_truth['dest'].loc[j]) - 1, indices+last_end] = 1
                last_end += df.shape[0]
    print("Done in " + str(time.time() - start_time))
    return ground_truth_membership_one_hot

def make_ground_truth(ts, num_trees, sample = None, chrs = None):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")
    ground_truth_membership_one_hot = np.zeros((6, num_trees))
    last_end = 0
    for chr in chrs:
        ground_truth = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/output/new_local_ancestry_chr' + str(chr) +'_' +str(sample)+'.csv', names = ['startpos', 'endpos', 'dest'])
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

def get_prod_term(gamma_arr):
    prod_term_arr = []
    for i in range(0, len(epoch_intervals)-1):
        prod_term = 0
        for j in range(1, i+1):
            prod_term -= gamma_arr[j-1]*(epoch_intervals_pow[j] - epoch_intervals_pow[j-1])
        prod_term_arr.append(prod_term)
    return prod_term_arr

def coal_rate_log_pdf_per_tree(t_arr, gamma_arr, epoch_index, prod_term_arr):
    eps = 1e-322
    log_pdf = np.log(eps + gamma_arr[epoch_index]) - gamma_arr[epoch_index]*(np.array(t_arr)- epoch_intervals_pow[epoch_index])
    log_pdf += np.array(prod_term_arr)[epoch_index]
    return log_pdf ## gives product of PDFs


def coal_rate_log_pdf(gamma_arr, t_arr, epoch_t_index, prod_term_arr):
    ## The likelihood for the coalescent 
    eps = 1e-20
    log_pdf = np.log(eps + gamma_arr[epoch_t_index]) - gamma_arr[epoch_t_index]*(np.array(t_arr)- epoch_intervals_pow[epoch_t_index])
    log_pdf += np.array(prod_term_arr)[epoch_t_index]
    return log_pdf ## gives product of PDFs

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

def main(sample_id, plot = False, gamma_arr = None):
    start_time = time.time()
    num_clusters = 3
    membership = [(0,10,0), (10,110,1), (110,210,2), (210,310,3), (310,410,4), (410,510,5), (510,610,6)]   ## (startpos, endpos, groupid)
    ts = tskit.load('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/example_chr18.trees')
    tree = ts.first()
    num_trees = 0
    for tid in range(len(list(ts.trees()))): #len(list(ts.trees()))
        if tree.num_sites > 0:
            num_trees += 1
        tree.next()
    print("Total number of trees = " + str(num_trees))
    ground_truth_membership = make_ground_truth(ts, num_trees, sample = sample_id, chrs = [18]) 
    own_membership = np.random.dirichlet(np.ones(num_clusters), num_trees).T #  ground_truth_membership[0:3] #
    num, denom, proportion_of_coalescing_all, epoch_index_all = fixed_parameters(ts, membership, num_trees, sample_id)
    log_likelihood_arr = []
    start_time_em = time.time()
    eps = 1e-100
    print("Starting the EM..")
    for epoch in range(30):  ## max-iters = 40
        gamma_arr = np.zeros((len(own_membership), len(membership), len(epoch_intervals) - 1))        
        for j in range(len(own_membership)):
            if epoch == 0:
                n = compute_gamma_num(own_membership[j], None, proportion_of_coalescing_all, epoch_index_all, len(membership))
            else:
                n = compute_gamma_num(own_membership[j], prev_gamma[j], proportion_of_coalescing_all, epoch_index_all, len(membership)) #compute_gamma_num(own_membership[j], prev_gamma[j], proportion_of_coalescing_all, epoch_index_all, len(unique_groups))
            for i in range(len(membership)):
                d = compute_gamma_denom(own_membership[j], denom[i])
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
            max_epoch_index = int(np.minimum(epoch_index_in_tree[i]+1, len(epoch_intervals_pow) - 1))
            for j in range(len(own_membership)):
                log_denom_em[j, tid] = -np.sum(gamma_arr[j,:,0:max_epoch_index]*denom[:,0:max_epoch_index,tid])  ##-np.sum(gamma_arr[j,:,:]*denom[:,:,tid])  ## summing only till the maximum epoch in that tree
        own_membership_update = np.exp(log_num_em + log_denom_em - np.repeat(np.max(log_num_em + log_denom_em, axis = 0).reshape(-1,1), len(own_membership), axis = 1).T)#np.exp(log_num_em + log_denom_em)
        for j in range(len(own_membership)):
            own_membership_update[j] *= tau[j]
        log_likelihood = np.sum(np.log(np.sum(own_membership_update, axis = 0)) + np.max(log_num_em + log_denom_em, axis = 0))
        log_likelihood_arr.append(log_likelihood)
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

        ## Early-stopping
        print("log-likelihood = " + str(log_likelihood_arr[-1]))
        if epoch > 20: ##min-iters = 10
            if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.00001:
                break ## stop if log-likelihood isn't changing much
    
    print("Sample = " + str(sample_id) + " Epochs = " + str(epoch) + " Total time = " + str(time.time() - start_time) + " EM time = " + str(time.time() - start_time_em))
    
    mapping = np.argmax(acc_arr, axis = 1)
    return gamma_arr, log_likelihood_arr, own_membership, ground_truth_membership, mapping

acc = 0
sample_id = 0
gamma_arr, log_likelihood_arr, own_membership, ground_truth_membership, mapping = main(sample_id, plot=False, gamma_arr =  None)
for i in range(gamma_arr.shape[0]):
    plt.clf()
    for j in range(gamma_arr.shape[1]):
        plt.plot(gamma_arr[i][j], marker = 'o')        
    plt.legend(['Admix', 'pop A', 'pop B', 'pop C', 'pop D', 'pop E', 'pop F'], fontsize = 14)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Gamma', fontsize = 14)
    plt.ylim(0, 2e-4)
    plt.show()
    plt.savefig('sim_true_gamma_' + str(i) + '.png')
    plt.close()  

plt.clf()
for i in range(len(own_membership)):
    y, x = calibration_curve(ground_truth_membership[mapping[i]], own_membership[i], n_bins = 20)
    plt.plot(x, y, marker = 'o')
plt.plot(x, x, ':')
plt.ylabel('True Probability')
plt.xlabel('Predicted Probability')
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.savefig('calibration_plot_relate.png')
plt.close()
plt.clf()
sns.distplot(own_membership[0][np.array(ground_truth_membership[mapping[0]], dtype = 'bool')])
plt.xlim(0,1)
plt.xlabel('Predicted Probability')
plt.savefig('posterior_prob_hist.png')
plt.close()
plt.clf()      
plt.plot(log_likelihood_arr)
plt.savefig('sim_true_log_likelihood.png')
plt.close()
plt.clf()
plt.figure(figsize=(40,4))
sns.heatmap(own_membership)
plt.savefig('sim_true_own_membership_' + str(sample_id) + '.png')
plt.close()
plt.clf()
plt.figure(figsize=(40,4))
sns.heatmap(ground_truth_membership)
plt.savefig('sim_true_ground_truth_membership_' + str(sample_id) + '.png')
plt.close()









