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

epoch_intervals = np.array([-np.inf] + np.linspace(3 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])
epoch_intervals_pow = np.power(10, epoch_intervals)

## only for unidirectional flow
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

def make_ground_truth(df, sample = None, chrs = None):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")
    if sample != None:
        ground_truth_membership_one_hot = np.zeros((6, df.shape[1]))
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
                df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/input_files/example_chr' + str(chr) + '_'+str(sample) +'.coaltimes', sep = ' ', usecols=['startpos', 'endpos'])
                for j in range(len(ground_truth)):
                    indices = np.array(df[(df['startpos'] >= ground_truth['startpos'].loc[j]) & (df['endpos']  <= ground_truth['endpos'].loc[j])].index.tolist(), dtype=np.int64)
                    ground_truth_membership_one_hot[int(ground_truth['dest'].loc[j]) - 1, indices+last_end] = 1
                last_end += df.shape[0]
    else:
        ground_truth_membership = np.ones((610, df.shape[0])) ## contribution of group 1 in group 2
        for i in range(0, 10):
            ground_truth = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/output/local_ancestry_chr1_'+str(i)+'.csv', names = ['startpos', 'endpos', 'dest'])
            for j in range(len(ground_truth)):
                indices = df[(df['startpos'] >= ground_truth['startpos'].loc[j]) & (df['endpos']  <= ground_truth['endpos'].loc[j])].index
                ground_truth_membership[i,indices] = ground_truth['dest'].loc[j]
        for i in range(10,110):
            ground_truth_membership[i] = 1
        for i in range(110,210):
            ground_truth_membership[i] = 2
        for i in range(210,310):
            ground_truth_membership[i] = 3
        for i in range(310,410):
            ground_truth_membership[i] = 4
        for i in range(410,510):
            ground_truth_membership[i] = 5 
        for i in range(510,610):
            ground_truth_membership[i] = 6   
        ## one-hot encoding
        ground_truth_membership = np.array(ground_truth_membership, dtype='int')
        classes = np.unique(ground_truth_membership)
        ground_truth_membership_one_hot = np.zeros((len(classes), ground_truth_membership.shape[0], ground_truth_membership.shape[1]))
        count = 0
        for c in classes:
            ground_truth_membership_one_hot[count] = np.array(ground_truth_membership == c, dtype='int')
            count += 1
    print("Done in " + str(time.time() - start_time))
    return ground_truth_membership_one_hot

def make_epoch_t_index(df_arr, cols):
    ## Finds the coalesceing epoch for each individual and each tree
    start_time = time.time()
    print("Bining the coalescene times into epochs..")
    epoch_t_index = np.zeros((len(cols), df_arr.shape[1]), dtype = np.int16)
    for i in range(len(cols)):
        t_arr = df_arr[i]#.values
        mask = np.array(np.repeat(epoch_intervals, len(t_arr)).reshape(-1,len(t_arr)) < np.log(t_arr)/np.log(10), dtype='int')
        epoch_t_index[i] = np.array(np.sum(mask, axis = 0) - 1, dtype=np.int16) 
        # np.argmax(epoch_intervals[epoch_intervals < np.log(t_arr[t])/np.log(10)])
    print("Done in " + str(time.time() - start_time))
    return np.array(epoch_t_index, dtype=np.int16)

def count_ties(df_arr, cols):
    ## Computes the number of ties in coalescent times for a given tree
    start_time = time.time()
    print("Calculating the ties in coalescene time..")
    ni = np.ones((len(cols), df_arr.shape[1]), dtype = np.int16)  ##hard-coding: one sample at a time
    # df_arr = df[cols].values
    for j in range(df_arr.shape[1]):
        t_arr = df_arr[:,j].tolist()
        count_dict = Counter(t_arr)
        for i in range(len(t_arr)):
            ni[i, j] = np.int16(count_dict[t_arr[i]])
    print("Done in " + str(time.time() - start_time))
    return ni

def get_prod_term(gamma_arr):
    prod_term_arr = []
    for i in range(0, len(epoch_intervals)-1):
        prod_term = 0
        for j in range(1, i+1):
            prod_term -= gamma_arr[j-1]*(epoch_intervals_pow[j] - epoch_intervals_pow[j-1])
        prod_term_arr.append(prod_term)
    return prod_term_arr

# def coal_rate_log_pdf(gamma_arr, t_arr, epoch_t_index):
#     ## The likelihood for the coalescent 
#     eps = 1e-20
#     prod_term_arr = []
#     for i in range(0, len(epoch_intervals)-1):
#         prod_term = 0
#         for j in range(1, i+1):
#             prod_term -= gamma_arr[j-1]*(epoch_intervals_pow[j] - epoch_intervals_pow[j-1])
#         prod_term_arr.append(prod_term)
#     log_pdf = np.log(eps + gamma_arr[epoch_t_index]) - gamma_arr[epoch_t_index]*(np.array(t_arr)- epoch_intervals_pow[epoch_t_index])
#     log_pdf += np.array(prod_term_arr)[epoch_t_index]
#     return log_pdf ## gives product of PDFs

def coal_rate_log_pdf(gamma_arr, t_arr, epoch_t_index, prod_term_arr):
    ## The likelihood for the coalescent 
    eps = 1e-20
    # prod_term_arr = []
    # for i in range(0, len(epoch_intervals)-1):
    #     prod_term = 0
    #     for j in range(1, i+1):
    #         prod_term -= gamma_arr[j-1]*(epoch_intervals_pow[j] - epoch_intervals_pow[j-1])
    #     prod_term_arr.append(prod_term)
    log_pdf = np.log(eps + gamma_arr[epoch_t_index]) - gamma_arr[epoch_t_index]*(np.array(t_arr)- epoch_intervals_pow[epoch_t_index])
    log_pdf += np.array(prod_term_arr)[epoch_t_index]
    return log_pdf ## gives product of PDFs

def compute_coalscene_event_matrix(df):
    r1 = []
    for epoch in range(len(epoch_intervals) - 1):
        cond = (df < epoch_intervals_pow[epoch + 1]) & (df > epoch_intervals_pow[epoch])
        r1.append(scipy.sparse.csr_matrix(np.array(cond, dtype='int')))
    return r1

# def fixed_parameters(df, ni, cols, r1, membership):
#     ## Computes the fixed parameters in the M-step to speed up computation
#     membership_scaled = membership/ni
#     membership_scaled = membership_scaled[np.array(cols, dtype='int')]
#     # r1 = np.zeros((len(epoch_intervals) - 1, df.shape[0], df.shape[1]))
#     # r1 = []
#     # ne_arr = np.zeros((len(epoch_intervals) - 1, df.shape[0], df.shape[1]))
#     ne_arr = []
#     # denom_1 = np.zeros((len(epoch_intervals) - 1, df.shape[0], df.shape[1]))
#     denom_1 = []
#     # for epoch in range(len(epoch_intervals) - 1):
#     #     cond = (df < epoch_intervals_pow[epoch + 1]) & (df > epoch_intervals_pow[epoch])
#     #     r1.append(scipy.sparse.coo_matrix(np.array(cond, dtype='int')))
#     for epoch in range(len(epoch_intervals) - 1):
#         cond = (df < epoch_intervals_pow[epoch + 1]) & (df > epoch_intervals_pow[epoch])
#         ne_arr.append(scipy.sparse.coo_matrix(np.array(membership_scaled*cond, dtype='float')))
#         denom_1.append(scipy.sparse.coo_matrix(np.array((df - epoch_intervals_pow[epoch]), dtype='float')*cond*membership_scaled))
#     return ne_arr, denom_1

# def estimate_gamma_matrix(own_membership, num, denom):
#     ## M-step for MLE estimation
#     eps = 1e-20
#     ne_arr = np.zeros(len(epoch_intervals) - 1)
#     denom_1 = np.zeros(len(epoch_intervals) - 1)
#     denom_2 = np.zeros(len(epoch_intervals) - 1)
#     for epoch in range(len(epoch_intervals) - 1):
#         ne_arr[epoch] = (num[epoch]@own_membership).sum()
#     for epoch in range(len(epoch_intervals) - 1):
#         denom_1[epoch] = (denom[epoch]@own_membership).sum()
#     for epoch in range(len(epoch_intervals) -1):
#         for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
#             denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
#     return ne_arr, denom_2 + denom_1 + eps

def fixed_parameters(df, ni, cols, membership):
    ## Computes the fixed parameters in the M-step to speed up computation
    start_time = time.time()
    print("Calculating the fixed parameters for EM..")
    # membership_scaled = 1/ni
    # membership_scaled = membership_scaled[np.array(cols, dtype='int')]
    # ni = ni[np.array(cols, dtype='int')]
    ne_arr = [[] for _ in range(len(membership))]
    denom_1 = [[] for _ in range(len(membership))]
    for epoch in range(len(epoch_intervals) - 1):
        cond = (df < epoch_intervals_pow[epoch + 1]) & (df > epoch_intervals_pow[epoch])
        for k in range(len(membership)):
            cond_k = cond[membership[k][0]:membership[k][1]]
            ni_k = np.array(ni[membership[k][0]:membership[k][1]], dtype='float')
            num_k = np.sum(cond_k/ni_k, axis = 0)
            denom_k = (df[membership[k][0]:membership[k][1]] - epoch_intervals_pow[epoch])*cond_k/ni_k
            denom_k = np.sum(denom_k, axis = 0)
            ne_arr[k].append(scipy.sparse.coo_matrix(num_k))
            denom_1[k].append(scipy.sparse.coo_matrix(denom_k))
            # ne_arr.append(scipy.sparse.coo_matrix(np.array(cond_k/ni_k, dtype='float')))
            # denom_1.append(scipy.sparse.coo_matrix(np.array((df - epoch_intervals_pow[epoch]), dtype='float')*cond_k/ni_k))
    print("Done in " + str(time.time() - start_time))
    return ne_arr, denom_1

def estimate_gamma_matrix(own_membership, num, denom):
    ## M-step for MLE estimation
    eps = 1e-20
    ne_arr = np.zeros(len(epoch_intervals) - 1)
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    denom_2 = np.zeros(len(epoch_intervals) - 1)
    for epoch in range(len(epoch_intervals) - 1):
        ne_arr[epoch] = (num[epoch]@own_membership)[0]#[membership[0]: membership[1]].sum()
    for epoch in range(len(epoch_intervals) - 1):
        denom_1[epoch] = (denom[epoch]@own_membership)[0]#[membership[0]: membership[1]].sum()
    for epoch in range(len(epoch_intervals) -1):
        for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
            denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
    return ne_arr, denom_2 + denom_1 + eps

# def estimate_gamma(df, ni, own_membership, cols, membership):
#     # membership = np.array(membership, dtype='bool')
#     eps = 1e-20
#     gamma_arr = np.zeros(len(epoch_intervals) - 1)
#     ne_arr = np.zeros(len(epoch_intervals) - 1)
#     denom_1 = np.zeros(len(epoch_intervals) - 1)
#     denom_2 = np.zeros(len(epoch_intervals) - 1)
#     membership_scaled = membership/ni
#     # own_membership = np.array(own_memberhsip, dtype='bool')
#     for epoch in range(len(epoch_intervals) - 1):
#         for i in range(len(cols)):
#             r1 = (df[i] < epoch_intervals_pow[epoch + 1]) & (df[i] > epoch_intervals_pow[epoch])
#             ne_arr[epoch] += np.sum(membership_scaled[int(cols[i])][r1] * own_membership[r1])
#     for epoch in range(len(epoch_intervals) - 1):
#         for i in range(len(cols)):
#             r1 = (df[i] < epoch_intervals_pow[epoch + 1]) & (df[i] > epoch_intervals_pow[epoch])
#             denom_1[epoch] += np.sum((df[i][r1] -epoch_intervals_pow[epoch])*(membership_scaled[int(cols[i])][r1] * own_membership[r1]))
#     for epoch in range(len(epoch_intervals) -1):
#         for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
#             denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
#     gamma_arr = ne_arr/(denom_1 + denom_2 + eps)
#     return ne_arr, denom_2 + denom_1 + eps

def estimate_gamma(df, ni, own_membership, cols, membership):
    # membership = np.array(membership, dtype='bool')
    eps = 1e-20
    gamma_arr = np.zeros(len(epoch_intervals) - 1)
    ne_arr = np.zeros(len(epoch_intervals) - 1)
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    denom_2 = np.zeros(len(epoch_intervals) - 1)
    # own_membership = np.array(own_memberhsip, dtype='bool')
    for epoch in range(len(epoch_intervals) - 1):
        for i in range(membership[0], membership[1]):
            r1 = (df[i] < epoch_intervals_pow[epoch + 1]) & (df[i] > epoch_intervals_pow[epoch])
            ne_arr[epoch] += np.sum(own_membership[r1]/ni[i][r1])
    for epoch in range(len(epoch_intervals) - 1):
        for i in range(membership[0], membership[1]):
            r1 = (df[i] < epoch_intervals_pow[epoch + 1]) & (df[i] > epoch_intervals_pow[epoch])
            denom_1[epoch] += np.sum((df[i][r1] -epoch_intervals_pow[epoch])*(own_membership[r1]/ni[i][r1]))
    for epoch in range(len(epoch_intervals) -1):
        for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
            denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
    gamma_arr = ne_arr/(denom_1 + denom_2 + eps)
    return ne_arr, denom_2 + denom_1 + eps

def main(sample_id, plot = False, gamma_arr = None):
    start_time = time.time()
    num_clusters = 3
    dtype_dict = {}
    for i in range(0,610):
        dtype_dict[str(i)] = "float32"  ##open the coalescene time as dtype int16
    df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_3admix/input_files/example_merged_'+str(sample_id) +'.coaltimes', sep = ' ', dtype=dtype_dict)
    print(df.info())
    print("Dataframe shape = " + str(df.shape))
    # df = df.sample(100000).reset_index(drop=True)#df.loc[0:100000]#
    df = df.dropna(axis=1)
    cols = df.columns[2:2+sample_id].tolist() + df.columns[3+sample_id:].tolist() # run EM for 40 steps
    # cols =  df.columns[312:].tolist() #df.columns[2:2+sample_id].tolist() + df.columns[3+sample_id:12].tolist() +  #run EM for 40 steps
    print(cols)
    df_arr = df[cols].values.T
    del(df)
    ground_truth_membership = make_ground_truth(df_arr, sample = sample_id, chrs = range(1,23)) #make_ground_truth(df) # #make_ground_truth(df, sample = sample_id) ##dont change this
    epoch_t_index = make_epoch_t_index(df_arr, cols)
    ni = count_ties(df_arr, cols)
    # membership = make_one_hot(np.vstack((np.zeros((10, df.shape[0])), np.ones((100, df.shape[0])), 2*np.ones((100, df.shape[0])), 3*np.ones((100, df.shape[0])), 4*np.ones((100, df.shape[0])), 5*np.ones((100, df.shape[0])), 6*np.ones((100, df.shape[0])))))#
    membership = [(0,9),(9,109),(109,209),(209,309),(309,409),(409,509),(509,609)]
    # membership = [(0,9),(9,109),(109,209),(209,309)]
    # membership = [(0,100),(100,200),(200,300)]
    own_membership = np.random.dirichlet(np.ones(num_clusters), df_arr.shape[1]).T #make_ground_truth(df)[:,sample_id] #
    # r1 = compute_coalscene_event_matrix(df_arr)
    num, denom = fixed_parameters(df_arr, ni, cols, membership)
    log_likelihood_arr = []
    start_time_em = time.time()
    print("Starting the EM..")
    for epoch in range(40):  ## max-iters = 40
        gamma_arr = np.zeros((len(own_membership), len(membership), len(epoch_intervals) - 1))        
        ## M-step
        for i in range(len(membership)):
            for j in range(len(own_membership)):
                n,d = estimate_gamma_matrix(own_membership[j], num[i], denom[i])
                # n,d = estimate_gamma_matrix(own_membership[j], num, denom, membership[i])
                # n1, d1 = estimate_gamma(df_arr, ni, own_membership[j], cols, membership[i])
                gamma_arr[j][i] = n/d 
        print(gamma_arr)
        tau = np.ones(len(own_membership))/len(own_membership)
        for j in range(len(own_membership)):
            tau[j] = np.clip(np.sum(own_membership[j])/own_membership[j].shape[0],1e-10, 1-1e-10)
        ## E-step
        own_membership_update = np.ones((len(own_membership), df_arr.shape[1]))
        log_likelihood = 0
        log_prob = np.zeros((len(own_membership), len(membership), df_arr.shape[1]))
        # for i in range(len(cols)):
        #     for k in range(len(membership)):
        #         for j in range(len(own_membership)):
        #             log_prob[j][k] = coal_rate_log_pdf(t_arr = df_arr[i], gamma_arr=gamma_arr[j][k], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])]
        #             log_likelihood += np.sum(log_prob[j][k]*own_membership[j]*membership[k,int(cols[i])])
        #     max_log_prob = np.max(np.max(log_prob, axis = 0),axis=0)
        #     for j in range(len(own_membership)):
        #         sum_probs = np.zeros(df_arr.shape[1])
        #         for k in range(len(membership)):
        #             sum_probs += (np.exp(np.clip(log_prob[j][k] - max_log_prob, -10, 10)))*membership[k, int(cols[i])]
        #         own_membership_update[j] *= sum_probs
        prod_term = np.zeros((len(own_membership), len(membership), len(epoch_intervals) - 1))
        for k in range(len(membership)):
            for j in range(len(own_membership)):
                prod_term[j][k] = get_prod_term(gamma_arr[j][k])
        for i in range(len(cols)):
            for k in range(len(membership)):
                for j in range(len(own_membership)):
                    if i < membership[k][1] and i >= membership[k][0]:
                        log_prob[j][k] = coal_rate_log_pdf(t_arr = df_arr[i], gamma_arr=gamma_arr[j][k], epoch_t_index=epoch_t_index[i], prod_term_arr=prod_term[j][k])/ni[i]
                        log_likelihood += np.sum(log_prob[j][k]*own_membership[j])
            max_log_prob = np.max(np.max(log_prob, axis = 0),axis=0)
            for j in range(len(own_membership)):
                sum_probs = np.zeros(df_arr.shape[1])
                for k in range(len(membership)):
                    if i < membership[k][1] and i >= membership[k][0]:
                        sum_probs += (np.exp(np.clip(log_prob[j][k] - max_log_prob, -10, 10)))
                own_membership_update[j] *= sum_probs
        log_likelihood_arr.append(log_likelihood)
        for j in range(len(own_membership)):
            own_membership_update[j] *= tau[j]
        # own_membership_update = own_membership_update + 1e-300
        print(own_membership_update)
        own_membership_update = np.maximum(own_membership_update, 1e-300)
        own_membership_update = own_membership_update/(np.sum(own_membership_update, axis = 0))
        own_membership = own_membership_update
        # membership_thresh = own_membership > 0.5
        membership_thresh = make_one_hot(np.argmax(own_membership, axis = 0), len(own_membership))
        acc_arr = np.zeros((len(own_membership), len(membership)))
        for i in range(len(own_membership)):
            for j in range(0,3): ###hard-coding: because onle first three rows have ground-truth information
                acc = np.sum(membership_thresh[i] == ground_truth_membership[j])
                # if acc < 0.5:
                #     acc = 1 - acc
                acc_arr[i][j] = acc
        overall_acc = np.sum(np.max(acc_arr, axis=1))/len(membership_thresh)/len(membership_thresh[0])
        print("Sample = " + str(sample_id) + " Accuracy = " + str(overall_acc))
        
        for i in range(gamma_arr.shape[0]):
            plt.clf()
            for j in range(gamma_arr.shape[1]):
                plt.plot(gamma_arr[i][j], marker = 'o')        
            plt.legend(['pop A', 'pop B', 'pop C', 'pop D', 'pop E', 'pop F'], fontsize = 14)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Gamma', fontsize = 14)
            plt.show()
            plt.savefig('gamma_' + str(i) + '_iter_' + str(epoch) + '.png')
            plt.close()  

        if epoch > 10: ##min-iters = 10
            if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.0001:
                break ## stop if log-likelihood isn't changing much
    print("Sample = " + str(sample_id) + " Epochs = " + str(epoch) + " Total time = " + str(time.time() - start_time) + " EM time = " + str(time.time() - start_time_em))


    plt.clf()      
    plt.plot(log_likelihood_arr)
    plt.savefig('log_likelihood.png')
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

    return overall_acc

acc = 0
count = 0

for sample_id in range(0,10):
    acc += main(sample_id, plot=False, gamma_arr =  None)
    count += 1   

print("Average accuracy = " + str(acc/count)) 


