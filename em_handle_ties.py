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

# df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_unidir_easy/input_files/output/relate_ne_chr1_0.coaltimes', sep = ' ')
# df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_unidir/input_files/example_chr1_50.coaltimes', sep = ' ')
epoch_intervals = np.array([-np.inf] + np.linspace(3 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])
epoch_intervals_pow = np.power(10, epoch_intervals)
## only for unidirectional flow
def make_ground_truth(df):
    ground_truth_membership = np.vstack((np.zeros((50, df.shape[0])), np.ones((50, df.shape[0])))) ## contribution of group 1 in group 2
    for i in range(50, 100):
        try:
            ground_truth = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_unidir_easy_v2/local_ancestry/local_ancestry_true_directional_'+str(i)+'.csv', names = ['startpos', 'endpos'])
        except FileNotFoundError:
            continue
        for j in range(len(ground_truth)):
            indices = df[(df['startpos'] >= ground_truth['startpos'].loc[j]) & (df['endpos']  <= ground_truth['endpos'].loc[j])].index
            ground_truth_membership[i,indices] = 0
    return ground_truth_membership

def make_epoch_t_index(df, cols):
    epoch_t_index = np.zeros((len(cols), df.shape[0]))
    for i in range(len(cols)):
        t_arr = df[cols[i]].values
        mask = np.array(np.repeat(epoch_intervals, len(t_arr)).reshape(-1,len(t_arr)) < np.log(t_arr)/np.log(10), dtype='int')
        epoch_t_index[i] = np.sum(mask, axis = 0) - 1 
        # np.argmax(epoch_intervals[epoch_intervals < np.log(t_arr[t])/np.log(10)])
    return np.array(epoch_t_index, dtype='int')

def count_ties(df, cols):
    ni = np.ones((100, df.shape[0]))  ##hard-coding 100 samples here!
    df_arr = df[cols].values
    for j in range(df.shape[0]):
        t_arr = df_arr[j].tolist()
        count_dict = Counter(t_arr)
        for i in range(len(t_arr)):
            ni[int(cols[i]), j] = count_dict[t_arr[i]]
    return ni

def coal_rate_log_pdf(gamma_arr, t_arr, epoch_t_index):
    eps = 1e-20
    # epoch_intervals = np.array([0] + np.linspace(3 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])
    prod_term_arr = []
    for i in range(0, len(epoch_intervals)-1):
        prod_term = 0
        for j in range(1, i+1):
            prod_term -= gamma_arr[j-1]*(epoch_intervals_pow[j] - epoch_intervals_pow[j-1])
        prod_term_arr.append(prod_term)
    log_pdf = np.log(eps + gamma_arr[epoch_t_index]) - gamma_arr[epoch_t_index]*(np.array(t_arr)- epoch_intervals_pow[epoch_t_index])
    log_pdf += np.array(prod_term_arr)[epoch_t_index]
    return log_pdf ## gives product of PDFs

# Estimate the coal-rate using the entire data
def estimate_mle(df, start = 2, end = 102, ignore = None):
    eps = 1e-20
    gamma_arr = np.zeros(len(epoch_intervals) - 1)
    ne_arr = np.zeros(len(epoch_intervals) - 1)
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    denom_2 = np.zeros(len(epoch_intervals) - 1)
    for epoch in range(len(epoch_intervals) - 1):
        for i in df.columns[start:end]:
            if i != ignore:
                ne_arr[epoch] += np.sum((df[i] < np.power(10, epoch_intervals[epoch + 1])) & (df[i] > np.power(10, epoch_intervals[epoch])))
    for epoch in range(len(epoch_intervals) - 1):
        for i in df.columns[start:end]:
            if i!= ignore:
                r1 = (df[i] < np.power(10, epoch_intervals[epoch + 1])) & (df[i] > np.power(10, epoch_intervals[epoch]))
                denom_1[epoch] += np.sum(df[i].loc[r1] - np.power(10, epoch_intervals[epoch]))
    for epoch in range(len(epoch_intervals) -1):
        for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
            denom_2[epoch] += ne_arr[e]*(np.power(10, epoch_intervals[epoch+1]) - np.power(10, epoch_intervals[epoch]))
    gamma_arr = ne_arr/(denom_1 + denom_2 + eps)
    return gamma_arr

def estimate_gamma(df, ni, own_membership, cols, membership):
    # membership = np.array(membership, dtype='bool')
    eps = 1e-20
    gamma_arr = np.zeros(len(epoch_intervals) - 1)
    ne_arr = np.zeros(len(epoch_intervals) - 1)
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    denom_2 = np.zeros(len(epoch_intervals) - 1)
    membership_scaled = membership/ni
    # own_membership = np.array(own_memberhsip, dtype='bool')
    for epoch in range(len(epoch_intervals) - 1):
        for i in range(len(cols)):
            r1 = (df[i] < epoch_intervals_pow[epoch + 1]) & (df[i] > epoch_intervals_pow[epoch])
            ne_arr[epoch] += np.sum(membership_scaled[int(cols[i])][r1] * own_membership[r1])
    for epoch in range(len(epoch_intervals) - 1):
        for i in range(len(cols)):
            r1 = (df[i] < epoch_intervals_pow[epoch + 1]) & (df[i] > epoch_intervals_pow[epoch])
            denom_1[epoch] += np.sum((df[i][r1] -epoch_intervals_pow[epoch])*(membership_scaled[int(cols[i])][r1] * own_membership[r1]))
    for epoch in range(len(epoch_intervals) -1):
        for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
            denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
    gamma_arr = ne_arr/(denom_1 + denom_2 + eps)
    return ne_arr, denom_2 + denom_1 + eps

def fixed_parameters(df, ni, cols, membership):
    membership_scaled = membership/ni
    membership_scaled = membership_scaled[np.array(cols, dtype='int')]
    r1 = np.zeros((len(epoch_intervals) - 1, df.shape[0], df.shape[1]))
    ne_arr = np.zeros((len(epoch_intervals) - 1, df.shape[0], df.shape[1]))
    denom_1 = np.zeros((len(epoch_intervals) - 1, df.shape[0], df.shape[1]))
    for epoch in range(len(epoch_intervals) - 1):
        r1[epoch] = (df < epoch_intervals_pow[epoch + 1]) & (df > epoch_intervals_pow[epoch])
    for epoch in range(len(epoch_intervals) - 1):
        ne_arr[epoch] = (r1[epoch]*membership_scaled)
    for epoch in range(len(epoch_intervals) - 1):
        denom_1[epoch] = ((df - epoch_intervals_pow[epoch])*ne_arr[epoch])
    return ne_arr, denom_1

def estimate_gamma_matrix(own_membership, num, denom):
    # membership = np.array(membership, dtype='bool')
    eps = 1e-20
    ne_arr = np.zeros(len(epoch_intervals) - 1)
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    denom_2 = np.zeros(len(epoch_intervals) - 1)
    for epoch in range(len(epoch_intervals) - 1):
        ne_arr[epoch] = np.sum(num[epoch]@own_membership)
    for epoch in range(len(epoch_intervals) - 1):
        denom_1[epoch] = np.sum(denom[epoch]@own_membership)
    for epoch in range(len(epoch_intervals) -1):
        for e in range(epoch+1, len(epoch_intervals) - 1):  ## Dont compute for the inf. (last value)
            denom_2[epoch] += ne_arr[e]*(epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch])
    return ne_arr, denom_2 + denom_1 + eps

def main(sample_id, plot = False, gamma_arr = None):
    start_time = time.time()
    num_clusters = 2
    tau = np.ones(num_clusters)/num_clusters
    df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_unidir_easy_v2/output/example_chr1_'+str(sample_id) +'.coaltimes', sep = ' ')
    # print(df)
    # df = pd.read_csv('sim_debug.coaltimes', sep = ' ')
    # df = df.sample(20000).reset_index(drop=True)
    cols = df.columns[2:2+sample_id].tolist() + df.columns[3+sample_id:102].tolist() # run EM for 40 steps
    df_arr = df[cols].values.T
    epoch_t_index = make_epoch_t_index(df, cols)
    ni = count_ties(df, cols)
    # membership = pd.read_csv('sim_debug.groundtruth', sep = ' ').values.T
    # own_membership = membership[sample_id] 
    membership = make_ground_truth(df)#np.vstack((np.zeros((50, df.shape[0])), np.ones((50, df.shape[0]))))#
    own_membership = np.random.uniform(0,1,df.shape[0])#make_ground_truth(df)[sample_id] # ## lets start from ground truth, debug!!
    ground_truth_membership = make_ground_truth(df)[sample_id] ##dont change this
    ne_arr_1, denom_1 = fixed_parameters(df_arr, ni, cols, membership)
    ne_arr_2, denom_2 = fixed_parameters(df_arr, ni, cols, 1-membership)
    log_likelihood_arr = []

    for epoch in range(20):  ## max-iters = 100
        # plt.clf()
        # plt.figure(figsize=(40,4))
        # sns.heatmap(own_membership.reshape(1,-1))
        # plt.savefig('own_membership_' + str(epoch) + '.png')
        # plt.close()
        start_time1 = time.time()
        gamma_arr = [[],[]]
        n1, d1 = estimate_gamma_matrix(own_membership, ne_arr_2, denom_2)
        n2, d2 = estimate_gamma_matrix(own_membership, ne_arr_1, denom_1)
        n3, d3 = estimate_gamma_matrix(1-own_membership, ne_arr_2, denom_2)
        n4, d4 = estimate_gamma_matrix(1-own_membership, ne_arr_1, denom_1)

        # n1, d1 = estimate_gamma_matrix(df_arr, ni, own_membership, cols, 1-membership)
        # n2, d2 = estimate_gamma_matrix(df_arr, ni, own_membership, cols, membership)
        # n3, d3 = estimate_gamma_matrix(df_arr, ni, 1-own_membership, cols, 1-membership)
        # n4, d4 = estimate_gamma_matrix(df_arr, ni, 1-own_membership, cols, membership)
        # gamma_arr[1].append((n1+n4)/(d1+d4)) ## source  = group1, target = group 2
        # gamma_arr[1].append((n2+n3)/(d2+d3)) ## source = group2, target = group 2
        # gamma_arr[0].append((n2+n3)/(d2+d3)) ## source = group 1, target = group 1
        # gamma_arr[0].append((n1+n4)/(d1+d4)) ## source  = group 2, target = group 1
        gamma_arr[1].append((n1)/(d1)) ## source  = group1, target = group 2
        gamma_arr[1].append((n2)/(d2)) ## source = group2, target = group 2
        gamma_arr[0].append((n3)/(d3)) ## source = group 1, target = group 1
        gamma_arr[0].append((n4)/(d4)) ## source  = group 2, target = group 1 --
        gamma_arr = np.array(gamma_arr)
        print(time.time() - start_time1)
        # plt.clf()
        # plt.plot(epoch_intervals[0:-1], gamma_arr[0][0])
        # plt.plot(epoch_intervals[0:-1], gamma_arr[0][1])
        # plt.plot(epoch_intervals[0:-1], gamma_arr[1][0])
        # plt.plot(epoch_intervals[0:-1], gamma_arr[1][1])
        # plt.legend(['gamma_00', 'gamma_01', 'gamma_10', 'gamma_11'])
        # plt.savefig('gamma_'+str(epoch)+'.png')
        # plt.close()
        start_time1 = time.time()
        tau = np.ones(num_clusters)/num_clusters
        tau[0] = np.clip(np.sum(1-own_membership)/own_membership.shape[0],1e-10, 1-1e-10)
        tau[1] = np.clip(np.sum(own_membership)/own_membership.shape[0],1e-10, 1-1e-10)
        ## E-step
        own_membership_update = np.ones((2, df.shape[0]))
        log_likelihood = 0
        # for i in range(len(cols)):
        #     log_prob00 = coal_rate_log_pdf(t_arr = df[cols[i]], gamma_arr=gamma_arr[0][0], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 1 <- 1
        #     log_prob01 = coal_rate_log_pdf(t_arr = df[cols[i]], gamma_arr=gamma_arr[0][1], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 1 <- 2
        #     log_prob10 = coal_rate_log_pdf(t_arr = df[cols[i]], gamma_arr=gamma_arr[1][0], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 2 <- 1
        #     log_prob11 = coal_rate_log_pdf(t_arr = df[cols[i]], gamma_arr=gamma_arr[1][1], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 2 <- 2
        #     log_ll = (log_prob00)*(1-own_membership)*(1-membership[int(cols[i])]) + (log_prob01)*(1-own_membership)*membership[int(cols[i])] +\
        #          (log_prob10)*own_membership*(1-membership[int(cols[i])]) + (log_prob11)*own_membership*membership[int(cols[i])]
        #     log_likelihood += np.sum(log_ll)
        # log_likelihood_arr.append(log_likelihood)
        for i in range(len(cols)):
            log_prob00 = coal_rate_log_pdf(t_arr = df_arr[i], gamma_arr=gamma_arr[0][0], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 1 <- 1
            log_prob01 = coal_rate_log_pdf(t_arr = df_arr[i], gamma_arr=gamma_arr[0][1], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 1 <- 2
            log_prob10 = coal_rate_log_pdf(t_arr = df_arr[i], gamma_arr=gamma_arr[1][0], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 2 <- 1
            log_prob11 = coal_rate_log_pdf(t_arr = df_arr[i], gamma_arr=gamma_arr[1][1], epoch_t_index=epoch_t_index[i])/ni[int(cols[i])] ## 2 <- 2
            log_ll = (log_prob00)*(1-own_membership)*(1-membership[int(cols[i])]) + (log_prob01)*(1-own_membership)*membership[int(cols[i])] +\
                 (log_prob10)*own_membership*(1-membership[int(cols[i])]) + (log_prob11)*own_membership*membership[int(cols[i])]
            log_likelihood += np.sum(log_ll)
            own_membership_update[0] *=  ((np.exp(np.clip(log_prob00 - log_prob11, -10, 10)))*(1-membership[int(cols[i])]) + np.exp(np.clip(log_prob01 - log_prob11, -10, 10))*membership[int(cols[i])])
            own_membership_update[1] *= (np.exp(np.clip(log_prob10 - log_prob11, -10, 10))*(1-membership[int(cols[i])]) + (np.exp(np.clip(log_prob11 - log_prob11, -10, 10)))*membership[int(cols[i])])
        log_likelihood_arr.append(log_likelihood)
        own_membership_update[0] *= tau[0]
        own_membership_update[1] *= tau[1]
        own_membership_update = own_membership_update/np.sum(own_membership_update, axis = 0)
        own_membership = own_membership_update[1]

        membership_thresh = own_membership > 0.5
        print(time.time() - start_time1)
        # ground_truth_membership = make_ground_truth(df) #pd.read_csv('sim_debug.groundtruth', sep = ' ').values.T ## the true ground truth for accuracy 
        acc = np.sum(membership_thresh == ground_truth_membership)/len(membership_thresh)
        if acc < 0.5:
            acc = 1 - acc
        print("Sample = " + str(sample_id) + " Accuracy = " + str(acc))
        # if epoch > 20: ##min-iters = 20
        #     if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.0001:
        #         break ## stop if log-likelihood isn't changing much
    print("Sample = " + str(sample_id) + " Epochs = " + str(epoch) + " Total time = " + str(time.time() - start_time))
    plt.clf()
    plt.plot(log_likelihood_arr)
    plt.savefig('log_likelihood.png')
    plt.close()
    ## Heatmap in genetic position
    # membership_with_location = np.zeros(((int(max(df['endpos'])) - int(min(df['startpos'])))))
    # for i in range(len(df)):
    #     membership_with_location[int(df['startpos'].loc[i]):int(df['endpos'].loc[i])] = 1- own_membership[i]
    
    # try:
    #     ground_truth = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_unidir_easy_v2/local_ancestry/local_ancestry_true_directional_'+str(sample_id) + '.csv', names = ['startpos', 'endpos'])
    # except FileNotFoundError:
    #     if sample_id >= 50:
    #         ground_truth = pd.DataFrame([[0,0]], columns = ['startpos', 'endpos'])
    #     else:
    #         ground_truth = pd.DataFrame([[0,max(df['endpos'])]], columns = ['startpos', 'endpos'])
    
    # membership_with_location_gt = np.zeros(((int(max(df['endpos'])) - int(min(df['startpos'])))))
    # for i in range(len(ground_truth)):
    #     membership_with_location_gt[int(ground_truth['startpos'].loc[i]):int(ground_truth['endpos'].loc[i])] = 1

    # membership_with_location = membership_with_location > 0.5
    # acc = np.sum(membership_with_location == membership_with_location_gt)/len(membership_with_location)
    # print("Sample = " + str(sample_id) + " Accuracy = " + str(acc))
    
    # if plot == True:
    #     plt.clf()
    #     plt.figure(figsize=(40,4))
    #     sns.heatmap(membership_with_location[::100].reshape(1,-1))
    #     plt.savefig('output/local_ancestry_propotions_' + str(sample_id) + '.png')
    #     plt.close()
    #     plt.clf()
    #     plt.figure(figsize=(40,4))
    #     sns.heatmap(membership_with_location_gt[::100].reshape(1,-1) > 0.5)
    #     plt.savefig('output/local_ancestry_propotions_gt_' + str(sample_id) +'.png')
    #     plt.close()

    return acc

acc = 0
count = 0

# gamma_num_arr = []
# gamma_denom_arr = []
# print("Estimating coal rate using all pairwise samples...")
# for i in tqdm(range(20)):
#     df = pd.read_csv('/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sims/sim_unidir_easy_v2/output/relate_ne_chr1_'+str(i) +'.coaltimes', sep = ' ')
#     cols = df.columns[2:2+i].tolist() + df.columns[3+i:102].tolist()
#     if i == 0:
#         ground_truth_membership = np.vstack((np.zeros((50, df.shape[0])), np.ones((50, df.shape[0])))) #make_ground_truth(df)
#     num1, denom1 = estimate_gamma_all_pairwise(df, ground_truth_membership[i], cols, ground_truth_membership)
#     num2, denom2 = estimate_gamma_all_pairwise(df, ground_truth_membership[i], cols, 1-ground_truth_membership)
#     if i == 0:
#         gamma_num_arr.extend([num1, num2])
#         gamma_denom_arr.extend([denom1, denom2])
#     else:
#         gamma_num_arr[0] += num1
#         gamma_num_arr[1] += num2
#         gamma_denom_arr[0] += denom1
#         gamma_denom_arr[1] += denom2
# gamma_arr = np.array(gamma_num_arr)/np.array(gamma_denom_arr)

for sample_id in range(50,100):
    acc += main(sample_id, plot=False, gamma_arr =  None)
    count += 1   

print("Average accuracy = " + str(acc/count)) 

# print(membership)
# print("EM converged in " + str(time.time() - start_time) + " seconds")
# print("Proportion of group 1 = " + str(np.sum(membership[0] > 0.5)/np.sum(membership[0] >= 0)))

## Heatmap in genetic position
# df['startpos'] = df['startpos']/100
# df['endpos'] = df['endpos']/100
# plt.clf()
# plt.figure(figsize=(40,4))
# sns.heatmap(membership_with_location.reshape(1,-1))
# plt.savefig('local_ancestry_propotions.png')
# plt.show()

## Heatmap ground-truth
# plt.clf()
# plt.figure(figsize=(40,4))
# sns.heatmap(membership_with_location_gt.reshape(1,-1) > 0.5)
# plt.savefig('local_ancestry_propotions_gt.png')
# plt.show()

