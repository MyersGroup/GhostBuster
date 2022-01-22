import numpy as np
import sys
from tqdm import tqdm
import msprime

path = sys.argv[1]
focal = int(sys.argv[2])
migr_time = float(sys.argv[3])
chrs = sys.argv[4]

chrs = chrs.split(",")
print(chrs)

def get_migrating_tracts_with_id(ts):
    N = 0
    for s in ts.first().samples():
        N += 1
    migrations = []
    for (m, migration) in tqdm(enumerate(ts.migrations())):
      if migration.time < migr_time:
        migrations.append({'left':migration.left, 'right':migration.right, 'node':migration.node, 'time':migration.time, 'dest':migration.dest})
    sorted_migrations = sorted(migrations, key=lambda d: d['left'])
    print(len(sorted_migrations))
    migration_array = [[[]] for _ in range(N)]
    tree = ts.first()
    for (m,migration) in tqdm(enumerate(sorted_migrations)):
        while tree.interval[0] < migration['left']:
               tree.next() 
        while migration['right'] >= tree.interval[0]:
            parent_node = migration['node']
            for i in tree.get_leaves(parent_node):
                migration_array[i].append([tree.interval[0], tree.interval[1], migration['dest']])
            if tree.next() == False:
               break;
    return migration_array

for chr in chrs:
    ts = msprime.load(path + 'stdpopsim_homsap_chr'+str(chr)+'.trees')
    migrating_tracts = get_migrating_tracts_with_id(ts)
    for j in [focal]: ## only first 10 are admixed
        migrating_tracts_i = migrating_tracts[j][1:]
        if migrating_tracts_i != []:
            mig = []
            startpos = migrating_tracts_i[0][0]
            for i in range(0,len(migrating_tracts_i)-1):
                if migrating_tracts_i[i][1] != migrating_tracts_i[i+1][0] or migrating_tracts_i[i][2] != migrating_tracts_i[i+1][2]:
                    mig.append([startpos, migrating_tracts_i[i][1], migrating_tracts_i[i][2]])
                    startpos = migrating_tracts_i[i][1]
            if len(migrating_tracts_i) > 0:
                mig.append([startpos, migrating_tracts_i[len(migrating_tracts_i)-1][1], migrating_tracts_i[len(migrating_tracts_i)-1][2]])
            # print(mig)
            # migrating_tracts_i = np.array(migrating_tracts_i[1:])
            # print(migrating_tracts_i)
            # migrating_tracts_i = migrating_tracts_i[(migrating_tracts_i[:,0] != 0) | (migrating_tracts_i[:,1] != 0)]
            if mig != []:
                np.savetxt(path + 'local_ancestry_chr'+str(chr)+'_'+str(j) +'.csv', np.array(mig), delimiter=',')

## python get_local_anc.py /well/myers/users/tgh473/workspace/ghost_buster/sims/stdpopsim_ancient_small/data_50/ 51 2273 1,2,3,4,5 
