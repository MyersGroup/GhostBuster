import msprime
import numpy as np
import sys

chr = sys.argv[1]

def migration_example(chr):

    gentime = 28
    admix   = 0.25 #admixture proportion between groups

    #recombination map
    infile = "/well/myers/users/tgh473/workspace/ghost_buster/msprime_maps_filtered/genetic_map_GRCh37_chr" + str(chr) + ".txt"
    recomb_map = msprime.RateMap.read_hapmap(infile)

    #construct a demographic history
    #see https://tskit.dev/msprime/docs/stable/demography.html

    demography = msprime.Demography()

    #present-day pops
    demography.add_population(name="focal", initial_size=10000)

    demography.add_population(name="A", initial_size=3000)
    demography.add_population(name="B", initial_size=3000)
    demography.add_population(name="C", initial_size=3000)
    demography.add_population(name="D", initial_size=3000)

    demography.add_population(name="AB", initial_size=10000)
    demography.add_population(name="CD", initial_size=10000)
    demography.add_population(name="ABCD", initial_size=10000)

    #define events (have to be ordered by time)

    demography.add_admixture(time = 5000/gentime, derived = "focal", ancestral = ["A", "B", "C", "D"], proportions = [admix, admix, admix, admix])

    demography.add_population_split(time=50000/gentime, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=50000/gentime, derived=["C", "D"], ancestral="CD")
    demography.add_population_split(time=100000/gentime, derived=["AB", "CD"], ancestral="ABCD")

    print(demography.debug())

    sam = []
    N = 20
    k = 0
    with open("msprime.poplabels", "w") as fp:
      with open("sample_ages.txt", "w") as fp2:
        fp.write("ID GROUP POP SEX\n")

        sam.append(msprime.SampleSet(N, population = 'focal'))
        for i in range(0, N):
          fp.write("tsk_" + str(k) + " focal focal 1\n")
          fp2.write("0\n")
          k = k+1

        sam.append(msprime.SampleSet(N, population = 'A'))
        for i in range(0,N):
          fp.write("tsk_" + str(k) + " A A 1\n")
          fp2.write("0\n")
          k = k+1

        sam.append(msprime.SampleSet(N, population = 'B'))
        for i in range(0,N):
          fp.write("tsk_" + str(k) + " B B 1\n")
          fp2.write("0\n")
          k = k+1

        sam.append(msprime.SampleSet(N, population = 'C'))
        for i in range(0,N):
          fp.write("tsk_" + str(k) + " C C 1\n")
          fp2.write("0\n")
          k = k+1

        sam.append(msprime.SampleSet(N, population = 'D'))
        for i in range(0,N):
          fp.write("tsk_" + str(k) + " D D 1\n")
          fp2.write("0\n")
          k = k+1
    fp.close()

    ts = msprime.sim_ancestry(
        samples = sam,
        ploidy = 1,
        demography = demography,
        recombination_rate = recomb_map,
        record_migrations = True
        )

    #add mutations
    mutated_ts = msprime.sim_mutations(ts, rate=1.25e-8, model = "binary")

    #write to file
    mutated_ts.dump("stdpopsim_homsap_chr" + str(chr) + ".trees")
    with open("stdpopsim_homsap_chr" + str(chr) + ".vcf", "w") as vcf_file:
        mutated_ts.write_vcf(vcf_file)

migration_example(chr)
