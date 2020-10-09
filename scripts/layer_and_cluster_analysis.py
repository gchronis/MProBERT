from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(d)

from helpers import bert_helper, datasets, grinders, helpers

import os, shutil
import numpy as np
import csv
import pickle
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr


"""
1) the words we want to collect data for
"""
men = datasets.get_men()
verbsim = datasets.get_verbsim()
ws353 = datasets.get_ws353()
ws353_rel = datasets.get_ws353_rel()
ws353_sim = datasets.get_ws353_sim()
simlex = datasets.get_simlex999()
simverb_3500 = datasets.get_simverb3500()

# get all the words
all_words = []
for dataset in [men, verbsim, ws353_rel, ws353_sim, simlex]:
    for row in dataset:
        w1 = row['word1']
        w2 = row['word2']
        all_words.append(w1)
        all_words.append(w2)
        
unique_words = set(all_words)
print("words to grind on: %s" % len(unique_words))

"""
2) the datasets we want to analyze
"""
# 5/2/20 already collected ws353 and ws353_rel
# 5/3/20 recollecting to get words ones we don't have data for bc of the 'not enough tokens to cluster' bug
# 5/29/20 collecting 
# 7/01/20 running analysis for o.g. FInkelstein et al. WS-353 dataset
#datasets = {'ws353': ws353 }

datasets = {'ws353_sim': ws353_sim, 
            'ws353_rel': ws353_rel,  
            'ws353': ws353,  
            'simlex': simlex, 
            'verbsim': verbsim, 
            'men': men,
            'simverb_3500': simverb_3500  }

"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10,50]


"""
4) The similarity measures we care about
"""
similarity_measures = ['max_sim', 'avg_sim']



"""
ACTUAL SCRIPT
"""

# okay do this thing
from datetime import date
today = date.today().isoformat()

results = []
for dataset, data in datasets.items():
    results_file = '../data/layer_and_cluster_analysis_results_'+dataset+'_'+today+'.csv'
    fieldnames = ['dataset', 'similarity_measure', 'layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

    with open(results_file, mode='w') as disk:
        writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)
        print("evaluating bert clusters against %s" % dataset)
        for layer_number in layers:
            print("\tlayer %s" % layer_number)
            for k in cluster_sizes:
                print("\t\tcluster %s" % k)
                for sim_measure in similarity_measures:
                    result = helpers.calculate_score(dataset, data, layer_number, k, sim_measure)
                    print(result)
                    #results.append(result)
                    writer.writerow(result)

