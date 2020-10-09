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
simlex = datasets.get_simlex999()

# get all the words
all_words = []
for dataset in [simlex]:
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
dataset = 'simlex'



# datasets = {'ws_353': ws353, 
#             'ws353_rel': ws353_rel,  
#             'simlex': simlex, 
#             'verbsim': verbsim, 
#             'men': men }

"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10,50]

poses = ['A', 'N', 'V']

"""
Get simlex words with abstractness ratings
"""
words = []

for row in simlex:
    pos = row['POS']

    w1 = row['word1']
    w1_conc = row['conc_w1']
    words.append({'word': w1, 'concreteness': w1_conc, 'POS': pos})
    w2 = row['word2']
    w2_conc = row['conc_w2']
    words.append({'word': w2, 'concreteness': w2_conc, 'POS': pos})

words = [dict(t) for t in {tuple(d.items()) for d in words}]

print("unique simlex words: %s" % len(words))

# okay do this thing
from datetime import date
today = date.today().isoformat()

results_file = '../data/concreteness_simlex_analysis_results_'+dataset+'_pos_intercluster_variance_'+today+'.csv'
fieldnames = ['layer', 'k_clusters', 'POS', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

with open(results_file, mode='w') as disk:
    writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)
    print("evaluating bert clusters against %s" % dataset)

    for l in layers:
        print("\tlayer %s" % l)
        for k in cluster_sizes:
            print("\t\tcluster %s" % k)
            for pos in poses:
                for row in words:
                    word = row['word']


                    # Retrieve variance between cluster centroids for this layer and K
                    variance = helpers.read_intercluster_variance_at_layer_and_cluster(word, l, k)
                    row['inter_cluster_variance'] = variance

                # remove words for which don't have enough tokens to create K unique clusters
                #print(words)
                filtered_data = list(filter(lambda row: row['inter_cluster_variance'] != None, words))

                df = pd.DataFrame.from_records(filtered_data)
                df = df[df['POS'] == pos]

                #print(df.to_string())
                X = df['concreteness']
                y = df['inter_cluster_variance']
                pearson_value = pearsonr(X,y)
                spearman_value = spearmanr(X,y)

                result = {
                    'layer': l,
                    'k_clusters': k,
                    'POS': pos,
                    'pearson': pearson_value[0],
                    'pearson_P': pearson_value[1],
                    'spearman': spearman_value[0],
                    'spearman_P': spearman_value[1],
                    'N': len(df)
                }
                print(result)
                writer.writerow(result)
