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


"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10,50]


dispersal_measures = [ 'average_centroid_distance', 'average_pairwise_token_distance', 'intercluster_variance', 'within_cluster_variance']


"""
Get simlex words with abstractness ratings
"""
words = []

for row in simlex:
    w1 = row['word1']
    w1_conc = row['conc_w1']
    words.append({'word': w1, 'concreteness': w1_conc})
    w2 = row['word2']
    w2_conc = row['conc_w2']
    words.append({'word': w2, 'concreteness': w2_conc})

words = [dict(t) for t in {tuple(d.items()) for d in words}]

print("unique simlex words: %s" % len(words))

# okay do this thing
from datetime import date
today = date.today().isoformat()

results_file = '../data/concreteness_simlex_analysis_results_'+dataset+'_'+today+'.csv'
fieldnames = ['dispersal_measure', 'layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

with open(results_file, mode='w') as disk:
    writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)
    print("evaluating bert clusters against %s" % dataset)

    for m in dispersal_measures:
        for l in layers:
            print("\tlayer %s" % l)
            for k in cluster_sizes:
                print("\t\tcluster %s" % k)
                for row in words:
                    word = row['word']

                    dispersal = helpers.dispersal_at_layer_and_cluster(word, l, k, m)
                    row['dispersal'] = dispersal

                filtered_data = list(filter(lambda row: row['dispersal'] != None, words))

                df = pd.DataFrame.from_records(filtered_data)

                #print(df.to_string())
                X = df['concreteness']
                y = df['dispersal']
                pearson_value = pearsonr(X,y)
                spearman_value = spearmanr(X,y)

                result = {
                    'dispersal_measure': m,
                    'layer': l,
                    'k_clusters': k,
                    'pearson': pearson_value[0],
                    'pearson_P': pearson_value[1],
                    'spearman': spearman_value[0],
                    'spearman_P': spearman_value[1],
                    'N': len(df)
                }
                print(result)
                writer.writerow(result)
