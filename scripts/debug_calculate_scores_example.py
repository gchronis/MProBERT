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


simlex = datasets.get_simlex999()

"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1]


"""
4) The similarity measures we care about
"""
similarity_measures = ['max_sim']


results = []
dataset_name = 'simlex999'
data = simlex
sim_measure = 'max_sim'
k = 1


"""
whats going on with these wordst hat arent getting counted
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10,50]
word = 'attend'

import pickle
ortho = pickle.load(open('../data/word_data/orthodontist/analysis_results/clusters.p', 'rb'))
#print(ortho)

for layer_number in [0]:
    print("layer %s" % layer_number)
    for k in cluster_sizes:
        print("\t cluster_size %s" % k)
        var = helpers.read_variance_for_word_at_layer_and_cluster(word, layer_number, k)
        #print("variance is %s" % var)
        print(type(var))
        print(np.average(var))
        print(type(np.average(var)))
####################


#results_file = '../data/layer_and_cluster_analysis_results_'+dataset+'.csv'
#fieldnames = ['dataset', 'similarity_measure', 'layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

# with open(results_file, mode='w') as disk:
#     writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)
# print("evaluating bert clusters against %s" % dataset_name)
# for layer_number in layers:
#     print("\tlayer %s" % layer_number)


#     data_with_observed_similarities = helpers.calculate_observed_similarities(dataset_name, data)
#     data_with_predicted_similarities = helpers.calculate_predicted_similarities(dataset_name, data_with_observed_similarities, sim_measure, layer_number, k)

#     no_data_for = list(filter(lambda row: row['predicted'] is None, data_with_predicted_similarities))
#     df = pd.DataFrame.from_records(no_data_for)
#     print(df)


#     # here's were you want to filter down to data to what you got results for
#     filtered_data = list(filter(lambda row: row['predicted'] is not None, data_with_predicted_similarities))

#     df = pd.DataFrame.from_records(filtered_data)
#     #print(df.to_string())
#     X = df['predicted']
#     y = df['observed']

#     # run pearson expected vs observed
#     pearson_value = pearsonr(X,y)

#     # run spearman expected vs observed
#     spearman_value = spearmanr(X,y)


#     # save results to file
#     output = { 
#               'dataset': dataset_name,
#               'similarity_measure': sim_measure,
#               'layer': layer_number,
#               'k_clusters': k,
#               'pearson': pearson_value[0],
#               'pearson_P': pearson_value[1],
#               'spearman': spearman_value[0],
#               'spearman_P': spearman_value[1],
#               'N': len(df)
#              }
#     print(output)