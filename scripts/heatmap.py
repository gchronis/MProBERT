from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(d)

from helpers import bert_helper, datasets, grinders, helpers

import os, shutil
import csv
import pickle

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
generate a heatmap of similarity/relatedness correlations broken out by layer and cluster.

code must be edited to specify 
- dataset
- distance metric
- date of the desired results file for that dataset

"""


def get_spearman_results_for(dataset):
    results_file = '../data/layer_and_cluster_analysis_results_'+dataset+ '_'+'2020-05-03'+'.csv'
    fieldnames = ['dataset', 'similarity_measure', 'layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']
    with open(results_file, mode='r') as disk:
        reader = csv.DictReader(disk, delimiter='\t', fieldnames=fieldnames)
        
        data = [row for row in reader]

        for row in data:
            row['spearman'] = float(row['spearman'])
            row['layer'] = int(row['layer'])
            row['k_clusters'] = int(row['k_clusters'])
        df = pd.DataFrame.from_records(data, coerce_float=True)

        return df


def get_table_data_for(dataset, sim_measure):
    df = get_spearman_results_for(dataset) 
    df = df[df['similarity_measure'] == sim_measure]

    df = df.pivot(index='k_clusters', columns='layer', values='spearman')
    df = df.round(3)
    #df.style.format("{:.3%}")
    #pd.options.display.float_format = '${:,.2f}'.format

    print(df)
    print("dataset %s" % dataset)
    print("sim measure %s" % sim_measure)
    #return df.to_csv(quoting=csv.QUOTE_ALL)
    return df


similarity_measures = ['max_sim', 'avg_sim']
# datasets = {'ws_353': ws353, 
#             'ws353_rel': ws353_rel,  
#             'simlex': simlex, 
#             'verbsim': verbsim, 
#             'men': men }
datasets = ['ws_353', 'ws353_rel', 'ws353_sim', 'simlex', 'verbsim', 'men', 'simverb_3500']

sns.set()


"""

EDIT to specify desired visualization

"""
res = get_table_data_for('men', 'max_sim')

f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(res, annot=True, fmt='.3f', linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()
#print(res.to_csv())

# for s in similarity_measures:
#     for d in datasets:
#         res = get_table_data_for(d, s)
#         sns.heatmap(res, cmap="YlGnBu")
#         plt.show()
#         print(res.to_csv())
        