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
Resources used 

 -  https://stackoverflow.com/questions/38197964/pandas-plot-multiple-time-series-dataframe-into-a-single-plot
 -  https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
"""

def read_results():
    # get the data
    results_file = '../data/concreteness_simlex_analysis_results_simlex_2020-05-03.csv'
    fieldnames = ['layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

    with open(results_file, mode='r') as disk:
        reader = csv.DictReader(disk, delimiter='\t', fieldnames=fieldnames)

        data = [row for row in reader]
        for row in data:
            row['spearman'] = float(row['spearman'])
            row['layer'] = int(row['layer'])
        df = pd.DataFrame.from_records(data, coerce_float=True)
        return df

def plot_axis(ax, dataset, df):


        #sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)
        
        #ax.set_xlabel('Number of K-Means Clusters')
        #ax.set_ylabel('Spearman Correlation')
        # multiline plot with group by
        for key, grp in df.groupby(['layer']): 
            ax.plot(grp['k_clusters'], grp['spearman'], label = "Layer {}".format(key), color=cm(1.*key/NUM_COLORS))
            ax.set_title(dataset)
"""
Global Parameters
"""
# Change default color cycle for all new axes
NUM_COLORS = 12
cm = plt.get_cmap('gist_rainbow')
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 


"""
similarity datasets
"""


#plt.figure(1)
# fig1, (ax1) = plt.subplots(nrows = 1, ncols = 1)
# fig1.text(0.5, 0.04, 'Number of K-Means Clusters', ha='center', va='center')
# fig1.text(0.06, 0.5, 'Spearman Correlation', ha='center', va='center', rotation='vertical')


# datasets = ['simlex']
# #similarity_measure = 'max_sim'
# axes = [ax1]
# zipp = zip(datasets, axes)

# for dataset, ax in zipp:
#     data = read_results()
#     ax.set_title('Relationship between average within-cluster variance\n and USF concreteness norms of words in SimLex999')
#     plot_axis(ax, dataset, data)
# plt.legend(bbox_to_anchor=(1.3, 1.3))
# #plt.legend(loc='lower right')    
# plt.tight_layout()




"""
break out by POS
"""


def read_pos_results():
    results_file = '../data/concreteness_simlex_analysis_results_simlex_pos_2020-05-04.csv'
    fieldnames = ['layer', 'k_clusters', 'POS', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

    with open(results_file, mode='r') as disk:
        reader = csv.DictReader(disk, delimiter='\t', fieldnames=fieldnames)

        data = [row for row in reader]
        for row in data:
            row['spearman'] = float(row['spearman'])
            row['spearman_P'] = float(row['spearman_P'])
            row['layer'] = int(row['layer'])
        df = pd.DataFrame.from_records(data, coerce_float=True)
        return df

#plt.figure(1)


simlex = datasets.get_simlex999()

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


poses = ['N', 'A', 'V']
dataset = 'simlex'
similarity_measure = 'max_sim'

fig1, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
fig1.text(0.5, 0.04, 'USF concreteness rating', ha='center', va='center')
fig1.text(0.06, 0.5, 'BERT-cloud Variance', ha='center', va='center', rotation='vertical')


axes = [ax1, ax2, ax3]
zipp = zip(poses, axes)




for pos, ax in zipp:


    for row in words:
        word = row['word']


        var_for_all_cluster_ids = helpers.read_variance_for_word_at_layer_and_cluster(word, 8, 1)

        if var_for_all_cluster_ids is not None:
            var = np.average(var_for_all_cluster_ids)
            row['within_cluster_variance'] = var

    filtered_data = list(filter(lambda row: row['within_cluster_variance'] != None, words))

    data = pd.DataFrame.from_records(filtered_data)

    #data = read_pos_results()
    data = data[data['POS'] == pos]
    #data = data[data['layer'] == 0]
    #print(data)


    X = data['concreteness']
    y = data['within_cluster_variance']


    names = df['word']
    c = np.random.randint(len(df))

    print(data)
    ax.scatter(X,y, c=c)
    ax.set_title(pos)
    #data = data[data['spearman_P'] < 0.10]
    print(len(data))
    #plot_axis(ax, pos, data)

plt.legend(bbox_to_anchor=(1.0, 1.8))
#plt.legend(loc='lower right')    


# # print it all out
plt.show()