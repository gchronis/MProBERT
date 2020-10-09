from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(d)

from helpers import bert_helper, datasets, grinders

import os, shutil
import csv
import pickle

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

"""
Resources used 

 -  https://stackoverflow.com/questions/38197964/pandas-plot-multiple-time-series-dataframe-into-a-single-plot
 -  https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
"""

def plot_axis(ax, dataset, similarity_measure):
    # get the data
    results_file = '../data/layer_and_cluster_analysis_results_'+dataset+ '_'+'2020-05-03'+'.csv'
    fieldnames = ['dataset', 'similarity_measure', 'layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']

    with open(results_file, mode='r') as disk:
        reader = csv.DictReader(disk, delimiter='\t', fieldnames=fieldnames)

        data = [row for row in reader]
        for row in data:
            row['spearman'] = float(row['spearman'])
            row['layer'] = int(row['layer'])
        df = pd.DataFrame.from_records(data, coerce_float=True)

        df = df[df['similarity_measure'] == similarity_measure]

        #sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)
        
        ax.set_title(dataset)
        #ax.set_xlabel('Number of K-Means Clusters')
        #ax.set_ylabel('Spearman Correlation')
        # multiline plot with group by
        for key, grp in df.groupby(['layer']): 
            ax.plot(grp['k_clusters'], grp['spearman'], label = "Layer {}".format(key), color=cm(1.*key/NUM_COLORS))


"""
Global Parameters
"""
similarity_measures = ['max_sim', 'avg_sim']
# Change default color cycle for all new axes
NUM_COLORS = 12
cm = plt.get_cmap('gist_rainbow')
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 


"""
similarity datasets
"""
#plt.figure(1)
fig1, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
fig1.text(0.5, 0.04, 'Number of K-Means Clusters', ha='center', va='center')
fig1.text(0.03, 0.5, 'Spearman Correlation', ha='center', va='center', rotation='vertical')


datasets = ['ws_353', 'simlex']
similarity_measure = 'max_sim'
axes = [ax1, ax2]
zipp = zip(datasets, axes)

for dataset, ax in zipp:
    plot_axis(ax, dataset, similarity_measure)
plt.legend(bbox_to_anchor=(1.0, 1.3))
#plt.legend(loc='lower right')    
plt.tight_layout()

"""
relatedness datasets
"""
plt.figure(1)

datasets = ['ws353_rel', 'verbsim', 'men']
#datasets = ['ws353_rel', 'verbsim', 'men']
similarity_measure = 'max_sim'
fig2, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
fig2.text(0.5, 0.04, 'Number of K-Means Clusters', ha='center', va='center')
fig2.text(0.03, 0.5, 'Spearman Correlation', ha='center', va='center', rotation='vertical')



axes = [ax1, ax2, ax3]
zipp = zip(datasets, axes)

for dataset, ax in zipp:
    plot_axis(ax, dataset, similarity_measure)
plt.legend(bbox_to_anchor=(1.0, 1.3))
#plt.tight_layout()
#plt.legend(loc='lower right')    


# print it all out
plt.show()
