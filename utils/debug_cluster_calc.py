from helpers import bert_helper, datasets, grinders

import os, shutil
import numpy as np
import csv
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine


DATA_DIR = './data/word_data'


"""
for each word file we have, do the following:
    for each layer we care about, calculate the token embedding at that layer for each token
        for each number of clusters we care about, calculate the centroids of those clusters
        
store results in a file, one for each word+layer+cluster_number combo, resulting in a file structure like the following:

word_data/
  |-airplane/
  | |- bnc_tokens.csv
  | |- clusters.p
  |
  | 
  | 
  | 
  
each cluster file is a csv with the following fields:
    word
    layer
    cluster_size_k
    cluster_number
    centroid
    token_ids
    within_cluster_variance

"""



"""
1) the words we want to collect data for
"""
men = datasets.get_men()
verbsim = datasets.get_verbsim()
ws353_rel = datasets.get_ws353_rel()
ws353 = datasets.get_ws353()
simlex = datasets.get_simlex999()

# get all the words
all_words = []
for dataset in [men, verbsim, ws353_rel, ws353, simlex]:
    for row in dataset:
        w1 = row['word1']
        w2 = row['word2']
        all_words.append(w1)
        all_words.append(w2)
        
unique_words = set(all_words)
print("words to grind on: %s" % len(unique_words))

"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10,50]


"""
Now, 
"""
# don't have afile for
# unique_words = ['ipod']
# test 
#unique_words = ['analyze', 'airplane']


# initialize BERT model
(model, tokenizer) = bert_helper.initialize()

# keep a count of how many words we
i = 0

word = 'sunlight'

word_results = []

i+=1
print("processed %s words" % i)
print("calculating clusters for %s" % word)



# create a directory to store all our clustering results in
results_dir = os.path.join(DATA_DIR, word, 'analysis_results')    
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

# read in tokens for this word
tokens = grinders.read_tokens_for(word)
#print(tokens)
print(len(tokens))


outpath = os.path.join(results_dir, 'clusters.p')
# with open(outpath, mode='w') as disk:
#     fieldnames = ['word', 'layer', 'k_clusters', 'cluster_id', 'centroid', 'sentence_uids', 'within_cluster_variance']
#     writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)

if tokens:
    # get the model activation for each token
    for token in tokens:
        token['vector'] = bert_helper.get_bert_vectors_for(word, token['sentence'], model, tokenizer)   
    # get rid of the data points we weren't able to vectorize (due to length)
    tokens = list(filter(lambda row: row['vector'] != None, tokens))

    # gwt the clusters!
    for layer in layers:
        #rint("layer %s" % layer)

        for k in cluster_sizes:
            print("clusters %s" % k)

            sub_results = bert_helper.calculate_clusters_for(tokens, layer, k, model, tokenizer)
            for row in sub_results:
                print("%s\t%s\t%s\t%s" % (row['layer'], row['k_clusters'], row['cluster_id'], row['within_cluster_variance']))
            word_results.append(sub_results)
            # for row in sub_results:
            #     if row != None:
            #         writer.writerow(row)

    df = pd.DataFrame.from_records(word_results)
    print(df)
    pickle.dump(word_results, open(outpath, 'wb'))
else:
    print("no tokens collected for %s" % word)
