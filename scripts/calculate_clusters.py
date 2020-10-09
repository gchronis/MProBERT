from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(d)

from helpers import bert_helper, datasets, grinders

import os, shutil
import numpy as np
import csv
import pickle

import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr
#from scipy.stats import spearmanr
#from scipy.spatial.distance import cosine


DATA_DIR = '../data/word_data'


"""
for each word file we have, do the following:
    for each layer we care about, calculate the token embedding at that layer for each token
        for each number of clusters we care about, calculate the centroids of those clusters
        
store results in a file, one for each word+layer+cluster_number combo, resulting in a file structure like the following:

word_data/
  |-airplane/
  | |- bnc_tokens.csv
  | |- clusters.punkt
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
ws353 = datasets.get_ws353()
ws353_rel = datasets.get_ws353_rel()
ws353_sim = datasets.get_ws353_sim()
simlex = datasets.get_simlex999()
simverb3500 = datasets.get_simverb3500()

# get all the words
all_words = []
#for dataset in [men, verbsim, ws353_rel, ws353, simlex, simverb3500]:
for dataset in [simlex]:
    for row in dataset:
        w1 = row['word1']
        w2 = row['word2']
        all_words.append(w1)
        all_words.append(w2)
        
unique_words = set(all_words)
print("words to grind on: %s" % len(unique_words))


####
# DEBUG 
# unique_words = ['believe']

"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10,50]



# initialize BERT model
(model, tokenizer) = bert_helper.initialize()

# keep a count of how many words we
i = 0
for word in unique_words:

    word_results = []

    i+=1
    if i % 50 == 0:
        print("processed %s words" % i)
        print("calculating clusters for %s" % word)

    # if you have the pickle file already, continue
    outpath = os.path.join(DATA_DIR, word, 'analysis_results', 'clusters.p')
    if os.path.isfile(outpath):
       continue


    # create a directory to store all our clustering results in
    results_dir = os.path.join(DATA_DIR, word, 'analysis_results')    
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # read in tokens for this word
    tokens = grinders.read_tokens_for(word)


    outpath = os.path.join(results_dir, 'clusters.p')


    if tokens:
        # get the model activation for each token
        for token in tokens:
            token['vector'] = bert_helper.get_bert_vectors_for(word, token['sentence'], model, tokenizer)   
        # get rid of the data points we weren't able to vectorize (due to length)
        tokens = list(filter(lambda row: row['vector'] != None, tokens))

        # gwt the clusters!
        for layer in layers:
            #print("layer %s" % layer)

            for k in cluster_sizes:
                #print("clusters %s" % k)

                sub_results = bert_helper.calculate_clusters_for(tokens, layer, k, model, tokenizer)
                #for row in results:
                    #print("%s\t%s\t%s\t%s" % (row['layer'], row['k_clusters'], row['cluster_id'], row['within_cluster_variance']))
                word_results.append(sub_results)

        pickle.dump(word_results, open(outpath, 'wb'))
    else:
        print("no tokens collected for %s" % word)
