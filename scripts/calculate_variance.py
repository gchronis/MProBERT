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
INPUTS: We begin with input data of the tokens for each word, structured as follows:


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


OUTPUTS: for each word file we have, do the following:
    for each layer we care about, calculate the token embedding at that layer for each token
        for each number of clusters we care about, calculate the centroids of those clusters
            calculate intracluster variance for each cluster
                take the average
            calculate variance between centroid clusters

    we use two dispersion measure:
    - total variance: the sum of the diagonal elements of the covariance matrix
    - generalized variance: the determinant of the covariance matrix

    we calculate these measures over several populations
    - for K = n, calculate the dispersion for each cluster in K, then take the average
    - for K <= n, calculate the union of cluster centroids, then calculate the dispersion for each cluster, then take the average
    - for K = n, calculate the dispersion between cluster centroids
    - for K <= n, take the union of cluster centroids U(n in K) and caclulate the dispersion

    As we do each of these for each dispersion measure, we calculate 8 values per word at each layer, leading to 8 correlation analyses down the line.

        
    store results in a file, one for each word+layer+cluster_number combo, resulting in a file structure like the following:

    word_data/
      |-airplane/
      | |- bnc_tokens.csv
      | |- clusters.punkt
      | |- dispersion.csv
      | 
      | 
      | 
  
each cluster file is a csv with the following fields:
    word: String
    layer: x in 0...11
    cluster_size_k: x in 1..10
    unioned? : True or False
    dispersion_measure: 
        - average_pairwise_token_distance
        - average_centroid_distance
        - average_token_total_variance
        - centroid_total_variance
        - average_generalized_variance
        - centroid_generalized_variance
    dispersion: value

"""



"""
1) the words we want to collect data for
"""
# men = datasets.get_men()
# verbsim = datasets.get_verbsim()
# ws353 = datasets.get_ws353()
# ws353_rel = datasets.get_ws353_rel()
# ws353_sim = datasets.get_ws353_sim()
simlex = datasets.get_simlex999()
# simverb3500 = datasets.get_simverb3500()

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
# no dat for funny???
#unique_words = ['aluminum']

"""
2) the layers we want to analzye
"""
layers = [x for x in range(12)]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,2,3,4,5,6,7,8,9,10]



dispersion_measures = [ 
        'average_pairwise_token_distance',
        'average_centroid_distance',
        'average_token_total_variance',
        'centroid_total_variance',
        'average_generalized_variance',
        'centroid_generalized_variance'
        ]

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
for word in unique_words:
    print(word)

    i+=1
    if i % 20 == 0:
        print("processed %s words" % i)
        print("calculating dspersion for %s" % word)

    #if you have the pickle file already, continue
    outpath = os.path.join(DATA_DIR, word, 'analysis_results', 'dispersion.csv')
    if os.path.isfile(outpath):
        print('already collected for %s' % word)
        continue

    word_results = []


    # create a directory to store all our clustering results in
    results_dir = os.path.join(DATA_DIR, word, 'analysis_results')    

    # read in tokens for this word
    tokens = grinders.read_tokens_for(word)
    #print(tokens)


    outpath = os.path.join(results_dir, 'dispersion.csv')
    with open(outpath, mode='w') as disk:
        fieldnames = ['word', 'layer', 'k_clusters', 'cluster_id', 'unioned?', 'dispersion_measure', 'dispersion']
        writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)

        if tokens:
            # get the model activation for each token
            for token in tokens:
                token['vector'] = bert_helper.get_bert_vectors_for(word, token['sentence'], model, tokenizer)   
            # get rid of the data points we weren't able to vectorize (due to length)
            tokens = list(filter(lambda row: row['vector'] != None, tokens))
            num_tokens = len(tokens)

            # gwt the clusters!
            for layer in layers:
                #print("layer %s" % layer)

                # max K=10
                ks = min(num_tokens,10)
                effective_cluster_sizes = range(1,ks+1)

                all_clusters = [bert_helper.calculate_clusters_for(tokens, layer, i, model, tokenizer) for i in effective_cluster_sizes]

                # print("length of cluster union %s" % len(all_clusters))
                # for cluster_size in all_clusters:
                #     print("number of clusters: %s" % len(cluster_size))

                for k in effective_cluster_sizes:

                    up_to_k = all_clusters[:k]
                    #print(len(up_to_k))

                    for unioned in [True, False]:
                        for dispersion_measure in dispersion_measures:

                           # Cosine distance
                            if dispersion_measure == 'average_pairwise_token_distance' and unioned == True:
                                # average token distance for unioned prototypes up to k
                                # get the avg distance for each cluster for each k up to K
                                distances = [ j['average_pairwise_token_distance'] for clusters in up_to_k for j in clusters ]
                                dispersion = np.average(distances)
                            elif dispersion_measure == 'average_pairwise_token_distance' and unioned == False:
                                # get the avg distance for each cluster for only k
                                # print(k)
                                # print(len(up_to_k))
                                # print(len(up_to_k[k-1]))
                                # print("K is %s" % k)
                                # for clusters in up_to_k:
                                #     print("cluster length: %s" % len(clusters))
                                #     for cluster in clusters:
                                #         print(len(clusters))
                                #distances = [ print(len(clusters)) for clusters in up_to_k ]
                                #distances = [ print(cluster) for cluster in up_to_k[k-1]]

                                distances = [ cluster['average_pairwise_token_distance']  for cluster in up_to_k[k-1]]
                                dispersion = np.average(distances)

                            elif dispersion_measure == 'average_centroid_distance' and unioned == True:
                                centroids = [j['centroid'] for clusters in up_to_k for j in clusters ]
                                dispersion = np.average(centroids)
                            elif dispersion_measure == 'average_centroid_distance' and unioned == False:
                                centroids = [ cluster['centroid'] for cluster in up_to_k[k-1] ]
                                dispersion = np.average(centroids)


                            # Total Variance
                            elif dispersion_measure == 'average_token_total_variance' and unioned == True:
                                variances = [ j['total_variance'] for clusters in up_to_k for j in clusters ]
                                dispersion = np.average(variances)
                            elif dispersion_measure == 'average_token_total_variance' and unioned == False:
                                variances = [ cluster['total_variance'] for cluster in up_to_k[k-1] ]
                                dispersion = np.average(variances)

                            elif dispersion_measure == 'centroid_total_variance' and unioned == True:
                                centroids = [j['centroid']for clusters in up_to_k  for j in clusters ]
                                dispersion = bert_helper.total_variance(centroids)
                            elif dispersion_measure == 'centroid_total_variance' and unioned == False:
                                centroids = [cluster['centroid'] for cluster in up_to_k[k-1]]
                                dispersion = bert_helper.total_variance(centroids)

                            # Generalized Variance
                            elif dispersion_measure == 'average_generalized_variance' and unioned == True:
                                variances = [j['generalized_variance'] for clusters in up_to_k  for j in clusters]
                                dispersion = np.average(variances)
                            elif dispersion_measure == 'average_generalized_variance' and unioned == False:
                                variances = [cluster['generalized_variance'] for cluster in up_to_k[k-1]]
                                dispersion = np.average(variances)

                            elif dispersion_measure == 'centroid_generalized_variance' and unioned == True:
                                centroids = [j['centroid']  for clusters in up_to_k for j in clusters]
                                dispersion = bert_helper.generalized_variance(centroids)
                            elif dispersion_measure == 'centroid_total_variance' and unioned == False:
                                centroids = [cluster['centroid'] for cluster in up_to_k[k-1]]
                                dispersion = bert_helper.generalized_variance(centroids)
                            else:
                                dispersion = None


                            result = {
                                'word': word,
                                'layer': layer,
                                'k_clusters': k,
                                'unioned?': unioned,
                                'dispersion_measure': dispersion_measure,
                                'dispersion': dispersion
                            }
                
                            #word_results.append(result)
                            writer.writerow(result)
                            #print(result)

        else:
            print("no tokens collected for %s" % word)
