import datasets
import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import pickle

"""
1) the words we want to collect data for
"""
ws353 = datasets.get_ws353()
# simlex999 = datasets.get_simlex999()

# # get a list of all the words in ws353
# first_word = [row['word1'] for row in ws353]
# second_word = [row['word2'] for row in ws353]
# ws353_wordlist = first_word + second_word

# # get a list of all the words in simlex999
# first_word = [row['word1'] for row in simlex999]
# second_word = [row['word2'] for row in simlex999]
# simlex999_wordlist = first_word + second_word


# all_words = ws353_wordlist + simlex999_wordlist
# print("Total words between 353 and simlex: %s" % len(all_words))
# unique_words = set(all_words) 
# print("Unique words between 353 and simlex: %s" % len(unique_words))


"""
2) the layers we want to analzye
"""
layers = [0,1,5,11]

"""
3) The cluster sizes we want to analyze
"""
cluster_sizes = [1,3,5,7]


"""
In the end we want two data structures that looks like this:

layer k_clusters ws353_pearson p ws353_spearman p ws353_n simlex_pearson p  simlex_spearman p  simlex_n
0     1          .77             .73              200     .54              .49                 988
....  ....       ....
0     7          .88             .80              180     .54              .65                 950
1     1          ....
...   ....       ....
11    7          ....



"""

DATA_DIR = './data/word_data'

def read_centroids_for_word_at_layer_and_cluster(word, layer_number, k):
    try:
        cluster_path = os.path.join(DATA_DIR, word, 'analysis_results', 'clusters.p')
        """
         this is a list of dicts with the structure:
                        {'word': tokens[0]['word'],
                        'layer': layer,
                        'k_clusters': k,
                        'cluster_id': cluster_index,
                        'centroid': cluster_centroids[cluster_index],
                        'sentence_uids': sentence_uids,
                        'within_cluster_variance': cluster_variance
                        }
        """
        data = pickle.load(open(cluster_path, 'rb'))
        data = [item for sublist in data for item in sublist]


        columns = ['word', 'layer', 'k_clusters', 'cluster_id', 'centroid', 'sentence_uids', 'within_cluster_variance']
        df = pd.DataFrame.from_records(data, columns=columns)
        #print(df)

        df = df[df['layer'] == layer_number]
        df = df[df['k_clusters'] == k]
        word_centroids = df['centroid']
        if len(word_centroids) > 0:
            return word_centroids
        else:
            return None
    except:
        return None


results_file = './data/bnc_cluster_analysis_ws353_similarity_results.csv'
fieldnames = ['layer', 'k_clusters', 'pearson', 'pearson_P', 'spearman', 'spearman_P', 'N']
with open(results_file, mode='w') as disk:
    writer = csv.DictWriter(disk, delimiter='\t', fieldnames=fieldnames)
    
    
    for layer_number in layers:
        print("layer %s" % layer_number)
        for k in cluster_sizes:
            print("for %s clusters" % k)
            # calc sim for all the word pairs
            data = ws353
            expected_similarities = []
            for row in data:
                word1 = row['word1']
                word2 = row['word2']
                observed_similarity = row['similarity']

                # get centroid data for these words at this layer and this k size
                pairwise_centroids = {}
                for word in [word1, word2]:
                    centroids = read_centroids_for_word_at_layer_and_cluster(word, layer_number, k)
                    pairwise_centroids[word] = centroids

                #print(pairwise_centroids)

                # calculate maxsim
                # calculate predicted similarity from of each pair of cluster centroids of both words
                predicted_similarities = []
                try:
                    for centroid1 in pairwise_centroids[word1]:
                        for centroid2 in pairwise_centroids[word2]:
                            predicted_similarity = 1 - cosine(centroid1, centroid2)
                            predicted_similarities.append(predicted_similarity)
                    # find the max of the pairwise similarities
                    max_sim = max(predicted_similarities)

                    row['predicted_similarity'] = max_sim
                except:
                    print("%s %s" % (word1, word2))

            # create data frame 
            df = pd.DataFrame.from_records(data)
            print(df.to_string())
            X = df['predicted_similarity']
            y = df['similarity']

            # run pearson expected vs observed
            pearson_value = pearsonr(X,y)

            # run spearman expected vs observed
            spearman_value = spearmanr(X,y)


            # save results to file
            output = {'layer': layer_number,
                      'k_clusters': k,
                      'pearson': pearson_value[0],
                      'pearson_P': pearson_value[1],
                      'spearman': spearman_value[0],
                      'spearman_P': spearman_value[1],
                      'N': len(df)
                     }
            print(output)
            #writer.writerow(output)
            