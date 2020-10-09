from helpers import bert_helper, datasets, grinders

import os, shutil
import numpy as np
import csv
import pickle
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr

DATA_DIR = '../data/word_data'



def calculate_score(dataset_name, data, layer_number, k, sim_measure, multicluster=False, multilayer=False):
    # add an 'expected similarity' entry to each entry
    data_with_observed_similarities = calculate_observed_similarities(dataset_name, data)
    data_with_predicted_similarities = calculate_predicted_similarities(dataset_name, data_with_observed_similarities, sim_measure, layer_number, k, multicluster, multilayer)


    # here's were you want to filter down to data to what you got results for
    filtered_data = list(filter(lambda row: row['predicted'] != None, data_with_predicted_similarities))

    df = pd.DataFrame.from_records(filtered_data)
    #print(df)
    #print(df.to_string())

    X = df['predicted']
    y = df['observed']

    # run pearson expected vs observed
    pearson_value = pearsonr(X,y)

    # run spearman expected vs observed
    spearman_value = spearmanr(X,y)


    # save results to file
    output = { 
              'dataset': dataset_name,
              'similarity_measure': sim_measure,
              'layer': layer_number,
              'k_clusters': k,
              'pearson': pearson_value[0],
              'pearson_P': pearson_value[1],
              'spearman': spearman_value[0],
              'spearman_P': spearman_value[1],
              'N': len(df)
             }
    return output

# add an 'observed' dict entry to data strcuture containing observed similarity value
# (this obviates the need to distinguish between similarity and relatedness datasets further down the pipe)
def calculate_observed_similarities(dataset, data):
    for row in data:
        word1 = row['word1']
        word2 = row['word2']
        if dataset in ['ws353_sim', 'ws353', 'simlex', 'simverb_3500']:
            observed_similarity = row['similarity']
        elif dataset in ['men', 'verbsim', 'ws353_rel']:
            observed_similarity = row['relatedness']
        row['observed'] = observed_similarity
    return data


# add a 'predicted' dict entry to data strcuture containing predicted similarity value
def calculate_predicted_similarities(dataset, data, sim_measure, layer_number, k, multicluster=False, multilayer=False):



    # print(dataset)
    # print(sim_measure)
    # print(layer_number)
    # print(k)
    for row in data:
        word1 = row['word1']
        word2 = row['word2']

        # use the union of  every cluster centroud if we are not doing a cluster analusis
        if multicluster == True and multilayer == True:
            w1_centroids = read_centroids_for_word_at_layers_and_clusters(word1, layer_number, k)
            w2_centroids = read_centroids_for_word_at_layers_and_clusters(word2, layer_number, k)
        if multicluster == True:
            w1_centroids = read_centroids_for_word_at_layer_and_clusters(word1, layer_number, k)
            w2_centroids = read_centroids_for_word_at_layer_and_clusters(word2, layer_number, k)

        else:
            w1_centroids = read_centroids_for_word_at_layer_and_cluster(word1, layer_number, k)
            w2_centroids = read_centroids_for_word_at_layer_and_cluster(word2, layer_number, k)

        # print(word1)
        # print(len(w1_centroids))        
        # print(word2)
        # print(len(w2_centroids))
        if (w1_centroids is None) or (w2_centroids is None):
            row['predicted'] = None

        elif sim_measure == 'avg_sim':
            row['predicted'] = avg_sim(w1_centroids, w2_centroids, k)

        elif sim_measure == 'max_sim':
            row['predicted'] = max_sim(w1_centroids, w2_centroids)

    return data
     
def read_clusters(word):   
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
    except:
        return None
    # the 'if item' removes None values for cluster sizes we didnt have enough tokens for
    data = [item for sublist in data if sublist for item in sublist]
    
    columns = ['word', 'layer', 'k_clusters', 'cluster_id', 'centroid', 'sentence_uids', 'within_cluster_variance', 'average_pairwise_token_distance']
    df = pd.DataFrame.from_records(data, columns=columns)
    return df


def read_centroids_for_word_at_layer_and_cluster(word, layer_number, k):
    df = read_clusters(word)
    if df is None:
        print("no tokens collected for %s" % word)
        return None
    df = df[df['layer'] == layer_number]
    df = df[df['k_clusters'] == k]
    word_centroids = df['centroid']
    if len(word_centroids) > 0:
        return word_centroids
    else:
        return None


def read_centroids_for_word_at_layer_and_clusters(word, layer_number, k):
    df = read_clusters(word)
    if df is None:
        print("no tokens collected for %s" % word)
        return None
    df = df[df['layer'] == layer_number]

    # you want to return centrouds for ALL clusters
    # so if K is ten you return the union of all centroids for clusters where K <= 10
    df = df[df['k_clusters'] <= k]
    word_centroids = df['centroid']
    if len(word_centroids) > 0:
        return word_centroids
    else:
        return None


def read_centroids_for_word_at_layers_and_clusters(word, layer_number, k):
    df = read_clusters(word)
    if df is None:
        print("no tokens collected for %s" % word)
        return None
    df = df[df['layer'] <= layer_number]

    # you want to return centrouds for ALL clusters
    # so if K is ten you return the union of all centroids for clusters where K <= 10
    df = df[df['k_clusters'] <= k]

    word_centroids = df['centroid']

    # debug
    # if layer_number == 4 and k == 4:
    #     print("number of clusters being considered: %s" % len(word_centroids))

    if len(word_centroids) > 0:
        return word_centroids
    else:
        return None


"""
# dispersal_measure is one of 
    'within_cluster_variance'
    'intercluster_variance'
    'average_pairwise_token_distance'
    'average_centroid_distance'
"""
def dispersal_at_layer_and_cluster(word, layer_number, k, dispersal_measure=None):
    df = read_clusters(word)
    df = df[df['layer'] == layer_number]
    df = df[df['k_clusters'] == k]

    if len(df) == 0:
        dispersal = None

    elif dispersal_measure == 'within_cluster_variance':
        word_centroids = df['within_cluster_variance']
        dispersal =  np.average(word_centroids)

    elif dispersal_measure == 'intercluster_variance':
        # variance between cluster centrouds
        centroids = df['centroid']
        centroids = pd.Series.to_numpy(centroids)
        intercluster_variance = bert_helper.variance_for_vectors(centroids)
        dispersal = intercluster_variance

    elif dispersal_measure == 'average_pairwise_token_distance':
        cluster_averages = df['average_pairwise_token_distance']
        dispersal =  np.average(cluster_averages) 


    elif dispersal_measure == 'average_centroid_distance':
        centroids = df['centroid']
        centroids = pd.Series.to_numpy(centroids).tolist()
        intercluster_average_distance = bert_helper.average_pairwise_token_distance(centroids)
        dispersal = intercluster_average_distance

    return dispersal



def read_intercluster_variance_at_layer_and_cluster(word, layer_number, k):
    centroids = read_centroids_for_word_at_layer_and_cluster(word, layer_number, k)
    if centroids is None:
        return None
    else:
        centroids = pd.Series.to_numpy(centroids)

        # Calculate the population variance for each component of the vectors. 
        a = np.var(centroids, axis=0)
        variance = np.average(a)
        return variance


def max_sim(w1_centroids, w2_centroids):
    predicted_similarities = []
    for centroid1 in w1_centroids:
        for centroid2 in w2_centroids:
            predicted_similarity = 1 - cosine(centroid1, centroid2)
            # TODO am I nuts? Why do I get a positive correlation for cosine distance over cosine similarity???
            #predicted_similarity = cosine(centroid1, centroid2)
            predicted_similarities.append(predicted_similarity)
    # find the max of the pairwise similarities
    return max(predicted_similarities)       

def avg_sim(w1_centroids, w2_centroids, k):
    predicted_similarities = []
    for centroid1 in w1_centroids:
        for centroid2 in w2_centroids:
            predicted_similarity = 1 - cosine(centroid1, centroid2)
            # TODO am I nuts? Why do I get a positive correlation for cosine distance over cosine similarity???
            #predicted_similarity = cosine(centroid1, centroid2)
            predicted_similarities.append(predicted_similarity)
    # find the avg of the pairwise similarities
    avg_sim = np.sum(predicted_similarities) / k / k
    return avg_sim 