#!pip install pytorch-pretrained-bert

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
#% matplotlib inline
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine




def initialize():

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return (model, tokenizer)


def get_bert_vectors_for(word, text, model, tokenizer):
    """
    Run the token sentence through the model and calculate a word vector
    based on the mean of the WordPiece vectors in the last layer
    """
    tokenized_word = tokenizer.tokenize(word)

    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    segments_ids = [1] * len(tokenized_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    #for tup in zip(tokenized_text, indexed_tokens):
        #print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    try:
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
    except:
        print("tokenized sequence too long")
        print(tokenized_text)
        return None

    # Rearrange hidden layers to be grouped by token
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings.size()

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings.size()


    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    token_embeddings.size()
    
    vectors = []
    
    # get a vector for each layer of the network
    for layer in range(12):
        piece_vectors = []
        for word_piece in tokenized_word:
            # TODO should be the matching slice, because this doesnt account for repeat word  pieces
            index = tokenized_text.index(word_piece)
            token = token_embeddings[index]
            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            #sum_vec = torch.sum(token[-4:], dim=0)

            # Use the vectors from the current layer
            vec = token[layer]
            piece_vectors.append(vec.numpy())

        # use the mean of all of the word_pieces. 
        layer_vector = np.average(piece_vectors, axis=0)    
    
        # add the vector for this layer to our grand list
        vectors.append(layer_vector)
    return vectors


def average_pairwise_token_distance(vectors):
    distances = []
    v1 = vectors.pop()
    while len(vectors) >= 1:
        for v in vectors:
            distance = 1 - cosine(v1, v)
            distances.append(distance)
        v1 = vectors.pop()

    if len(distances) >= 1:
        average =  np.average(distances)
    else:
        average =  0

    return average

"""
COntroversial method for calculating variance
"""
def variance_for_vectors(vectors):
    if len(vectors) == 1:
        return 0
    a = np.var(vectors, axis=0)
    variance = np.average(a)
    return variance


def generalized_variance(vectors):
    if len(vectors) == 1:
        return 0
    """
    our variable are our dimensions, which are columns. 
    NumPy wants the variables as the rows, so first we have to flip the matrix
    we do this with 'rowvar=0' flag, which treats the columns as variables
    """
    # calculate covariance matrix
    #https://stackoverflow.com/questions/15036205/numpy-covariance-matrix
    covar = np.cov(vectors,rowvar=0) # rowvar false, each column is a variable

    # calculate determinant of covariance matrix
    # our values are too small to use the basic det function, which underflows
    #det = np.linalg.det(covar)

    # so we have to use slogdet
    sign, logdet = np.linalg.slogdet(covar)
    det = sign * np.exp(logdet)
    return det


def total_variance(vectors):
    if len(vectors) == 1:
        return 0
    # calculate covariance matrix
    # https://stackoverflow.com/questions/15036205/numpy-covariance-matrix
    covar = np.cov(vectors,rowvar=0) # rowvar false, each column is a variable

    # calculate sum of diagonal elements
    total_variance = covar.trace()
    return total_variance


def calculate_clusters_for(tokens, layer, k, model, tokenizer):
    if (len(tokens)) >= k:

        layer_vectors = [token['vector'][layer] for token in tokens]
        clusters = []

        # calculate clusters
        kmeans_obj = KMeans(n_clusters=k)
        kmeans_obj.fit(layer_vectors)
        label_list = kmeans_obj.labels_
        cluster_centroids = kmeans_obj.cluster_centers_


        # store cluster_id with token
        for index,datapoint in enumerate(tokens):
            datapoint['cluster_id'] = label_list[index]


        # retrieve centroid for each cluster and uids of sentences in cluster:
        for cluster_index in range(k):
            sentence_uids = []
            cluster_vectors = []

            for index, datapoint in enumerate(tokens):
                if datapoint['cluster_id'] == cluster_index:
                    sentence_uids.append(datapoint['uid'])
                    cluster_vectors.append(layer_vectors[index])


            # calculate variance for this cluster
            # the sentence uids are the ones in this cluster. you need to get the vectors for them 
            # and calculate variance.
            cluster_var = variance_for_vectors(cluster_vectors)
            total_var = total_variance(cluster_vectors)
            generalized_var = generalized_variance(cluster_vectors)

            avg_pairwise_token_distance = average_pairwise_token_distance(cluster_vectors)


            single_cluster_data = {'word': tokens[0]['word'],
                        'layer': layer,
                        'k_clusters': k,
                        'cluster_id': cluster_index,
                        'centroid': cluster_centroids[cluster_index],
                        'sentence_uids': sentence_uids,
                        'within_cluster_variance': cluster_var, # this is the wonky calculation
                        'total_variance': total_var,
                        'generalized_variance': generalized_var,
                        'average_pairwise_token_distance': avg_pairwise_token_distance
                        }
            clusters.append(single_cluster_data)      
        return clusters
    else:
        return None