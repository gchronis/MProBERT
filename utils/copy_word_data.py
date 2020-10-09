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


"""
1) the words we want to collect data for
"""
men = datasets.get_men()
verbsim = datasets.get_verbsim()
ws353_rel = datasets.get_ws353_rel()
ws353_sim = datasets.get_ws353_sim()
ws353 = datasets.get_ws353()
simlex = datasets.get_simlex999()
simverb3500 = datasets.get_simverb3500()

# get all the words
all_words = []
for dataset in [men, verbsim, ws353_sim, ws353_rel, ws353, simlex, simverb3500]:
    for row in dataset:
        w1 = row['word1']
        w2 = row['word2']
        all_words.append(w1)
        all_words.append(w2)
        
unique_words = set(all_words)
print("words to grind on: %s" % len(unique_words))


DEST_DATA_DIR = '~/Desktop/word_data'
SOURCE_DATA_DIR = '../data/word_data'


# make directory for that word
os.makedirs(DEST_DATA_DIR)

for word in unique_words:
    print("copying data for %s" % word)

    source_word_dir = os.path.join(SOURCE_DATA_DIR, word)
    #source_results_dir = os.path.join(source_word_dir, 'analysis_results')
    source_token_path = os.path.join(source_word_dir, 'BNC_tokens.csv')

    #dest_word_dir = os.path.join(DEST_DATA_DIR, word)    
    #dest_results_dir = os.path.join(dest_word_dir, 'analysis_results') 
    filename = word + '_'+'BNC_tokens.csv'
    dest_token_path = os.path.join(DEST_DATA_DIR, filename)







    # copy clusters file to that directory
    try:
        shutil.copyfile(source_token_path, dest_token_path)
    except:
        None

    # copy analysis_results to that directory
    #shutil.copytree(source_results_dir, dest_results_dir)
