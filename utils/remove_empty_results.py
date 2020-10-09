### SCRIPT TO GET RID OF ALL THe empty result picky files you saved

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


DATA_DIR = '../data/word_data'


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

# first look and see how many of these files are empty


empty_file_count = 0
empty_file_words = []

for word in unique_words:

    # open up pickle file
    result_path = os.path.join(DATA_DIR, word, 'analysis_results', 'clusters.p')



    #print(word)
    try:
        data = pickle.load(open(result_path, 'rb'))
    except:
        ("no pickle data for %s " % word)
        data = None

    if data == []:
        print(word)
        empty_file_words.append(word)
        empty_file_count += 1
        os.remove(result_path)
