from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(d)

from helpers import datasets
from helpers import grinders
import os
import csv


words_to_collect = []


"""
1) the words we want to collect data for
"""
men = datasets.get_men()
verbsim = datasets.get_verbsim()
ws353 = datasets.get_ws353()
ws353_rel = datasets.get_ws353_rel()
simlex = datasets.get_simlex999()
bless = datasets.get_bless()
brysbaert = datasets.get_brysbaert()
simverb3500 = datasets.get_simverb3500()

# get all the words
all_words = []
for dataset in [simverb3500]:
    for row in dataset:
        w1 = row['word1']
        w2 = row['word2']
        all_words.append(w1)
        all_words.append(w2)

#all_words = all_words + [row['word'] for row in brysbaert]
print(len(all_words))
        
unique_words = set(all_words)

# collect this new word unless its not new and we already have data for it
for word in unique_words:
    pathname = os.path.join("../data/word_data/", word)
    if not os.path.isdir(pathname):
        words_to_collect.append(word)
        

print("Total words between all datasets: %s" % len(all_words))
print("Unique words between all_datasets: %s" % len(unique_words))
print("New words that we don't have tokens collected yet for yada yada: %s" % len(words_to_collect))


grinders.collect_bnc_tokens_for_words(words_to_collect, override=True)
# tokens_file = 'simverb3500_unseen_tokens.csv'
# grinders.collect_bnc_tokens_for_words(words_to_collect, override=True, outfile=tokens_file)


ALLWORDS_DIR = '../data/word_data'

# you already have tokens collected for each word 
# now these tokens ought to be sorted into their own files

# ensure that there is a word_data directory to store in our words
# you have to delete it first with rm -rf if we are reloading
os.mkdir(ALLWORDS_DIR)


# create files for each word we care about
for word in words_to_collect:
    word_dir = os.path.join(ALLWORDS_DIR, word)
    os.mkdir(word_dir)


# read in the big long file
tokens_path = os.path.join('../data', tokens_file)
with open(tokens_path, mode="r") as infile:
    fieldnames = ["word", "sentence", "POS", "id"]
    reader = csv.DictReader(infile, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, fieldnames=fieldnames)
    
    # split the big long file into smaller, sorted files that are easier to process one at a time
    for row in reader:
        
        word = row["word"]
        text = row["sentence"]
        pos = row["POS"]
        uid = "BNC_" + str(int(row["id"]))

        # open file for this word to spit tokens into
        token_file = os.path.join(ALLWORDS_DIR, word, "BNC_tokens.csv")
        with open(token_file, mode="a") as outfile:
            # finally, write all of the info with the vector to disk
            writer = writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow([word, text, pos, uid])