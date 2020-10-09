from helpers import datasets
import csv
import os

def randomly(seq, pseudo=True):
    import random
    shuffled = list(seq)  
    if pseudo:
        seed = lambda : 0.479032895084095295148903189394529083928435389203890819038471
        random.shuffle(shuffled, seed)
    else:
        print("shuffling indexes")
        random.shuffle(shuffled) 
        print("done shuffling")
    return list(shuffled)

def collect_bnc_tokens_for_words(words, max_num_examples=100, override=False, outfile='bnc_words_with_context_tokens.csv'):
    import os.path
    import csv
    
    filename = outfile
    parent_dir = '../data'
    pathname = os.path.join(parent_dir, filename)  
    
    # do we already have the data collected?
    if os.path.isfile(pathname) and override==False:
        print("data already exist at %s" % pathname)
        return
    
    else:    
        bnc_reader = datasets.get_bnc()
        corpus = bnc_reader.tagged_sents(strip_space=True)
        corpus_length = datasets.bnc_length()
        print("# Sentences in BNC corpus: %s" % corpus_length)

        
        with open(pathname, mode='w') as outfile:
            writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            
            # create a data structure for keeping tabs on how many tokens we have collected
            unigrams = {}
            for word in words:
                    unigrams[word]=max_num_examples                    
            
            # come up with a random order in which to traverse the BNC
            randomized_indexes = randomly([x for x in range(corpus_length)], pseudo=False)
            print(randomized_indexes[:50])
            
            
            """"
            Iterate through the corpus, looking at words one by one, and 
            keep iterating as long as we still have tokens to collect
            """
            i = 0
            
            while (unigrams and randomized_indexes):
                # track progress
                i+=1
                if i % 100000 == 0:
                    print("Processed %s sentences" % i)
                
                # fetch the next random sentence
                corpus_index = randomized_indexes.pop()
                sentence = corpus[corpus_index]
               
            
                # keep track of words we've seen in this sentence, so we don't collect
                # a word twice if it appears twice in the sentence. 
                seen_words = set()
                
                for word_tuple in sentence:
                    word = word_tuple[0].lower()
                    tag = word_tuple[1]

                    token_count = unigrams.get(word) 
                    
                    # collect this sentence as a token of the word
                    if (token_count != None) and (word not in seen_words):

                        string = ' '.join([w[0] for w in sentence])
                        
                        if i % 100000 == 0:
                            print(word)
                            print(tag)
                            print(string)
                            print(corpus_index)
                        
                        writer.writerow([word, string, tag, corpus_index])
                        seen_words.add(word)
                        if unigrams[word]==0:
                            del unigrams[word]
                        else:
                            unigrams[word] -=1

# read in the tokens for this word
def read_tokens_for(word, data_dir='../data/word_data'):
    try:
        pathname = os.path.join(data_dir, word, 'BNC_tokens.csv')
        with open(pathname, mode='r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word", "sentence", "tag", "uid"])  
            data = [row for row in reader]
            return data
    except: None