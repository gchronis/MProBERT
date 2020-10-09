from nltk.corpus.reader import bnc
import csv

WORDSIM_353_PATH = '../data/wordsim353/combined.csv'
WORDSIM_353_SIM_PATH = '../data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'
WORDSIM_353_REL_PATH = '../data/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt'
SIMLEX_999_PATH = '../data/SimLex-999/SimLex-999.txt'
MEN_PATH = '../data/MEN/EN-MEN-LEM.txt'
VERBSIM_PATH = '../data/verbsim/200601-GWC-130verbpairs.txt'
BLESS_PATH = '../data/BLESS/BLESS.txt'
BRYSBAERT_PATH = '../data/brysbaert/Concreteness_ratings_Brysbaert_et_al_BRM.txt'
SIMVERB_PATH = '../data/simverb3500/SimVerb-3500.txt'

def get_bnc():
    bnc_reader = bnc.BNCCorpusReader(root='../data/BNC/Texts/', fileids=r'[A-K]/\w*/\w*\.xml')
    return bnc_reader

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

def bnc_length(pathname='../data/bnc_length.txt'):
    try:
        with open(pathname, 'r') as fh:
            count = int(fh.read())
            return count
    except:
        print("BNC not yet indexed. Calculating length and writing to 'data/count_of_bnc_sentences.txt'")
        bnc_reader = get_bnc()
        corpus = bnc_reader.tagged_sents(strip_space=True)
        length = len(corpus)
        with open(pathname, 'w') as disk:
            disk.write(str(length))
        return length

def bnc_sentence_to_string(sentence):
    words = [word.lower() for (word, pos) in sentence]
    return " ".join(words)


def get_ws353(path=WORDSIM_353_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',', fieldnames=["word1", "word2", "similarity"])
        line_count = 0
        headers = next(csv_reader)
        for row in csv_reader:
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['similarity'] = float(row['similarity'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from WordSim similarity dataset" % line_count)
    return data



def get_ws353_sim(path=WORDSIM_353_SIM_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word1", "word2", "similarity"])
        line_count = 0
        for row in csv_reader:
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['similarity'] = float(row['similarity'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from WordSim similarity dataset" % line_count)
    return data


def get_simlex999(path=SIMLEX_999_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word1", "word2", "POS", "SimLex999", "conc_w1", "conc_w2", "concQ", "assoc_USF", "sim_assoc333", "SD_simlex"])
        line_count = 0
        headers = next(reader)
        for row in reader:
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['similarity'] = float(row['SimLex999'])
            row['conc_w1'] = float(row['conc_w1'])
            row['conc_w2'] = float(row['conc_w2'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from simlex999 dataset" % line_count)
    return data

def get_ws353_rel(path=WORDSIM_353_REL_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word1", "word2", "relatedness"])
        line_count = 0
        for row in csv_reader:
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['relatedness'] = float(row['relatedness'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from WordSim relatedness dataset" % line_count)
    return data

def get_verbsim(path=VERBSIM_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=['rank', 'word1', 'word2', 'relatedness'])
        headers = next(csv_reader)
        line_count = 0
        for row in csv_reader:
            row['rank'] = int(row['rank'])
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['relatedness'] = float(row['relatedness'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from VerbSim dataset" % line_count)
    return data

def get_men(path=MEN_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=["word1", "word2", "relatedness"])
        line_count = 0
        for row in csv_reader:
            # remove the POS tag at the end of the word
            row['word1'] = row['word1'].lower()[:-2]
            row['word2'] = row['word2'].lower()[:-2]
            row['relatedness'] = float(row['relatedness'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from MEN relatedness dataset" % line_count)
    return data   

def get_bless(path=BLESS_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["concept", "class", "relation", "relatum"])
        line_count = 0
        for row in csv_reader:
            # remove the POS tag at the end of the word
            row['word1'] = row['concept'].lower()[:-2]
            row['word2'] = row['relatum'].lower()[:-2]
            data.append(row)
            line_count +=1
    print("processed %s word pairs from BLESS dataset" % line_count)
    return data   

def get_brysbaert(path=BRYSBAERT_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        fieldnames = ["Word", "Bigram",  "Conc.M", "Conc.SD", "Unknown", "Total", "Percent_known", "SUBTLEX", "Dom_Pos"]
        csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=fieldnames)
        next(csv_reader)
        line_count = 0
        for row in csv_reader:
            # remove the POS tag at the end of the word
            row['word'] = row['Word'].lower()
            row['concreteness'] = float(row['Conc.M'])
            row['std_dev'] = float(row['Conc.SD'])
            row['dom_pos'] = row['Dom_Pos']
            if float(row["Bigram"]) == 1:
                row['bigram'] = True
            else:
                row['bigram'] = False
            data.append(row)
            line_count +=1
    print("processed %s word pairs from brysbaert dataset" % line_count)
    return data    


def get_simverb3500(path=SIMVERB_PATH):
    data = []
    with open(path, mode='r') as csv_file:
        fieldnames = ['word1', 'word2', 'POS', 'score', 'relation']
        csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=fieldnames)
        line_count = 0
        for row in csv_reader:
            row['word1'] = row['word1'].lower()
            row['word2'] = row['word2'].lower()
            row['similarity'] = float(row['score'])
            data.append(row)
            line_count +=1
    print("processed %s word pairs from brysbaert dataset" % line_count)
    return data    
