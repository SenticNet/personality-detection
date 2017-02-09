import numpy as np
import theano
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import csv
import getpass


def build_data_cv(datafile, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)

    with open(datafile, "rb") as csvf:
        csvreader=csv.reader(csvf,delimiter=',',quotechar='"')
        first_line=True
        for line in csvreader:
            if first_line:
                first_line=False
                continue
            status=[]
            sentences=re.split(r'[.?]', line[1].strip())
            try:
                sentences.remove('')
            except ValueError:
                None

            for sent in sentences:
                if clean_string:
                    orig_rev = clean_str(sent.strip())
                    if orig_rev=='':
                            continue
                    words = set(orig_rev.split())
                    splitted = orig_rev.split()
                    if len(splitted)>150:
                        orig_rev=[]
                        splits=int(np.floor(len(splitted)/20))
                        for index in range(splits):
                            orig_rev.append(' '.join(splitted[index*20:(index+1)*20]))
                        if len(splitted)>splits*20:
                            orig_rev.append(' '.join(splitted[splits*20:]))
                        status.extend(orig_rev)
                    else:
                        status.append(orig_rev)
                else:
                    orig_rev = sent.strip().lower()
                    words = set(orig_rev.split())
                    status.append(orig_rev)

                for word in words:
                    vocab[word] += 1


            datum  = {"y0":1 if line[2].lower()=='y' else 0,
                  "y1":1 if line[3].lower()=='y' else 0,
                  "y2":1 if line[4].lower()=='y' else 0,
                  "y3":1 if line[5].lower()=='y' else 0,
                  "y4":1 if line[6].lower()=='y' else 0,
                  "text": status,
                  "user": line[0],
                  "num_words": np.max([len(sent.split()) for sent in status]),
                  "split": np.random.randint(0,cv)}
            revs.append(datum)


    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype=theano.config.floatX)
    W[0] = np.zeros(k, dtype=theano.config.floatX)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype(theano.config.floatX).itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype=theano.config.floatX)
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            print word

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d" , " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
#    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_mairesse_features(file_name):
    feats={}
    with open(file_name, "rb") as csvf:
        csvreader=csv.reader(csvf,delimiter=',',quotechar='"')
        for line in csvreader:
            feats[line[0]]=[float(f) for f in line[1:]]
    return feats

if __name__=="__main__":
    w2v_file = sys.argv[1]
    data_folder = sys.argv[2]
    mairesse_file = sys.argv[3]
    print "loading data...",
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    num_words=pd.DataFrame(revs)["num_words"]
    max_l = np.max(num_words)
    print "data loaded!"
    print "number of status: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    mairesse = get_mairesse_features(mairesse_file)
    cPickle.dump([revs, W, W2, word_idx_map, vocab, mairesse], open("essays_mairesse.p", "wb"))
    print "dataset created!"

