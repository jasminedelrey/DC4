import csv
import re
import nltk
from nltk import punkt, word_tokenize
from nltk.corpus import stopwords
import gensim
import pandas as pd
from string import punctuation
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import gensim
import pyemd
from sklearn.metrics import pairwise_distances

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from fuzzywuzzy import fuzz
import scipy
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces, remove_stopwords, \
    stem_text

# clean up data
from sklearn.model_selection import train_test_split

SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}


def clean(text, stem_words=True):
    def pad_str(s):
        return ' ' + s + ' '

    if pd.isnull(text):
        return ''

    #    stops = set(stopwords.words("english"))
    # Clean the text, with the option to stem words.

    # Empty question

    if type(text) != str or text == '':
        return ''

    # Clean the text
    text = re.sub("\'s", " ",
                  text)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    # remove comma between numbers, i.e. 15,000 -> 15000

    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    #     # all numbers should separate from words, this is too aggressive

    #     def pad_number(pattern):
    #         matched_string = pattern.group(0)
    #         return pad_str(matched_string)
    #     text = re.sub('[0-9]+', pad_number, text)

    # add padding to punctuations and special chars, we still need them later

    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)

    #    def pad_pattern(pattern):
    #        matched_string = pattern.group(0)
    #       return pad_str(matched_string)
    #    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text)

    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']),
                  text)  # replace non-ascii word with special word

    # indian dollar

    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE)
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)

    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"

    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
    # Return a list of words
    return text


# #Can read tsvfile through this
with open('quora_duplicate_questions .tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    count = 0
    for row in reader:
        if count <= 11:
            row['question1'] = clean(row['question1'])
            row['question2'] = clean(row['question1'])
#         # count += 1

data = pd.read_csv("/Users/jasmine/PycharmProjects/dc4/quora_duplicate_questions .tsv", sep='\t')

# USING FEATURES
# length based features
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions
data['diff_len'] = data.len_q1 - data.len_q2

# character length based features
data['len_char_q1'] = data.question1.apply(lambda x:
                                           len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x:
                                           len(''.join(set(str(x).replace(' ', '')))))

# word length based features
data['len_word_q1'] = data.question1.apply(lambda x:
                                           len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x:
                                           len(str(x).split()))

# common words in the two questions
data['common_words'] = data.apply(lambda x:
                                  len(set(str(x['question1'])
                                          .lower().split())
                                      .intersection(set(str(x['question2'])
                                                        .lower().split()))), axis=1)

fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1',
        'len_char_q2', 'len_word_q1', 'len_word_q2',
        'common_words']

data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(
    str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x:
                                        fuzz.partial_ratio(str(x['question1']),
                                                           str(x['question2'])), axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:
                                                  fuzz.partial_token_set_ratio(str(x['question1']),
                                                                               str(x['question2'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x:
                                                   fuzz.partial_token_sort_ratio(str(x['question1']),
                                                                                 str(x['question2'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x:
                                          fuzz.token_set_ratio(str(x['question1']),
                                                               str(x['question2'])), axis=1)

data['fuzz_token_sort_ratio'] = data.apply(lambda x:
                                           fuzz.token_sort_ratio(str(x['question1']),
                                                                 str(x['question2'])), axis=1)

fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
        'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
        'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']

# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy

tfv_q1 = TfidfVectorizer(min_df=3,
                         max_features=None,
                         strip_accents='unicode',
                         analyzer='word',
                         token_pattern=r'w{1,}',
                         ngram_range=(1, 2),
                         use_idf=1,
                         smooth_idf=1,
                         sublinear_tf=1,
                         stop_words='english')

tfv_q2 = deepcopy(tfv_q1)
q1_tfidf = tfv_q1.fit_transform(data.question1.fillna(""))
q2_tfidf = tfv_q2.fit_transform(data.question2.fillna(""))

# Using SVD
from sklearn.decomposition import TruncatedSVD

svd_q1 = TruncatedSVD(n_components=180)
svd_q2 = TruncatedSVD(n_components=180)
question1_vectors = svd_q1.fit_transform(q1_tfidf)
question2_vectors = svd_q2.fit_transform(q2_tfidf)

from scipy import sparse

fs3_1 = sparse.hstack((q1_tfidf, q2_tfidf))
tfv = TfidfVectorizer(min_df=3,
                      max_features=None,
                      strip_accents='unicode',
                      analyzer='word',
                      token_pattern=r'w{1,}',
                      ngram_range=(1, 2),
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1,
                      stop_words='english')
q1q2 = data.question1.fillna("")
q1q2 += " " + data.question2.fillna("")
fs3_2 = tfv.fit_transform(q1q2)
fs3_3 = np.hstack((question1_vectors, question2_vectors))
svd = TruncatedSVD(n_components=2,
                   algorithm=str,
                   n_iter=1,
                   random_state=None,
                   tol=0,
                   )

q1q2 = data.question1.fillna("")
q1q2 += " " + data.question2.fillna("")
fs3_4 = svd.fit_transform(q1q2)
tfv = TfidfVectorizer(min_df=3,
                      max_features=None,
                      strip_accents='unicode',
                      analyzer='word',
                      token_pattern=r'w{1,}',
                      ngram_range=(1, 2),
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1,
                      stop_words='english')
fs3_5 = tfv.fit_transform(q1q2)
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# IMPLEMENTING WORD2VEC
stop_words = set(stopwords.words('english'))


def sent2vec(s, model):
    M = []
    words = word_tokenize(str(s).lower())
    for word in words:
        if word not in stop_words:
            if word.isalpha():
                if word in model:
                    M.append(model[word])
                    M = np.array(M)
                    if len(M) > 0:
                        v = M.sum(axis=0)
                        return v / np.sqrt((v ** 2).sum())
                    else:
                        return np.zeros(300)


w2v_q1 = np.array([sent2vec(q, model) for q in data.question1])
w2v_q2 = np.array([sent2vec(q, model) for q in data.question2])
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

data['cosine_distance'] = [cosine(x, y)
                           for (x, y) in zip(w2v_q1, w2v_q2)]
data['cityblock_distance'] = [cityblock(x, y)
                              for (x, y) in zip(w2v_q1, w2v_q2)]
data['jaccard_distance'] = [jaccard(x, y)
                            for (x, y) in zip(w2v_q1, w2v_q2)]
data['canberra_distance'] = [canberra(x, y)
                             for (x, y) in zip(w2v_q1, w2v_q2)]
data['euclidean_distance'] = [euclidean(x, y)
                              for (x, y) in zip(w2v_q1, w2v_q2)]
data['minkowski_distance'] = [minkowski(x, y, 3)
                              for (x, y) in zip(w2v_q1, w2v_q2)]
data['braycurtis_distance'] = [braycurtis(x, y)
                               for (x, y) in zip(w2v_q1, w2v_q2)]

fs4_1 = ['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance', 'euclidean_distance',
         'minkowski_distance', 'braycurtis_distance']
w2v = np.hstack((w2v_q1, w2v_q2))


def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


data['wmd'] = data.apply(lambda x: wmd(x['question1'],
                                       x['question2'], model), axis=1)
model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: wmd(x['question1'],
                                            x['question2'], model), axis=1)
fs4_2 = ['wmd', 'norm_wmd']

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y = data.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
X = data[fs_1 + fs_2 + fs3_4 + fs4_1 + fs4_2]
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)
X = np.hstack((X, fs3_3))

np.random.seed(42)
n_all, _ = y.shape
idx = np.arange(n_all)
np.random.shuffle(idx)
n_split = n_all // 10
idx_val = idx[:n_split]
idx_train = idx[n_split:]
x_train = X[idx_train]
y_train = np.ravel(y[idx_train])
x_val = X[idx_val]
y_val = np.ravel(y[idx_val])

logres = linear_model.LogisticRegression(C=0.1,
                                         solver='sag', max_iter=1000)
logres.fit(x_train, y_train)
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("Logistic regr accuracy: %0.3f" % log_res_accuracy)

