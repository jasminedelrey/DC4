import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import nltk
import re
import tensorflow.compat.v1 as tf
import csv

tf.disable_eager_execution()

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

#PREPROCESSING
def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))


def removing_stopwords(stop_words, tokens):
    res = []
    for token in tokens:
        if token not in stop_words:
            res.append(token)
    return res


def process_text(text):
    text = text.encode('ascii', errors='ignore').decode()
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'#+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub('\?', text)
    text = re.sub('\!', text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemma_listt = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, 'v')
        if lemma == token:
            lemma = lemmatizer.lemmatize(token)
        lemma_listt = lemma_listt.append(lemma)
        return lemma_listt


def process_all(text):
    text = process_text(text)
    return ' '.join(removing_stopwords(stop_words, text.split()))


def cosine_similarity(v1, v2):
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    if (not magnitude1) or (not magnitude2):
        return 0
    return np.dot(v1, v2) / (magnitude1 * magnitude2)


def test_similarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    return cosine_similarity(vec1, vec2)


def find_average(a_list):
    inventory = len(a_list)
    for i in a_list:
        total += a_list[i]
    average = total / inventory
    return average


# overall_test = []
# with open('quora_duplicate_questions .tsv') as tsvfile:
#     reader = csv.DictReader(tsvfile, dialect='excel-tab')
#     count = 0
#     for row in reader:
#         overall_test.append((row['id']))
#         overall_test.append(test_similarity(row['question1'], row['question2']))


q1 = input("Enter your first question: ")
q2 = input("Enter your second question: ")
similarity = test_similarity(q1, q2)
if similarity < 0.7:
    print("Your questions are not considered duplicates.")
else:
    print("Your questions are duplicates!")
