# Text Classification with AG's News Topic Classification Dataset

# Author: Thu Pham

# See example data from
# https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv

# The AG's News Topic Classification dataset is based on the AG dataset, a
# collection of 1,000,000+ news articles gathered from more than 2,000 news
# sources by an academic news search engine. This dataset contains 30,000
# training samples and 1,900 testing samples from 4 largest classes of the
# AG corpus. The total training sample number is 120,000 with 7,600 testing
# samples.

###############################################################################
### Note. Install all required packages prior to importing
###############################################################################
import multiprocessing

import re,string
from pprint import pprint

import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from gensim.models import Word2Vec, LdaMulticore
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import nltk
stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = False

from nltk.stem import PorterStemmer
import pandas as pd

import time

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

from collections import defaultdict
import operator
from sklearn.cluster import SpectralBiclustering
from six import iteritems
from sklearn.metrics.cluster import v_measure_score

#Functionality to turn stemming on or off
STEMMING = False  # judgment call, parsed documents more readable if False

MAX_NGRAM_LENGTH = 1  # try 1 for unigrams... 2 for bigrams... and so on
VECTOR_LENGTH = 1000  # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = False
SET_RANDOM = 9999

##############################
### Utility Functions
##############################
# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']

# text parsing function for entire document string
def parse_doc(text):
    text = text.lower()
    text = re.sub(r'&(.)+', "", text)  # no & references
    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation
    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote
    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings
    if text.isdigit(): text = ""  # omit words that are all digits
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
def parse_words(text):
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text

##############################
### Gather Original Data
##############################
header_list = ['Class', 'Title', 'Description']

data_train = pd.read_csv('data.csv', names=header_list)
data_train = data_train.groupby('Class').apply(lambda x: x.sample(1000))
train_target = data_train['Class']
data_train.head()

data_test = pd.read_csv('test.csv', names=header_list)
test_target = data_test['Class']
data_test.head()

##############################
### Prepare Training Data
##############################
train_tokens = []  # list of token lists for gensim Doc2Vec
train_text = [] # list of document strings for sklearn TF-IDF
labels = []  # use filenames as labels
for doc in data_train['Description']:
    text_string = doc
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    train_tokens.append(tokens)
    train_text.append(text_string)
print('\nNumber of training documents:',
	len(train_text))
print('\nFirst item after text preprocessing, train_text[0]\n',
	train_text[0])
print('\nNumber of training token lists:',
	len(train_tokens))
print('\nFirst list of tokens after text preprocessing, train_tokens[0]\n',
	train_tokens[0])

##############################
### Prepare Test Data
##############################
test_tokens = []  # list of token lists for gensim Doc2Vec
test_text = [] # list of document strings for sklearn TF-IDF
labels = []  # use filenames as labels
for doc in data_test['Description']:
    text_string = doc
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    test_tokens.append(tokens)
    test_text.append(text_string)
print('\nNumber of testing documents:',
	len(test_text))
print('\nFirst item after text preprocessing, test_text[0]\n',
	test_text[0])
print('\nNumber of testing token lists:',
	len(test_tokens))
print('\nFirst list of tokens after text preprocessing, test_tokens[0]\n',
	test_tokens[0])

##############################
### TF-IDF Vectorization
##############################
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH),
    max_features = VECTOR_LENGTH, stop_words='english')
tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)
print('\nTFIDF vectorization. . .')
start_time = time.time()
print('\nTraining tfidf_vectors_training.shape:', tfidf_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT tfidf_vectorizer.fit_transform
tfidf_vectors_test = tfidf_vectorizer.transform(test_text)
print('\nTest tfidf_vectors_test.shape:', tfidf_vectors_test.shape)
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10,
	random_state = SET_RANDOM)
tfidf_clf.fit(tfidf_vectors, train_target)
tfidf_pred = tfidf_clf.predict(tfidf_vectors_test)  # evaluate on test set
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))
print("--- %s seconds ---" % (time.time() - start_time))

###########################################
### Cluster Analysis
###########################################
print('\nCluster Analysis. . .')
k = 4
random_seed = 9999

km = KMeans(n_clusters=k, random_state=random_seed)
km_clusters = km.fit_predict(tfidf_vectors)

# Top Keywords
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

get_top_keywords(tfidf_vectors, km_clusters, tfidf_vectorizer.get_feature_names(), 10)


# Sample of documents in each cluster
pd.set_option('display.max_columns', None)
pd.reset_option('max_columns')

km_clusters = km.labels_.tolist()
data_train['km cluster'] = km_clusters

print(data_train.groupby('km cluster').apply(lambda x: x.sample(10)))


###########################################
### Multidimensional Scaling
###########################################
print('\nMultidimensional Scaling. . .')

n = 2

# TSNE
tsne = TSNE(n_components=n, metric="euclidean", random_state=random_seed)
tsne_fit = tsne.fit_transform(tfidf_vectors.toarray())
data_train['tsne-2d-one-km'] = tsne_fit[:,0]
data_train['tsne-2d-two-km'] = tsne_fit[:,1]
print(data_train.head())

# Visualize with a scatter plot
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one-km", y="tsne-2d-two-km",
    hue="km cluster",
    palette=sns.color_palette("hls", k),
    data=data_train,
    legend="full",
    alpha=0.3
)
#plt.show()

###########################################
### Hierarchical Clustering
###########################################
print('\nHierarchical Clustering. . .')

# AgglomerativeClustering
hcluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
hierarchical_clusters = hcluster.fit_predict(tfidf_vectors.toarray())

# Top Keywords

get_top_keywords(tfidf_vectors, hierarchical_clusters, tfidf_vectorizer.get_feature_names(), 10)

# TSNE
hierarchical_clusters = hcluster.labels_.tolist()
data_train['hierarchial cluster'] = hierarchical_clusters

tsne = TSNE(n_components=n, metric="euclidean", random_state=random_seed)
tsne_fit = tsne.fit_transform(tfidf_vectors.toarray())
data_train['tsne-2d-one-hc'] = tsne_fit[:,0]
data_train['tsne-2d-two-hc'] = tsne_fit[:,1]
print(data_train.head())

# Visualize with a scatter plot
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one-hc", y="tsne-2d-two-hc",
    hue="hierarchial cluster",
    palette=sns.color_palette("hls", k),
    data=data_train,
    legend="full",
    alpha=0.3
)
#plt.show()

###########################################
### Latent Dirichlet Allocation
###########################################
lda_topics = [0,1,2,3]

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=4, random_state=9999)
lda = lda_model.fit(tfidf_vectors)

def get_model_topics(model, vectorizer, topics, n_top_words=10):
    word_dict = {}
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_dict[topics[topic_idx]] = top_features

    return pd.DataFrame(word_dict)

get_model_topics(lda, tfidf_vectorizer, lda_topics, 10)

###########################################
### Spectral Biclustering
###########################################

categories = [1,2,3,4]

y_true = train_target

cocluster = SpectralBiclustering(n_clusters=4,
                                 svd_method='arpack', random_state=9999)

print("Biclustering...")
start_time = time.time()
cocluster.fit(tfidf_vectors)
y_cocluster = cocluster.row_labels_
print("Done in {:.2f}s. V-measure: {:.4f}".format(
    time.time() - start_time,
    v_measure_score(y_cocluster, y_true)))

feature_names = tfidf_vectorizer.get_feature_names()
document_names = list(data_train.Class)
X = tfidf_vectors

def bicluster_ncut(i):
    rows, cols = cocluster.get_indices(i)
    if not (np.any(rows) and np.any(cols)):
        import sys
        return sys.float_info.max
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]
    # Note: the following is identical to X[rows[:, np.newaxis], cols].sum() but
    # much faster in scipy <= 0.16
    weight = X[rows][:, cols].sum()
    cut = (X[row_complement][:, cols].sum() +
           X[rows][:, col_complement].sum())
    return cut / weight


def most_common(d):
    return sorted(iteritems(d), key=operator.itemgetter(1), reverse=True)


bicluster_ncuts = list(bicluster_ncut(i) for i in range(16))
best_idx = np.argsort(bicluster_ncuts)[:16]

print()
print("Best biclusters:")
print("----------------")
for idx, cluster in enumerate(best_idx):
    n_rows, n_cols = cocluster.get_shape(cluster)
    cluster_docs, cluster_words = cocluster.get_indices(cluster)
    if not len(cluster_docs) or not len(cluster_words):
        continue

    # categories
    counter = defaultdict(int)
    for i in cluster_docs:
        counter[document_names[i]] += 1
    cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name)
                           for name, c in most_common(counter)[:4])

    # words
    out_of_cluster_docs = cocluster.row_labels_ != cluster
    out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
    word_col = X[:, cluster_words]
    word_scores = np.array(word_col[cluster_docs, :].sum(axis=0) -
                           word_col[out_of_cluster_docs, :].sum(axis=0))
    word_scores = word_scores.ravel()
    important_words = list(feature_names[cluster_words[i]]
                           for i in word_scores.argsort()[:-11:-1])

    print("bicluster {} : {} documents, {} words".format(
        idx, n_rows, n_cols))
    print("categories   : {}".format(cat_string))
    print("words        : {}\n".format(', '.join(important_words)))
