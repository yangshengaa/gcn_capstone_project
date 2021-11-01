"""
Store data parser for each dataset. 

Each parser, load_*, returns three components:
- features: the feature matrix (observation x features)
- labels: the corresponding labels for each observation 
- adjacency_matrix: the adjacency matrix of the prescribed graph (None if not defined)
"""

# load packages 
import os
import re
import numpy as np
from scipy import sparse
import pandas as pd
import networkx as nx
import multiprocessing as mp 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import torch

# specify path to read 
DATA_PATH = 'data/'

# ===========================
# ------- for all -----------
# ===========================
def read_data(source: str):
    """
    read data from a specified source
    :param source: the source to read 
    :return features, labels, and adjacency matrix. 
        If the graph structure is not yet defined, return None 
    """
    if source == 'cora':
        return load_cora()
    elif source == 'finefoods':
        return load_finefoods()
    raise NotImplementedError


# ===========================
# -------- cora -------------
# ===========================

def load_cora():
    """
    load cora data, and use DataLoader to prepare data for pytorch 
    :return features, labels, adjacency_matrix:
    """
    # load content (features)
    content = pd.read_table(
        os.path.join(DATA_PATH, 'cora/cora.content'),
        header=None,
        index_col=0
    )
    features, labels = content.iloc[:, :-1], content.iloc[:, -1]
    labels_num = LabelEncoder().fit_transform(labels)

    # load cites (graph)
    cora_edges = []
    with open(os.path.join(DATA_PATH, 'cora/cora.cites')) as f:
        for line in f.readlines():
            v1, v2 = line.strip().split()
            cora_edges.append((int(v2), int(v1)))  # from latter to the front 
    # construct a directed graph 
    cora_graph = nx.DiGraph(cora_edges)
    adjacency_matrix = np.array(nx.attr_matrix(cora_graph, rc_order=content.index.to_list()))

    # convert to tensor
    features = torch.from_numpy(features.astype(np.float32).to_numpy())  # crucial to keep float32
    labels = torch.from_numpy(labels_num).type(torch.LongTensor)
    
    return features, labels, adjacency_matrix


# ==========================
# ------- finefoods --------
# ==========================

SIA_SCORES = SentimentIntensityAnalyzer().polarity_scores
import time 

def load_finefoods():
    """ 
    load and create features from the food review data set 
    
    The parsed features include: 
    - review helpfulness;
    - sentiment of the summary (split into intensity of four categories);
    - bag of words of the review text (counts)
    """
    # load preprocessed features and labels 
    features_path = os.path.join(DATA_PATH, 'finefoods/finefoods_features.npz')
    labels_path = os.path.join(DATA_PATH, 'finefoods/finefoods_labels.txt')
    if os.path.exists(features_path) and os.path.exists(labels_path):
        features = torch.from_numpy(sparse.load_npz(features_path).toarray().astype(np.float32))
        labels = torch.from_numpy(np.loadtxt(labels_path).astype(np.float32))
        return features, labels, None
    
    # if not yet computed, read from raw 
    # load features 
    content_raw = open(os.path.join(DATA_PATH, 'finefoods/finefoods.txt'), 'rb').readlines()
    content_all = ''.join([x.decode('utf-8', 'ignore') for x in content_raw]).lower().strip()
    content_by_review = content_all.split('\n\n')
    # TODO: comment the following line for parsing the entire dataset, if memory permits.
    content_by_review = content_by_review[:2000] # the first 2000 of them for testing 
    # multiprocessing to parse each 
    start = time.time()
    pool = mp.Pool()
    processed_content = pool.map(parse_each_review, content_by_review, chunksize=300)
    pool.close()
    pool.join()
    print('Parsing Raw Data Takes: {}'.format(time.time() - start))
    
    # labels
    labels = np.array([review[2] for review in processed_content])
    np.savetxt(labels_path, labels, fmt='%.1f')  # save preprocessed results

    # features
    features_without_review_bow = np.array([[review[1]] + review[3:7] for review in processed_content])
    # bag of words (bow) for review 
    review_texts = [review[-1] for review in processed_content]
    bows = CountVectorizer(stop_words='english', max_features=10000).fit_transform(review_texts).toarray()  # drop stopwords
    features = np.hstack((features_without_review_bow, bows))
    sparse.save_npz(features_path, sparse.csr_matrix(features))  # convert to sparse to save 
    print('parsed data are saved in files')

    # build graph based on user item interactions
    adjacency_matrix = None 

    # convert to torch tensors 
    features = torch.from_numpy(features.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.float32))
    return features, labels, adjacency_matrix


def parse_each_review(review: str):
    """ 
    parse each review, return features and labels 
    :param review: individual review
    :return [(productId, userId), review_helpfulness, review_score, review_summary_sentiment, review_text]
    """
    # extract information
    pattern = 'product/productid: (.*)\nreview/userid: (.*)\nreview.*review/helpfulness: (.*)\nreview/score: (.*)\nreview.*review/summary: (.*)\nreview/text: (.*)'
    res = re.search(pattern, review, flags=re.DOTALL)
    review_info_list = list(res.groups())
    product_id, user_id, helpfulness_raw, review_score, review_summary, review_text = review_info_list

    # process each segment  
    # compute helpfulness
    helpfulness_raw = helpfulness_raw.split('/')
    helpfulness_score = int(helpfulness_raw[0]) / (int(helpfulness_raw[1]) + 1)  # regularize
    # rate 
    review_score = float(review_score)
    # sentiment of summary 
    sentiment_scores = list(SIA_SCORES(review_summary).values())
    # remove punc and stop words
    review_text = re.sub(r'[^\w\s]', '', review_text)

    # put back 
    review_info_list = [(product_id, user_id), helpfulness_score, review_score] + sentiment_scores + [review_text]
    return review_info_list

# ==================================================
# ------ other datasets should be placed here ------
# ==================================================

# def load_other_datasets(): ...

# for testing purposes only 
if __name__ == '__main__':
    DATA_PATH = '../' + DATA_PATH
    load_finefoods()
