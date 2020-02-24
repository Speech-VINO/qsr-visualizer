import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import json

def obtain_files():
    folders = ['train_annotations', 'test_annotations']
    files = np.array([])
    for folder in folders:
        files = np.append(files, glob("../" + folder + "/*.json"))

    return files

def obtain_corpus(files):
    corpus = []
    labels = []
    for file in tqdm(files):
        data = json.load(open(file, 'r'))
        corpus.append([data['form'][i]['text'] for i in range(len(data['form']))])
        labels.append([data['form'][i]['label'] for i in range(len(data['form']))])

    return corpus, labels

def tfidf_vectorizer(corpus):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([' '.join(c) for c in corpus])
    
    return tfidf

def latent_semantic_analysis():
    svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
    svd.fit(tfidf)

    return svd

def one_hot_encoded_labels(labels):
    one_hot = [pd.get_dummies(labels[i]) for i in range(len(labels))]

    return one_hot

def calculate_similarity(svd):
    similarity = cosine_similarity(svd.components_.T)
    similarity_mean = similarity.mean()
    similarity_std = similarity.std()

    return similarity, simiularity_mean, similarity_std

