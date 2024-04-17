import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.downloader as api

class Dataset():

    glove_embs = api.load("glove-wiki-gigaword-50")

    def __init__(self, labels_url, truth_url):
        labels = []
        with open (labels_url, 'r') as file:
            i = 0
            for line in file:
                if (i >= 10000):
                    break
                labels.append(json.loads(line))
                i += 1

        truths = []
        with open (truth_url, 'r') as file:
            i = 0
            for line in file:
                if (i >= 10000):
                    break
                truths.append(json.loads(line))
                i += 1

        self.labels, self.truth = self.preprocess(labels, truths)

    def preprocess(self, labels, truths):
        preprocessed_labels = []
        for label in labels:
            text_1 = label['pair'][0].lower()
            tokens_1 = word_tokenize(text_1)

            text_2 = label['pair'][1].lower()
            tokens_2 = word_tokenize(text_2)

            stop_words = set(stopwords.words('english'))

            filtered_1 = [word for word in tokens_1 if word not in stop_words]
            filtered_2 = [word for word in tokens_2 if word not in stop_words]

            embedded_1 = self.embed(filtered_1)
            embedded_2 = self.embed(filtered_2)

            preprocessed_labels.append(embedded_1 - embedded_2)

        preprocessed_truth = []
        for truth in truths:
            preprocessed_truth.append(1 if truth['same'] else 0)

        return preprocessed_labels, preprocessed_truth

    def embed(self, filtered):
        return self.glove_embs[filtered].mean(axis=0)