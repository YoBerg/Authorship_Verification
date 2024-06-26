import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import pandas as pd
import numpy as np
from io import StringIO
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset

class DANdataset():

    def __init__(self, labels_url, truth_url, mode, type):
        if (type == "glove"):
            self.embs = api.load("glove-wiki-gigaword-50")
            self.keys = self.embs.index_to_key
        else:
            self.embs = KeyedVectors.load("./w2v_model.kvmodel").wv
            self.keys = self.embs.index_to_key

        labels = []
        with open (labels_url, 'r') as file:
            i = 0
            j = 0
            for line in file:
                if (mode == 'test' and j < 500):
                    j += 1
                    continue
                # print(i)
                if (i >= 500 and i < 29500):
                    i += 1
                    continue
                if (i >= 30000):
                    break
                labels.append(json.loads(line))
                i += 1

        truths = []
        with open (truth_url, 'r') as file:
            i = 0
            j = 0
            for line in file:
                if (mode == 'test' and j < 500):
                    j += 1
                    continue
                if (i >= 500 and i < 29500):
                    i += 1
                    continue
                if (i >= 30000):
                    break
                truths.append(json.loads(line))
                i += 1

        self.labels, self.truth = self.preprocess(labels, truths)

    def preprocess(self, labels, truths):
        preprocessed_labels = []
        i = 0
        for label in labels:
            print(i)
            i += 1
            text_1 = label['pair'][0].lower()
            tokens_1 = word_tokenize(text_1)

            text_2 = label['pair'][1].lower()
            tokens_2 = word_tokenize(text_2)

            stop_words = set(stopwords.words('english'))

            filtered_1 = [word for word in tokens_1 if word not in stop_words and word in self.keys]
            filtered_2 = [word for word in tokens_2 if word not in stop_words and word in self.keys]

            embedded_1 = self.embed(filtered_1)
            embedded_2 = self.embed(filtered_2)

            preprocessed_labels.append(embedded_1 - embedded_2)

        preprocessed_truth = []
        for truth in truths:
            preprocessed_truth.append(1 if truth['same'] else 0)

        return preprocessed_labels, preprocessed_truth

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        return self.labels[idx], self.truth[idx]

    def embed(self, filtered):
        try:
            return self.embs[filtered].mean(axis=0).astype(np.float64)
        except KeyError:
            return np.zeros(50, dtype=np.float32)


class TFIDFdataset():

    def __init__(self, labels_url, truth_url, mode, model):
        self.model = model
        labels = []
        with open (labels_url, 'r') as file:
            i = 0
            j = 0
            for line in file:
                if (mode == 'test' and j < 500):
                    j += 1
                    continue
                print(i)
                if (i >= 1000 and i < 29000):
                    i += 1
                    continue
                if (i >= 30000):
                    break
                labels.append(json.loads(line))
                i += 1

        truths = []
        with open (truth_url, 'r') as file:
            i = 0
            j = 0
            for line in file:
                if (mode == 'test' and j < 500):
                    j += 1
                    continue
                if (i >= 1000 and i < 29000):
                    i += 1
                    continue
                if (i >= 30000):
                    break
                truths.append(json.loads(line))
                i += 1

        self.labels, self.truth = self.preprocess(labels, truths)

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        return self.labels[idx], self.truth[idx]

    def tfidf(self, sentences):
        tfidf = TfidfVectorizer()
        self.tfidf = tfidf.fit(sentences)

    def preprocess(self, labels, truths):
        corpus = []
        for label in labels:
            corpus.append(label['pair'][0].lower())
            corpus.append(label['pair'][1].lower())
        self.tfidf(corpus)

        preprocessed_labels = []
        pairs = [[],[]]
        for label in labels:
            text_1 = label['pair'][0].lower()
            text_2 = label['pair'][1].lower()

            tfidf_1 = self.tfidf.transform([text_1]).toarray()
            tfidf_2 = self.tfidf.transform([text_2]).toarray()

            # while(len(tfidf_1) < 100):
            #     tfidf_1 = np.append(tfidf_1, 0.0)

            # while(len(tfidf_2) < 100):
            #     tfidf_2 = np.append(tfidf_2, 0.0)

            # tfidf_1 = tfidf_1[100:]
            # tfidf_2 = tfidf_2[100:]

            if (self.model == "DAN"):
                print(tfidf_1.mean())
                print(tfidf_2.mean())

                preprocessed_labels.append(tfidf_1.mean() - tfidf_2.mean())
            else:
                pairs[0].append(tfidf_1)
                pairs[1].append(tfidf_2)


        preprocessed_truth = []
        for truth in truths:
            preprocessed_truth.append(1 if truth['same'] else 0)

        if (self.model == "DAN"):
            return preprocessed_labels, preprocessed_truth
        else:
            return pairs, preprocessed_truth

class BoWdataset():

    def __init__(self, labels_url, truth_url, mode):
        labels = []
        with open (labels_url, 'r') as file:
            i = 0
            j = 0
            for line in file:
                if (mode == 'test' and j < 500):
                    j += 1
                    continue
                print(i)
                if (i >= 1000 and i < 29000):
                    i += 1
                    continue
                if (i >= 30000):
                    break
                labels.append(json.loads(line))
                i += 1

        truths = []
        with open (truth_url, 'r') as file:
            i = 0
            j = 0
            for line in file:
                print(i)
                if (mode == 'test' and j < 500):
                    j += 1
                    continue
                if (i >= 1000 and i < 29000):
                    i += 1
                    continue
                if (i >= 30000):
                    break
                truths.append(json.loads(line))
                i += 1

        self.labels, self.truth = self.preprocess(labels, truths)

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        return self.labels[idx], self.truth[idx]

    # def create_bow(self, corpus):
    #     countvectorizer = CountVectorizer()
    #     countvectorizer.fit_transform(corpus)

    #     self.bow = countvectorizer
    #     self.vocab = countvectorizer.vocabulary_

    # def preprocess(self, labels, truths):
    #     corpus = []
    #     for label in labels:
    #         corpus.append(label['pair'][0].lower())
    #         corpus.append(label['pair'][1].lower())

    #     # tokenized_corpus = [word_tokenize(text.lower()) for text in corpus]

    #     # stop_words = set(stopwords.words('english'))
    #     # stemmer = nltk.stem.PorterStemmer()

    #     # filtered_corpus = [[stemmer.stem(word) for word in text if word not in stop_words] for text in tokenized_corpus]

    #     print('a')
    #     self.create_bow(corpus)

    #     preprocessed_labels = []
    #     for label in labels:
    #         print('d')
    #         text_1 = label['pair'][0].lower()
    #         text_2 = label['pair'][1].lower()

    #         text_1 = word_tokenize(text_1.lower())
    #         # text_1 = [stemmer.stem(word) for word in text_1]

    #         freq_1 = 0

    #         # keys = list(bow.keys())

    #         for word in text_1:
    #             if (word in self.vocab):
    #                 freq_1 += self.bow[self.vocab[word]]

    #         text_2 = word_tokenize(text_2.lower())
    #         # text_2 = [stemmer.stem(word) for word in text_2]

    #         freq_2 = 0

    #         # keys = list(bow.keys())

    #         for word in text_2:
    #             if (word in self.vocab):
    #                 freq_2 += self.bow[self.vocab[word]]

    #         preprocessed_labels.append(freq_1 - freq_2)

    #     preprocessed_truth = []
    #     for truth in truths:
    #         preprocessed_truth.append(1 if truth['same'] else 0)

    #     return preprocessed_labels, preprocessed_truth

    def bow(self, sentences):
        vectorizer = CountVectorizer()
        vectorizer.fit(sentences)
        self.vectorizer = vectorizer

    def preprocess(self, labels, truths):
        corpus = []
        for label in labels:
            corpus.append(label['pair'][0].lower())
            corpus.append(label['pair'][1].lower())
        self.bow(corpus)

        preprocessed_labels = []
        pairs = [[],[]]
        for label in labels:
            text_1 = label['pair'][0].lower()
            text_2 = label['pair'][1].lower()

            tfidf_1 = self.vectorizer.transform([text_1]).toarray()
            tfidf_2 = self.vectorizer.transform([text_2]).toarray()

            print(tfidf_1.sum())
            print(tfidf_2.sum())

            preprocessed_labels.append(float(tfidf_1.sum() - tfidf_2.sum()))

        preprocessed_truth = []
        for truth in truths:
            preprocessed_truth.append(1 if truth['same'] else 0)

        return preprocessed_labels, preprocessed_truth


# ##### Don't need this anymore but will keep it here for now #####
#
# class Pan20Dataset_Iterative(IterableDataset):
#     """
#     Filepath: The filepath to the dataset
#     Embedding: The embedding to use. Expects it to have the encode function for a list of words
#     """
#     def __init__(self, train_filepath, truth_filepath, embedding = None):
#         chunksize = 1
#         self.reader = pd.read_json(train_filepath, lines=True, chunksize = chunksize)
#         self.truth_reader = pd.read_json(truth_filepath, lines=True, chunksize = chunksize)
#         self.embedding = embedding()

#     """
#     Produces an iterator of the dataset.
#     Every call returns the next item in the dataset.
#     Return tuple: (id, fandoms - 2 element list, pairs - 2 elements list)
#     If an embedding is provided, the pairs are transformed into a tensor of embeddings.
#     """
#     def __iter__(self):
#         labels = iter(self.truth_reader)
#         for item in self.reader:
#             label = next(labels).values[0]
#             data = item.values[0]
#             # Apply transform on dataset.
#             if self.embedding:
#                 for i in range(len(data[2])):
#                     processed = embeds.preprocess_text(data[2][i])
#                     data[2][i] = self.embedding.encode(processed)
            
#             # Returns id, fandoms, pairs, same_author?, author ids
#             yield data[0], data[1], data[2], label[1], label[2]

class Pan20Dataset(Dataset):
    def __init__(self, X_filepath, y_filepath, embedding = None, dlen=52601):
        self.X_filepath = X_filepath
        self.y_filepath = y_filepath
        self.embedding = embedding

        self.X_offsets = self.get_line_offsets(self.X_filepath, dlen)
        self.y_offsets = self.get_line_offsets(self.y_filepath, dlen)

    def get_line_offsets(self, filepath, pbar_len = None):
        # Compute the byte offset of each line in the file
        line_offsets = []
        if pbar_len:
            pbar = tqdm(total=pbar_len)
        with open(filepath, 'rb') as f:
            offset = 0
            for line in f:
                line_offsets.append(offset)
                offset += len(line)
                if pbar_len:
                    pbar.update(1)
        return line_offsets

    def set_embedding(self, embedding):
        self.embedding = embedding

    def __len__(self):
        return len(self.X_offsets)

    def __getitem__(self, idx):
        # Load and process data from file based on byte offset
        with open(self.X_filepath, 'rb') as f:
            f.seek(self.X_offsets[idx])
            line = f.readline().decode('utf-8')
            # Process the line as needed
            X_sample = pd.read_json(StringIO(line), lines=True)
            
        with open(self.y_filepath, 'rb') as f:
            f.seek(self.y_offsets[idx])
            line = f.readline().decode('utf-8')
            y_sample = pd.read_json(StringIO(line), lines=True)

        text_id = X_sample['id'][0]
        domains = X_sample['fandoms'][0]
        pair = X_sample['pair'][0]
        same = y_sample['same'][0]
        authors = y_sample['authors'][0]
        
        if self.embedding:
            for i in range(len(pair)):
                # processed = preprocess_text(pair[i])
                processed = word_tokenize(pair[i])
                pair[i] = self.embedding.encode(processed)

        return text_id, domains, pair, same, authors

def padding_collate_fn(batch):
    """
    A DataLoader collate function written for the Pan20Dataset that
    automatically pads the text embeddings.
    """
    ids_batch, domains_batch, pairs_batch, same_batch, authors_batch = zip(*batch)
    
    # Get the length of the longest set in the dataset
    max_len = max(max(len(pair[0]) for pair in pairs_batch),
                  max(len(pair[1]) for pair in pairs_batch))
    
#     dim = pairs_batch[0][0].shape[1]
    
    padded_tensors1 = []
    padded_tensors2 = []
    for pair in pairs_batch:
        # Extract individual tensors from the pair
        tensor1, tensor2 = pair
        # Pad the tensors to max_len
        padded_tensor1 = torch.nn.functional.pad(tensor1, (0, 0, 0, max_len - tensor1.size(0)))
        padded_tensor2 = torch.nn.functional.pad(tensor2, (0, 0, 0, max_len - tensor2.size(0)))
        
        padded_tensors1.append(padded_tensor1)
        padded_tensors2.append(padded_tensor2)
        
    text_batch_1 = torch.stack(padded_tensors1, dim=0)
    text_batch_2 = torch.stack(padded_tensors2, dim=0)

    same_batch = torch.tensor(np.array(same_batch, dtype=int), dtype=torch.long)
    
    return text_batch_1, text_batch_2, same_batch

def text_only_collate_fn(batch):
    """
    A DataLoader collate function that returns only the raw text.
    """
    _, _, pairs_batch, _, _ = zip(*batch)
    texts = []
    for pair in pairs_batch:
        text1, text2 = pair
        texts.append(text1)
        texts.append(text2)

    return texts

def cutting_collate_fn(batch):
    """
    A DataLoader collate function written for the Pan20Dataset that
    automatically truncates the text embeddings to the size of the shortest example.
    """
    ids_batch, domains_batch, pairs_batch, same_batch, authors_batch = zip(*batch)
    
    # Get the length of the longest set in the dataset
    min_len = min(min(len(pair[0]) for pair in pairs_batch),
                  min(len(pair[1]) for pair in pairs_batch))
    
#     dim = pairs_batch[0][0].shape[1]
    
    cut_tensors1 = []
    cut_tensors2 = []
    for pair in pairs_batch:
        # Extract individual tensors from the pair
        tensor1, tensor2 = pair
        # Pad the tensors to max_len
        cut_tensor1 = tensor1[:min_len]
        cut_tensor2 = tensor2[:min_len]
        
        cut_tensors1.append(cut_tensor1)
        cut_tensors2.append(cut_tensor2)
        
    text_batch_1 = torch.stack(cut_tensors1, dim=0)
    text_batch_2 = torch.stack(cut_tensors2, dim=0)

    same_batch = torch.tensor(np.array(same_batch, dtype=int), dtype=torch.long)
    
    return text_batch_1, text_batch_2, same_batch