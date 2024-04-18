import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.downloader as api
import pandas as pd

import torch
from torch.utils.data import IterableDataset

import embeds

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



class Pan20Dataset(IterableDataset):
    """
    Filepath: The filepath to the dataset
    Embedding: The embedding to use. Expects it to have the encode function for a list of words
    """
    def __init__(self, filepath, embedding = None):
        self.reader = pd.read_json(filepath, lines=True, chunksize = 1)
        self.embedding = embedding

    """
    Produces an iterator of the dataset.
    Every call returns the next item in the dataset.
    Return tuple: (id, fandoms - 2 element list, pairs - 2 elements list)
    If an embedding is provided, the pairs are transformed into a tensor of embeddings.
    """
    def __iter__(self):
        for item in self.reader:
            data = item.values[0]
            # Apply transform on dataset.
            if self.embedding:
                for i in range(len(data[2])):
                    processed = embeds.preprocess_text(data[2][i])
                    data[2][i] = self.embedding.encode(processed)
            
            yield data[0], data[1], data[2]

def padding_collate_fn(batch):
    """
    A DataLoader collate function written for the Pan20Dataset that
    automatically pads the text embeddings.
    """
    ids_batch, domains_batch, pairs_batch = zip(*batch)
    
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
    
    return ids_batch, domains_batch, text_batch_1, text_batch_2