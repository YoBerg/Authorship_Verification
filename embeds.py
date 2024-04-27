import torch
import torch.nn as nn

import pandas as pd

from collections import defaultdict
from tqdm import tqdm
import string

# from dataset import Pan20Dataset, text_only_collate_fn
from torch.utils.data import DataLoader
import gensim.downloader as api
from gensim.models import KeyedVectors

import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# BoW
# dunno if we want to do it like this. My biggest gripe is that word_to_ix is separate from embeddings.
class BagOfWords(nn.Embedding):
    def __init__(self, dataloader, words=False):
        self.word_to_ix = defaultdict(lambda: 0) # If word is not in dataset, get 0.
        # Cycle through dataset and get an index for each unique element
        index = 1 # Leave 0 open for null token
        pbar = tqdm(dataloader)
        if not words:
            for batch in pbar:
                for text in batch:
                    tokens = word_tokenize(text)
                    for token in tokens:
                        if token not in self.word_to_ix:
                            self.word_to_ix[token] = index
                            
                            index += 1
                pbar.set_description("Size: {}".format(index))
        else:
            for token in pbar:
                if token not in self.word_to_ix:
                    self.word_to_ix[token] = index
                    
                    index += 1
                pbar.set_description("Size: {}".format(index))
                    
        n = index
        super(BagOfWords, self).__init__(n, n)
        self.weight.data.copy_(torch.eye(n))
                    
    # Encode a list of words into embeddings.
    def encode(self, data):
        # Set one-hot encodings
        indices = [self.word_to_ix[word] for word in data]
        one_hot = torch.zeros(len(indices), self.num_embeddings)
        one_hot[range(len(indices)), indices] = 1
        return one_hot
    
    # Just a redirect to encode
    def embed(self, data):
        return self.encode(data)

# TF-IDF



# Word2Vec
# Needs to be trained.
class Word2Vec(nn.Embedding):
    def __init__(self, dataloader, dim, tokenizer):
        self.word_to_ix = defaultdict(lambda: 0) # If word is not in dataset, get 0.
        # Cycle through dataset and get an index for each unique element
        index = 1 # Leave 0 open for null token
        pbar = tqdm(dataloader)
        for batch in pbar:
            for text in batch:
                tokens = tokenizer(text)
                for token in tokens:
                    if token not in self.word_to_ix:
                        self.word_to_ix[token] = index
                        
                        index += 1
            pbar.set_description("Size: {}".format(index))
                    
        n = index - 1
        super(Word2Vec, self).__init__(n, dim)

    def save(self, filename):
        # TODO. How to save both word_to_ix and weights (preferably together)
        pass

    def encode(self, data):
        indices = torch.tensor([self.word_to_ix[word] for word in data])
        return self(indices)

    # Just a redirect to encode
    def embed(self, data):
        return self.encode(data)


    # Fit a continuous bag of words model
    def fit(self, dataloader):
        pass


# GloVe
class GloVe(nn.Embedding):
    def __init__(self, file = None):
        if file:
            self.kv = KeyedVectors.load(file)
        else:
            self.kv = api.load("glove-wiki-gigaword-50")
            
        super(GloVe, self).__init__(len(self.kv), self.kv.vector_size)
        self.weight.data.copy_(torch.from_numpy(self.kv.vectors))
        self.word_to_ix = defaultdict(lambda:0,self.kv.key_to_index)

    def save(self, filename):
        self.kv.save(filename + '.kvmodel')

    def encode(self, data):
        indices = torch.tensor([self.word_to_ix[word] for word in data])
        return self(indices)
        
    # Just a redirect to encode
    def embed(self, data):
        return self.encode(data)




# Use other library's preprocessors
# def preprocess_text(text):
#     # Remove unwanted characters
#     text = text.replace('\n', ' ')  # Replace newline characters with space
#     text = text.replace('\t', ' ')  # Replace tab characters with space

#     text = text.replace('\'', '') # Remove apostrophes
    
#     # Separate punctuation from words
#     for punctuation in string.punctuation:
#         text = text.replace(punctuation, ' ' + punctuation + ' ')

#     # Convert text to lowercase
#     text = text.lower()

#     text = text.split()

#     return text

# Mostly as an example of how to do this.
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(prog="embeds.py", description='Train an embedding model')
#     parser.add_argument('-i', '--input', type=str, help='Dataset Input', required=True)
#     parser.add_argument('--model', choices=['bow', 'tfidf', 'word2vec','glove','openai'], 
#                         default='bow', type=str, help='Embedding Model to learn')
#     parser.add_argument('--full_dataset', default=False, action='store_false', 
#                         help='Train on a most common words dataset (useful for bow)')
#     parser.add_argument('--n', type=int, default=10000, help='Number of words to train on (for common words)')
#     args = parser.parse_args()

#     filepath = args.input

#     if args.model == "bow":
#         if not args.full_dataset:
#             print("Training Bag of Words model")
#             tokens = pd.read_csv(filepath)["word"].values[:args.n]
#             bag_of_words = BagOfWords(tokens, words=True)
#         else:
#             # Ok I know this doesn't work right now, but this model 
#             # takes a full hour to train on my machine so... Can't recommend
#             dataset = Pan20Dataset(filepath, PATH_TO_TRUTH)
#             loader = DataLoader(dataset, batch_size=64, collate_fn = text_only_collate_fn, shuffle=True)
#             bag_of_words = BagOfWords(loader)
