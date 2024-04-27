print("Starting imports...")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

import nltk
# from nltk.tokenize import word_tokenize

from collections import defaultdict
import string
from tqdm import tqdm
from io import StringIO
import random

import embeds
from dataset import Pan20Dataset, text_only_collate_fn, cutting_collate_fn
from models import LSTM
# import gensim.downloader as api
# from gensim.models import KeyedVectors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./dataset"
TEST_FILE = "pan20-authorship-verification-test.jsonl"
TEST_TRUTH_FILE = "pan20-authorship-verification-test-truth.jsonl"
TRAIN_FILE = "pan20-authorship-verification-training-small.jsonl"
TRUTH_FILE = "pan20-authorship-verification-training-small-truth.jsonl"
PATH_TO_TRAIN = os.path.join(DATA_ROOT, TRAIN_FILE)
PATH_TO_TRUTH = os.path.join(DATA_ROOT, TRUTH_FILE)
PATH_TO_TEST_DATA = os.path.join(DATA_ROOT, TEST_FILE)
PATH_TO_TEST_TRUTH = os.path.join(DATA_ROOT, TEST_TRUTH_FILE)

sample_size = 2048
test_size = 2048

def plot_learning_curve(train_losses, test_losses=None, dev_losses=None, file = None):
    """
    Plot learning curve given a list of training and optionally test losses.
    
    Args:
    train_losses (list): List of training losses.
    test_losses (list, optional): List of test losses. If not provided, only training curve will be plotted.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plotting training curve
    plt.plot(epochs, train_losses, label='Training Accuracy', color='blue')
    
    # Plotting test curve if provided
    if dev_losses:
        plt.plot(epochs, dev_losses, label='Validation Accuracy', color='orange')

    # Plotting test curve if provided
    if test_losses:
        plt.plot(epochs, test_losses, label='Test Accuracy', color='red')

    
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # matplotlib.pyplot.show()
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()
    
print("Loading embedding model...")

glove = embeds.GloVe("glove.kvmodel")

print("Creating dataset...")

dataset = Pan20Dataset(PATH_TO_TRAIN, PATH_TO_TRUTH, embedding = glove, dlen=52601)

test_dataset = Pan20Dataset(PATH_TO_TEST_DATA, PATH_TO_TEST_TRUTH, embedding = glove, dlen=14311)

indices = list(range(len(dataset)))

random.shuffle(indices)

# random_sampler = RandomSampler(dataset)
# subset_sampler = SubsetRandomSampler(list(RandomSampler(dataset, num_samples=64)))

test_indices = list(range(len(test_dataset)))
random.shuffle(test_indices)

loader = DataLoader(dataset, batch_size=64, collate_fn = lambda x: tuple(x_.to(device) for x_ in cutting_collate_fn(x)), sampler=indices[:sample_size])
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn = lambda x: tuple(x_.to(device) for x_ in cutting_collate_fn(x)), sampler=test_indices[:test_size])

print("Creating model...")


lstm = LSTM(50)
lstm = lstm.to(device)
lstm.load_data(loader)

print("Beginning training...")

losses, batch_losses = lstm.fit(epochs = 16, lr = 1e-3, decay = 1e-4)

print("Saving results...")

torch.save(lstm.state_dict(), "lstm.pt")
plot_learning_curve(losses, file="train_curve.png")

print("Showing Evaluation...")

lstm.evaluate_dataloader(test_loader)

print("Done!")