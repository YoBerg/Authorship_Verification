import torch
import torch.nn as nn

from collections import defaultdict
import string

# BoW
# dunno if we want to do it like this. My biggest gripe is that word_to_ix is separate from embeddings.
class BagOfWords(nn.Embedding):
    def __init__(self, dataloader):
        self.word_to_ix = defaultdict(lambda: 0) # If word is not in dataset, get 0.
        # Cycle through dataset and get an index for each unique element
        index = 1 # Leave 0 open for null token
        for data in dataloader:
            for word in data:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = index
                    
                    index += 1
                    
        n = index - 1
        super(BagOfWords, self).__init__(n, n)
        self.weight.data.copy_(torch.eye(n))
                    
    # Encode a list of words into embeddings.
    def encode(self, data):
        # Set one-hot encodings
        indices = [self.word_to_ix[word] for word in data]
        one_hot = torch.zeros(len(indices), self.num_embeddings)
        one_hot[range(len(indices)), indices] = 1
        return one_hot

# TF-IDF



# Word2Vec



# GloVe



# OpenAI Text Embedding API




# 
def preprocess_text(text):
    # Remove unwanted characters
    text = text.replace('\n', ' ')  # Replace newline characters with space
    text = text.replace('\t', ' ')  # Replace tab characters with space

    text = text.replace('\'', '') # Remove apostrophes
    
    # Separate punctuation from words
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ' + punctuation + ' ')

    # Convert text to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = text.split()

    return text

if __name__ == "__main__":
    # Test model
    dataset = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc fringilla augue metus, aliquam vehicula lectus ultricies at. Aliquam rhoncus lectus in lobortis varius. Mauris non diam porttitor, laoreet risus non, imperdiet massa. In accumsan diam sit amet augue vehicula consectetur. Mauris in ante sed lorem molestie lobortis quis eget eros. Sed posuere diam ultrices, porta elit id, blandit arcu. Quisque dignissim consequat nisl, ac aliquet eros scelerisque ut. Duis nunc magna, ultrices id purus suscipit, ultrices bibendum felis. Fusce nec accumsan orci.", 
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. In id euismod erat. Donec hendrerit rhoncus neque, eu bibendum ante imperdiet nec. Fusce feugiat tincidunt mollis. Curabitur vitae condimentum lorem. Morbi ut tortor massa. Mauris commodo blandit ex, ac porta nulla vestibulum nec. Suspendisse a nisi vel enim commodo malesuada.",
                    "Cras dui velit, condimentum non porta at, aliquet non dolor. Donec at malesuada massa. Sed feugiat est erat, vel dignissim ligula rhoncus ut. Nullam non quam tincidunt, aliquam ligula ac, varius neque. Aenean nec eros vitae diam consectetur faucibus ac eget nunc. Vestibulum volutpat vestibulum odio, blandit suscipit mauris efficitur nec. Sed vestibulum justo mi, id placerat est fringilla ac. Nulla maximus, libero quis dictum cursus, odio turpis vulputate turpis, sed mollis arcu risus sed nibh. Vestibulum lobortis dapibus felis non suscipit.",
                    "Pellentesque condimentum arcu in justo aliquet suscipit. Etiam pellentesque id nunc vel varius. Integer commodo feugiat augue in pharetra. Quisque pulvinar dolor id purus semper, quis luctus leo cursus. Mauris lacinia elementum augue, in sodales tortor placerat varius. Aliquam vestibulum in dui maximus bibendum. Nam a tellus porttitor, accumsan risus sed, fringilla mauris. Donec efficitur nunc non odio consequat feugiat. In quis mi porta, iaculis lorem sit amet, imperdiet dui. Curabitur id erat in lectus dignissim tempus. Mauris vel leo at sem varius aliquam. Pellentesque vitae magna nulla. Suspendisse interdum sem quis mauris malesuada, at placerat justo congue.",
                    "Proin nec accumsan ante. Interdum et malesuada fames ac ante ipsum primis in faucibus. Etiam sed iaculis sapien. Maecenas libero enim, ornare a facilisis nec, lobortis sit amet nisl. Donec quis aliquam justo. Proin eleifend sollicitudin nisl vel finibus. Nam accumsan turpis orci. Nam fringilla vel tellus quis tempus. Pellentesque vel molestie justo. Phasellus luctus rhoncus eleifend. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vivamus eleifend nisl at dapibus tincidunt. Nullam faucibus ante eu sem lobortis rutrum. Maecenas sit amet tellus eget arcu tincidunt congue ut ut massa. In metus metus, vehicula a ultrices in, malesuada eu nunc."]

    # Preprocess
    # lowercase
    # separate punctuation
    # Split sentences into array

    # Get bag of words embedding. Map applies the function to every element of the dataset.
    bag_of_words = BagOfWords(list(map(preprocess_text, dataset)))

    print(bag_of_words.encode(["amet", "nunc", "."]))
