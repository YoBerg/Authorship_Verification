from dataset import Dataset

if __name__ == '__main__':
    features_url = './dataset/pan20-authorship-verification-training-small.jsonl'
    truth_url = './dataset/pan20-authorship-verification-training-small-truth.jsonl'

    train_set = Dataset(features_url, truth_url)