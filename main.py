from dataset import DANdataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    features_url = './dataset/pan20-authorship-verification-training-small.jsonl'
    truth_url = './dataset/pan20-authorship-verification-training-small-truth.jsonl'

    train_set = DANdataset(features_url, truth_url)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)