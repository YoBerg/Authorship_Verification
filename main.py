from dataset import DANdataset, TFIDFdataset
from torch.utils.data import DataLoader
import torch
from models import DAN
from sklearn.metrics import roc_auc_score, brier_score_loss
torch_device = torch.device("cpu")

def DAN_embed():
    features_url = './dataset/pan20-authorship-verification-training-small.jsonl'
    truth_url = './dataset/pan20-authorship-verification-training-small-truth.jsonl'

    train_set = DANdataset(features_url, truth_url, 'train')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = DANdataset(features_url, truth_url, 'test')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    # train_set = TFIDFdataset(features_url, truth_url)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    model = DAN("embed")

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    num_epochs = 300
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(torch_device).squeeze(1)
            labels = labels.to(torch_device)

            scores = model(features)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        model.eval()

        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(torch_device).squeeze(1)
                labels = labels.to(torch_device)

                scores = model(features)

                _, predictions = scores.max(1)

                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)

        accuracy = num_correct / num_samples
        print(f'Training Accuracy: {accuracy}')

        model.train()

    groundtruth = []
    final_predictions = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(torch_device).squeeze(1)

            scores = model(features)

            _, predictions = scores.max(1)
            predictions = predictions.tolist()

            for i, prediction in enumerate(predictions):
                final_predictions.append(prediction)
                groundtruth.append(labels[i])

                if (prediction == labels[i] and labels[i] == 1):
                    TP += 1
                elif (prediction == labels[i] and labels[i] == 0):
                    TN += 1
                elif (labels[i] == 0):
                    FP += 1
                else:
                    FN += 1

    model.train()

    print("True Positive: " + str(TP))
    print("True Negative: " + str(TN))
    print("False Positive: " + str(FP))
    print("False Negative: " + str(FN))
    print("ROC AUC Score: " + str(roc_auc_score(groundtruth, final_predictions)))
    print("Brier Score: " + str(brier_score_loss(groundtruth, final_predictions)))

if __name__ == '__main__':
    DAN_embed()