import torch
import torch.nn as nn
# torch.set_default_dtype(torch.float64)
# torch.use_deterministic_algorithms(True)

import numpy as np
from tqdm import tqdm

from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
import matplotlib.pyplot as plt
from pan20_verif_evaluator import auc, c_at_1, f1, f_05_u_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DAN(torch.nn.Module):
    def __init__(self, data_type):
        super(DAN, self).__init__()
        if (data_type == "embed"):
            self.input_size = 50
        else:
            self.input_size = 1
        self.linear = torch.nn.Linear(in_features=self.input_size, out_features=2)
        self.data_type = data_type

    def forward(self, x):
        return (self.linear(torch.tensor(x)))

class LSTM(torch.nn.Module):
    def __init__(self, dim, bidirectional = False):
        # TODO: Declare RNN model architecture
        super(LSTM, self).__init__()
        self.dataloader = None
        self.test_dataloader = None

        # Setup model
        self.model = nn.LSTM(dim, 64, num_layers = 2, bidirectional = bidirectional, batch_first = True)
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.linear = nn.Sequential(
            nn.Linear(64 * 2, 1),
            # nn.ReLU(),
            # nn.Linear(64,1),
            nn.Sigmoid()
        )
        
    def forward(self, X1, X2):
        # TODO: Implement RNN forward pass
        
        # catted = torch.cat((X1,X2),dim=1) # Try catting after running thru model
        
        o1, _ = self.model(X1)
        o2, _ = self.model(X2)

        # _, (h,_) = self.model(catted) # H is outputted as (D*n_layer, N, H_out)

        f1 = o1[:,-1]
        f2 = o2[:,-1]

        catted = torch.cat((f1,f2),dim=1)

        # return self.cos(f1, f2)
        return self.linear(catted).flatten()
    
    # Sets up the dataloader. Run this before fit.
    def load_data(self, dataloader):
        self.dataloader = dataloader
    
    def load_test_data(self, dataloader):
        self.test_dataloader = dataloader

    def fit(self, epochs = 31, lr = 1e-4, decay = 1e-4, max_run = float("inf")):        
        if self.dataloader is None:
            print("Data has not yet been loaded. Use load_data to set the dataloader.")
            return
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.MSELoss()
        # loss_fn = nn.BCELoss() # Try MSE too?
        run_loss = 0
        n_run = 0
        batch_losses = []
        losses = []
        
        self.train()
        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            self.train()
            batches = iter(self.dataloader)
            
            pbar = tqdm(batches)
            for X1_batch, X2_batch, y_batch in pbar:
                optimizer.zero_grad()
                y_pred = self.forward(X1_batch, X2_batch)

                loss = loss_fn(y_pred, y_batch.float())
                run_loss += loss.item()
                n_run += 1
                
                loss.backward()
                optimizer.step()

                pbar.set_description("Batch Loss: {}".format(loss.item()))
                batch_losses.append(loss.item())

                if n_run > max_run: break
            
            print("Loss: {}".format( run_loss / n_run))
            losses.append(run_loss / n_run)
            run_loss = 0
            n_run = 0
        return losses, batch_losses
        
    # Return prediction labels instead of probabilities.
    def predict(self, X1, X2):
        self.eval()
        output = self.forward(X1, X2)
        return torch.round(output)

    def predict_proba(self, X1, X2):
        self.eval()
        return self.forward(X1, X2)

    def evaluate_dataloader(self, dataloader, shift = 0):
        self.eval()
        
        all_preds = torch.tensor([])
        all_y = torch.tensor([])
        for batch in tqdm(dataloader):
            x1, x2, y = batch
            preds = self.predict_proba(x1,x2).detach().cpu()
            y = y.detach().cpu()
            all_preds = torch.cat((all_preds, preds))
            all_y = torch.cat((all_y, y))

        all_preds = all_preds + shift

        print(classification_report(all_y, torch.round(all_preds)))

        ac_s = auc(all_y, all_preds)
        c1_s = c_at_1(all_y, all_preds)
        f1_s = f1(all_y, all_preds)
        f5_s = f_05_u_score(all_y, all_preds)
        br_s = brier_score_loss(all_y, preds)
        
        print(f"AUC Score: {ac_s}")
        print(f"C@1 Score: {c1_s}")
        print(f"f1  Score: {f1_s}")
        print(f"f.5 Score: {f5_s}")
        print(f"Final Score: {np.average([ac_s, c1_s, f1_s, f5_s])}")

        print(f"Brier Score Loss: {br_s}")

        cm = confusion_matrix(all_y, torch.round(all_preds))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('confusion_matrix.png')