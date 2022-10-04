import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader



class AllDataset(Dataset):
    def __init__(self, file_path, mean_x=None, std_x=None):
        data = np.genfromtxt(file_path, delimiter=',', skip_header = 1)
        x = data[:,1:9]
        y = data[:,9:]
        
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)

        if mean_x is None:
            self.mean_x = self.x.mean(dim = 0) # for each feature
            print("mean_x size", self.mean_x.shape)
            self.std_x = self.x.std(dim = 0)
            print("mean_x size", self.std_x.shape)
        else:
            self.mean_x = mean_x
            self.std_x = std_x

        self.standard_x = (self.x - self.mean_x)/self.std_x
        self.startard_y = (self.y - self.mean_x)/self.std_x
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx,:].float(), self.y[idx,:].float()



class ThreeDataset(Dataset):
    def __init__(self, file_path, mean_x=None, std_x=None, idx=1):
        data = np.genfromtxt(file_path, delimiter=',', skip_header = 1)
        x = data[:,idx:idx+3]
        y = data[:,9:]
        

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)

        print(self.x.shape)
        if mean_x is None:
            self.mean_x = self.x.mean(dim = 0) # for each feature
            print("mean_x size", self.mean_x.shape)
            self.std_x = self.x.std(dim = 0)
            print("std_x size", self.std_x.shape)
        else:
            self.mean_x = mean_x
            self.std_x = std_x
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx,:].float(), self.y[idx,:].float()


class SingleDataset(Dataset):
    def __init__(self, file_path, mean_x=None, std_x=None, idx=1):
        data = np.genfromtxt(file_path, delimiter=',', skip_header = 1)
        x = data[:,idx:idx+1]
        y = data[:,9:]
        
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)

        if mean_x is None:
            self.mean_x = self.x.mean(dim = 0) # for each feature
            print("mean_x size", self.mean_x.shape)
            self.std_x = self.x.std(dim = 0)
            print("std_x size", self.std_x.shape)
        else:
            self.mean_x = mean_x
            self.std_x = std_x
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx,:].float(), self.y[idx,:].float()




class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super().__init__()

        self.input_fc = nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.output_fc = nn.Linear(in_features=hidden_size, out_features=output_dim)

    def forward(self, x):
        # x = [batch size, dim]
        h_1 = F.relu(self.input_fc(x))
        y_pred = self.output_fc(h_1)
        # y_pred = [batch size, output dim]
        return y_pred, h_1

class MLP_Deep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, hidden_size)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x = [batch size, dim]
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]
        return y_pred, h_2


def train(model, iterator, optimizer, criterion, device, dataset):

    epoch_loss = 0
    epoch_mae = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        mae = F.l1_loss(y_pred, y, reduction = 'mean')


        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_mae += mae.item()

    return epoch_loss / len(iterator), epoch_mae / len(iterator)


def evaluate(model, iterator, criterion, device, dataset):

    epoch_loss = 0
    epoch_mae = 0
    epoch_mae_d = np.zeros(dataset.y.shape[1])

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            mae = F.l1_loss(y_pred, y)

            # mae of each metric
            mae_d = F.l1_loss(y_pred, y, reduction='none')
            mae_d = mae_d.mean(axis=0)


            epoch_loss += loss.item()
            epoch_mae += mae.item()
            epoch_mae_d += mae_d.cpu().detach().numpy()

    return epoch_loss / len(iterator), epoch_mae / len(iterator), epoch_mae_d / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    # load dataset
    TRAIN = False
    EPOCHS = 100000
    
    # [Dataset for all feature model]
    train_dataset = AllDataset('./data/train.csv')
    test_dataset = AllDataset('./data/test.csv', mean_x = train_dataset.mean_x, std_x = train_dataset.std_x)

    # [Dataset for single feature model]
    # train_dataset = SingleDataset('./train.csv', idx=1) # idx start from 1 (idx 0 is stuID)
    # test_dataset = SingleDataset('./test.csv', mean_x = train_dataset.mean_x, std_x = train_dataset.std_x, idx=1)

    # [Dataset for three feature model]
    # train_dataset = ThreeDataset('./train.csv', idx=1)
    # test_dataset = ThreeDataset('./test.csv', mean_x = train_dataset.mean_x, std_x = train_dataset.std_x, idx=1)

    train_iterator = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = MLP(input_dim = 8, output_dim = 8) # [Set input_dim compatable with dataset]   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    model = model.to(device)
    criterion = criterion.to(device)

    if TRAIN:
        best_test_loss = float('inf')

        for epoch in range(EPOCHS):

            start_time = time.monotonic()

            train_loss, train_mae = train(model, train_iterator, optimizer, criterion, device, train_dataset)
            test_loss, test_mae, test_mae_d = evaluate(model, test_loader, criterion, device, test_dataset)

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if (epoch+1) % 1000 == 0:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(model.state_dict(), './log/tut1-model-1feat3.pt') # save model
                
                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} ', f'\tTrain MAE: {train_mae:.3f} ')
                print(f'\t Test Loss: {test_loss:.3f} ', f'\tTest MAE: {test_mae:.3f} ', f'\tTest MAE-D: {test_mae_d} ')
    else:
        # load and evaluate model
        ckpt_dir='./log/tut1-model.pt'
        model.load_state_dict(torch.load(ckpt_dir))
        model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(test_loader, desc="Evaluating", leave=False):
                x = x.to(device)
                y = y.to(device)
                y_pred, _ = model(x)
                mae = F.l1_loss(y_pred, y)
                # mae of each metric
                mae_d = F.l1_loss(y_pred, y, reduction='none')
                mae_d = mae_d.mean(axis=0)

                print(f'\tTest MAE: {mae:.3f} ', f'\tTest MAE-D: {mae_d} ')

                out = np.concatenate([x.cpu().detach().numpy(), y.cpu().detach().numpy(), y_pred.cpu().detach().numpy()], axis = 1)
                print(out.shape)
                np.savetxt("./prediction/allfeat_test_result.csv", out, delimiter=",")


if __name__ == "__main__":
    main()
