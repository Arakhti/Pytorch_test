import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import pathlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

from sklearn import preprocessing

localpath = pathlib.Path().resolve()

# Train data
df = pd.read_csv(f"{localpath}/train.csv")
df['Sex'] = pd.factorize(df['Sex'])[0]
# NaN values replaced by mean
df['Age'].fillna(df['Age'].mean(), inplace=True);
df['Fare'].fillna(df['Fare'].mean(), inplace=True);
# Normalization
normalized_df=(df-df.mean())/df.std()
normalized_df['Survived'] = df['Survived'] # We don't normalize labels
print(normalized_df)
print(normalized_df)

# Test data 
test_df = pd.read_csv(f"{localpath}/test.csv")
test_df['Sex'] = pd.factorize(test_df['Sex'])[0]
# NaN values replaced by mean
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True);
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True);
# Normalization
normalized_test_df=(test_df-test_df.mean())/test_df.std()
normalized_test_df['PassengerId'] = test_df['PassengerId'] # We don't normalize ids
print(normalized_test_df.iloc[152])



learning_rate = 1e-1
batch_size = 64
epochs = 25

class TitanicDataset(Dataset):
    
    def __init__(self, pandaDataframe, transform=None):
        self.data = pandaDataframe.filter(items=['Pclass', 'Sex', 'Age', 'Fare'])
        self.label = pandaDataframe['Survived']
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # For now we use features 2 (Pclass), 4 (Sex), 5 (Age), 9 (Fare)
        infos = self.data.iloc[index].astype(np.float32).values
        label = self.label.iloc[index]
        
        if self.transform is not None:
            infos = self.transform(infos)
            
        return infos, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        y.resize_((y.shape[0], 1))
        # load data on GPU
        #X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        binaryPred = torch.round(pred)
        loss, current = loss.item(), batch * len(X)
        correct += (binaryPred == y).sum().item()
        #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Training Error: \n Accuracy: {(100*correct):>0.1f}%,\n")

def test_loop(dataset, model, loss_fn):
    filteredDataset = dataset.filter(items=['Pclass', 'Sex', 'Age', 'Fare'])
    TestTensor = torch.Tensor(filteredDataset.values)
    results = []
    index = 0

    with torch.no_grad():
        for X in TestTensor:
            index += 1
            pred = model(X)
            binaryPred = torch.round(pred)
            if (math.isnan(binaryPred.item())):
                print("binaryPred")

            results = np.append(results, binaryPred.item())
        ids = dataset['PassengerId'].tolist()
        result_df = pd.DataFrame({'PassengerId': ids, 'Survived':results.astype(int)})
        print(result_df)
        result_df.to_csv(f"{localpath}/submission.csv", index = False)


model = NeuralNetwork()

# Initialize the loss function
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


train_dataset = TitanicDataset(normalized_df, transform=torch.from_numpy)
train_dataloader = DataLoader(train_dataset, batch_size=64)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("Done!")
test_loop(normalized_test_df, model, loss_fn)

print("End")
