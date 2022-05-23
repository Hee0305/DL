# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:17:08 2022

@author: admin
"""


''' 1. Module Import '''
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available')



## hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
input_size = 28    # 2100 samples (1KHz) ==> 210 samples (100Hz)
hidden_size = 60    # 
output_size = 10     # number of classifier
seq_length = 28
n_layers = 4
dropout_p = .2
learning_rate = 0.001

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


''' 3. MNIST 데이터 다운로드 (Train set, Test set 분리하기) '''
train_dataset = datasets.MNIST('./data/MNIST',
                               train = True,
                               transform = transform_image,
                               download = True)
test_dataset = datasets.MNIST('./data/MNIST',
                              train = False,
                              transform = transform_image,
                              download = True)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = True)

test_loader = DataLoader(dataset = test_dataset,
                         batch_size = BATCH_SIZE,
                         shuffle = False)


''' 4. 데이터 확인하기 (1) '''
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break


''' 5. 데이터 확인하기 (2) '''
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
    plt.title('Class: ' + str(y_train[i].item()))
    

''' 6. Bi-LSTM 모델 설계하기 '''
class BiLSTM(nn.Module):

    def __init__(self,input_size,seq_length,hidden_size,output_size,n_layers=4,dropout_p=.2):
        
        super().__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p


        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,batch_first=True,dropout=dropout_p,bidirectional=True)
        
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)

        z, _ = self.lstm(x)
        # |z| = (batch_size, h, hidden_size * 2)
        z = z[:, -1]
        # |z| = (batch_size, hidden_size * 2)
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y


''' 7. Optimizer, Objective Function 설정하기 '''
model = BiLSTM(input_size=input_size,seq_length=seq_length,
               hidden_size=hidden_size,output_size=output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

print(model)    
    

''' 8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(model, train_loader, optimizer, log_interval):
    
    model.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
    
        image = image.squeeze().to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        output = model(image)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100.*batch_idx / len(train_loader), 
                loss.item()))
            
            
''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
    
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            
            image = image.squeeze().to(device)
            label = label.to(device)
            
            output = model(image)
            
            test_loss += criterion(output, label).item()
            
            prediction = output.max(1, keepdim = True)[1]
            
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    test_accuracy = 100.*correct / len(test_loader.dataset)
    
    return test_loss, test_accuracy            
            


''' 10. BiLSTM 학습 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
for epoch in range(1, EPOCHS + 1):
    
    train(model, train_loader, optimizer, log_interval = 200)
    
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))