# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:53:31 2022

@author: Kang
"""

#!/usr/bin/env python
# coding: utf-8

# 오토인코더

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np



num_epochs = 30
log_interval = 200
noise_factor = 0.2
fig_num = 5
encoded_space_dim = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.ToTensor()])


''' MNIST 데이터 다운로드 (Train set, Test set 분리하기) '''
train_dataset = datasets.MNIST(root="./data", 
                               train=True, 
                               transform=transform, 
                               download=True)

test_dataset = datasets.MNIST(root="./data", 
                              train=False, 
                              transform=transform, 
                              download=True)

train_loader = DataLoader(train_dataset, 
                          batch_size=128, 
                          shuffle=True)

test_loader = DataLoader(test_dataset, 
                         batch_size=32, 
                         shuffle=False)


''' AutoEncoder 모델 설계하기 '''
class Encoder(nn.Module):    
    # def __init__(self, encoded_space_dim,fc2_input_dim):
    def __init__(self, encoded_space_dim):    
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
    
class Decoder(nn.Module):    
    def __init__(self, encoded_space_dim):    
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


''' Optimizer, Objective Function 설정하기 '''
encoder = Encoder(encoded_space_dim)
decoder = Decoder(encoded_space_dim)
encoder.to(device)
decoder.to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)
criterion = torch.nn.MSELoss()


''' 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(encoder, decoder, train_loader, optimizer, noise_factor, log_interval):
    
    encoder.train()
    decoder.train()
    
    train_loss = []
    
    # for image_batch, _ in dataloader: 
    for batch_idx, (image, _) in enumerate(train_loader):
        
        image_noisy = add_noise(image, noise_factor)
        image_noisy = image_noisy.to(device)    
        encoded_data = encoder(image_noisy)
        decoded_data = decoder(encoded_data)
        
        loss = criterion(decoded_data, image_noisy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), len(train_loader.dataset), 
                100.*batch_idx / len(train_loader), loss.item()))
            
        train_loss.append(loss.item())
            


''' 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(encoder, decoder, test_loader):
    
    encoder.eval()
    decoder.eval()
    
    test_loss = 0
    
    with torch.no_grad(): 

        conc_out = []
        conc_label = []
        
        for image, _ in test_loader:
            
            image = image.to(device)
            encoded_data = encoder(image)
            decoded_data = decoder(encoded_data)
            
            conc_out.append(decoded_data.cpu())
            conc_label.append(image.cpu())
            
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        test_loss = criterion(conc_out, conc_label)

    return test_loss



def add_noise(inputs,noise_factor=0.3):
    noisy = inputs+torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy


def plot_ae_outputs(encoder,decoder,n=5,noise_factor=0.3):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
        ax = plt.subplot(3,n,i+1)
        img = test_dataset[i][0].unsqueeze(0)
        image_noisy = add_noise(img,noise_factor)     
        image_noisy = image_noisy.to(device)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            rec_img  = decoder(encoder(image_noisy))

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original Image')

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Noisy Image')

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Reconstructed Image')

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.7, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)     
    plt.show() 



''' Test set Loss & plot '''

for epoch in range(1, num_epochs+1):
    
    train(encoder, decoder, train_loader, optimizer, noise_factor, log_interval)
    
    test_loss = evaluate(encoder, decoder, test_loader)
    
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}\n".format(epoch, test_loss))
    
    plot_ae_outputs(encoder,decoder,fig_num,noise_factor)
    
    
    
    
    
