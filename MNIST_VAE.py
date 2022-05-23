# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:46:15 2022

@author: Kang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hyperparameters
save_path = './vae'
x_dim  = 784
h_dim1 = 512
h_dim2 = 256
z_dim = 2#200
epochs = 30
batch_size = 100

if not os.path.isdir(save_path):
    os.makedirs(os.path.join(save_path))
    
    

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
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)



class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.mean = nn.Linear(h_dim2, z_dim)
        self.var = nn.Linear(h_dim2, z_dim)
 
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mean(h), self.var(h) # mu, log_var
    
    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterization(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE(x_dim=x_dim, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim).to(device)
    
print(vae)


optimizer = optim.Adam(vae.parameters())

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))



def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



for epoch in range(1, epochs+1):
    
    train(epoch)
    test()


##
with torch.no_grad():
    z = torch.randn(64, z_dim).to(device)
    sample = vae.decoder(z).to(device)
    
    save_image(sample.view(64, 1, 28, 28), './vae/sample_' + '.png')    
    
