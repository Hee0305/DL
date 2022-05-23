# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:03:51 2022

@author: admin
"""

# GAN 구현하기


from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pylab as plt
import numpy as np

from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
save_path = './vanillaGAN'
save_path2 = save_path + '/training_generated_img_all.png'
save_path3 = save_path + '/generated_img.png'
batch_size = 128
latent_size = 128 
epochs = 200
k = 1 


if not os.path.isdir(save_path):
    os.makedirs(os.path.join(save_path))


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])


train_dataset = datasets.MNIST(root="./data", 
                               train=True, 
                               transform=transform, 
                               download=True)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)


''' GAN 모델 설계하기 '''
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)


''' Optimizer, Objective Function 설정하기 '''
generator = Generator(latent_size).to(device)
discriminator = Discriminator().to(device)
print(generator)
print(discriminator)


optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()

losses_g = [] 
losses_d = [] 
images = [] 


def save_generator_image(image, path):
    save_image(image, path)


''' 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    
    real_label = torch.ones(b_size, 1).to(device)
    fake_label = torch.zeros(b_size, 1).to(device)
    
    optimizer.zero_grad()
    
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    
    return loss_real + loss_fake



def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)    
    
    real_label = torch.ones(b_size, 1).to(device)
    
    optimizer.zero_grad()
    
    output = discriminator(data_fake)
    
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    
    return loss


generator.train()
discriminator.train()

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for idx, data in tqdm(enumerate(train_loader), total=int(len(train_dataset)/train_loader.batch_size)):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        for step in range(k):                                
            data_fake = generator(torch.randn(b_size, latent_size).to(device)).detach()
            data_real = image
            loss_d += train_discriminator(optim_d, data_real, data_fake)
        
        data_fake = generator(torch.randn(b_size, latent_size).to(device))
        loss_g += train_generator(optim_g, data_fake)
    
    generated_img = generator(torch.randn(b_size, latent_size).to(device)).cpu().detach()
    generated_img = make_grid(generated_img)
    
    save_generator_image(generated_img, save_path +"/training_generated_img_{}.png".format(epoch))
    images.append(generated_img)
    
    epoch_loss_g = loss_g / idx 
    epoch_loss_d = loss_d / idx 
    
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    
    print(f"Epoch {epoch} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")


save_image(images,save_path2)


plt.figure()
losses_g = [fl.item() for fl in losses_g ]
plt.plot(losses_g, label='Generator loss')
losses_d = [f2.item() for f2 in losses_d ]
plt.plot(losses_d, label='Discriminator Loss')
plt.legend()



fake_images = generator(torch.randn(batch_size, latent_size).to(device))
save_image(fake_images,save_path3)
for i in range(10):
    fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i],(28, 28))
    plt.imshow(fake_images_img, cmap = 'gray')    
    plt.show()



