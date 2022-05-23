# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:47:32 2022

@author: Kang
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#set manual seed to a constant get a consistent output
manualSeed = 42 #random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Hyperparameters
save_path = './DCGAN_MNIST'
save_path2 = save_path + '/weights'
nCh = 1             # number of channels
z_dim = 100         # input noise dimension
ngf = 64            # number of generator filters
ndf = 64            # number of discriminator filters
nEpochs = 40        # number of epochs
batch_size = 128    # batch size

if not os.path.isdir(save_path):
    os.makedirs(os.path.join(save_path))

if not os.path.isdir(save_path2):
    os.makedirs(os.path.join(save_path2))


#loading the dataset
dataset = datasets.MNIST(root="./data", download=True,
                       transform=transforms.Compose([
                       transforms.Resize(64),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size,
                                         shuffle=True)

#checking the availability of cuda devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False), # input z is going0 into a convolution
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), # (ngf*8) x 4 x 4
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), # (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),   # (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nCh, 4, 2, 1, bias=False),     # (ngf) x 32 x 32
            nn.Tanh()                                              # (nCh) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = Generator().to(device)
netG.apply(weights_init)

#load weights to test the model
#netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
print(netG)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nCh, ndf, 4, 2, 1, bias=False),       # input is (nCh) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),     # (ndf) x 32 x 32
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),   # (ndf*2) x 16 x 16
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),   # (ndf*4) x 8 x 8
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),       # (ndf*8) x 4 x 4
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netD = Discriminator().to(device)
netD.apply(weights_init)

#load weights to test the model 
#netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
print(netD)


criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


fixed_noise = torch.randn(128, z_dim, 1, 1, device=device)

g_loss = []
d_loss = []

for epoch in range(1,nEpochs+1):
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # train with real
        netD.zero_grad()
        
        real_img = data[0].to(device)

        output = netD(real_img)
        
        real_label = torch.ones_like(torch.Tensor(len(output))).to(device)
        fake_label = torch.zeros_like(torch.Tensor(len(output))).to(device)  
        
        errD_real = criterion(output, real_label)
        
        errD_real.backward()
        
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        
        fake_label = torch.zeros_like(torch.Tensor(len(output))).to(device)  
        errD_fake = criterion(output, fake_label)
        
        errD_fake.backward()
        
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        netG.zero_grad()
        
        output = netD(fake)
        real_label = torch.ones_like(torch.Tensor(len(output))).to(device)
        errG = criterion(output, real_label)
        
        errG.backward()
        
        D_G_z2 = output.mean().item()
        
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
              % (epoch, nEpochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
    #save the output 
    fake = netG(fixed_noise)
    save_image(fake.detach(),save_path+'/fake_samples_epoch_%03d.png' % (epoch),normalize=True)
    
    # Check pointing for every epoch
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), save_path2+'/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), save_path2+'/netD_epoch_%d.pth' % (epoch))
    
    
save_image(real_img,save_path+'/real_samples.png',normalize=True)    