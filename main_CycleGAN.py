#!/usr/bin/env python
# coding: utf-8



## CycleGAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as T 
from torch.utils import data

import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


iters = 50000   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Data Loader
# cycleGAN의 경우 두 가지 서로 다른 도메인의 데이터(MNIST, SVHN)가 필요
# 모델이 학습 된 이후에는 MNIST -> SVHN로, 또는 SVHN -> MNIST로 바꿀 수 있음
def MNIST_DATA(root='./data/', download=True, batch_size=32):
    
    transform = T.Compose([T.Resize(32), 
                           T.ToTensor(),
                           T.Normalize([0.5],[0.5])])
    
    mnist_train = datasets.MNIST(root = root,  
                                 train = True, 
                                 transform = transform,
                                 download = True) 
    mnist_test = datasets.MNIST(root = root,
                                train = False,  
                                transform = transform,
                                download = True)

    trainDataLoader = data.DataLoader(dataset = mnist_train, 
                                      batch_size = batch_size,  
                                      shuffle =True) 

    testDataLoader = data.DataLoader(dataset = mnist_test, 
                                     batch_size = batch_size,
                                     shuffle = False) 

    return mnist_train, mnist_test, trainDataLoader, testDataLoader


def SVHN_DATA(root='./data/', download =True, batch_size=32):

    transform = T.Compose([T.Resize(32), 
                           T.ToTensor(),
                           T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    
    svhn_train = datasets.SVHN(root = root,  
                               split = 'train', # SVHN 데이터의 경우 'split' argument로 train/test를 지정
                               transform = transform, 
                               download = True)  

    svhn_test = datasets.SVHN(root = root,
                              split = 'test',  # SVHN 데이터의 경우 'split' argument로 train/test를 지정
                              transform = transform,
                              download = True)
    
    trainDataLoader = data.DataLoader(dataset = svhn_train,  
                                      batch_size = batch_size,  
                                      shuffle =True)

    testDataLoader = data.DataLoader(dataset = svhn_test, 
                                     batch_size = batch_size, 
                                     shuffle = False)
    
    return svhn_train, svhn_test, trainDataLoader, testDataLoader


mnist_trainset, mnist_testset, mnist_trainloader, mnist_testloader = MNIST_DATA(batch_size = 4) 
svhn_trainset, svhn_testset, svhn_trainloader, svhn_testloader = SVHN_DATA(batch_size = 4) 


## MNIST, SVHN 데이터 확인
def show_images_by_class(dataset, cmap=None):
 
    per_class_index = []
    try:
        labels = dataset.targets # dataset의 레이블 정보
    except:
        labels = dataset.labels 
  
    for idx in range(10):
        try:
            per_class_index += [(labels == idx).nonzero()[1].item()]
        except:
            per_class_index += [(labels == idx).nonzero()[0][1].item()]

    images = dataset.data[torch.Tensor(per_class_index).long()]
  
    plt.figure(figsize=(16,160)) # 세로 사이즈 20, 가로 사이즈 20*10  
  
    for a in range(1, 11):      
        plt.subplot(1, 10, a)
        try:
            plt.imshow(images[a-1], cmap)
        except:
            plt.imshow(images[a-1].transpose(1,2,0), cmap)
        plt.xticks([])
        plt.yticks([])  
    plt.show()  


show_images_by_class(mnist_testset, plt.cm.gray)


show_images_by_class(svhn_testset)


## Generator, Discriminator

'''
코드 단순화를 위한 함수들을 정의
'''

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='lrelu'):
    """ 
    코드 단순화를 위한 convolution block 생성을 위한 함수
    Conv -> Batchnorm -> Activation function 으로 이어지는 일련의 레이어를 생성
    """
    layers = []
    
    # Conv.
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    
    # Batchnorm
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    
    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass
                
    return nn.Sequential(*layers)
  
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='lrelu'):
    """ 
    코드 단순화를 위한 deconvolution block 생성을 위한 함수
    Deconv -> Batchnorm -> Activation function 으로 이어지는 일련의 레이어를 생성
    """
    
    layers = []
    
    # Deconv.
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    
    # Batchnorm
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    
    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass
                
    return nn.Sequential(*layers)


'''
Generator와 Discriminator를 선언
MNIST는 흑백 이미지이므로 채널이 1이고, SVHN은 컬러 이미지이므로 채널이 3이다
'''

class Generator(nn.Module):
    """ Generator. """
    def __init__(self, in_dim=1, out_dim=3):
        super(Generator, self).__init__()
        # encoding blocks
        self.conv1 = conv(in_dim, 64, 4)
        self.conv2 = conv(64, 64*2, 4)
        
        # intermediate blocks
        self.conv3 = conv(64*2, 64*2, 3, 1, 1)
        self.conv4 = conv(64*2, 64*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(64*2, 64, 4)
        self.deconv2 = deconv(64, out_dim, 4, bn=False, activation='tanh')
        
    def forward(self, x):
        out = self.conv1(x)      # (B, 64, 16, 16)
        out = self.conv2(out)    # (B, 128, 8, 8)
        
        out = self.conv3(out)    # (B, 128, 8, 8)
        out = self.conv4(out)    # (B, 128, 8, 8)
        
        out = self.deconv1(out)  # (B, 64, 16, 16)
        out = self.deconv2(out)  # (B, out_dim, 32, 32)
        return out


class Discriminator(nn.Module):
    """ Discriminator. """
    def __init__(self, in_dim=1):
        super(Discriminator, self).__init__()
        self.conv1 = conv(in_dim, 64, 4, bn=False)
        self.conv2 = conv(64, 64*2, 4)
        self.conv3 = conv(64*2, 64*4, 4)
        self.fc = conv(64*4, 1, 4, 1, 0, bn=False, activation='none')
        
    def forward(self, x):
        out = self.conv1(x)    # (B, 64, 16, 16)
        out = self.conv2(out)  # (B, 128, 8, 8)
        out = self.conv3(out)  # (B, 256, 4, 4)
        out = self.fc(out).squeeze()
        out = torch.sigmoid(out)
        return out



G_ms = Generator(in_dim=1, out_dim=3).train().to(device)
G_sm = Generator(in_dim=3, out_dim=1).train().to(device)
D_m = Discriminator(in_dim=1).train().to(device)
D_s = Discriminator(in_dim=3).train().to(device)


g_optimizer = optim.Adam(list(G_ms.parameters()) + list(G_sm.parameters()), lr=0.0002, betas=(0.5, 0.99))
d_optimizer = optim.Adam(list(D_m.parameters()) + list(D_s.parameters()), lr=0.0002, betas=(0.5, 0.99))


# Fixed된 이미지를 프린팅하는 함수
def plot_images(images):
    print_list = []
    for i in range(2):
        print_list += [images[0][i], images[1][i], images[2][i]]
  
    for i in range(2):
        print_list += [images[3][i], images[4][i], images[5][i]] 
    plt.figure(figsize=(8,14)) # 세로 사이즈 40, 가로 사이즈 20*3
  
    for a in range(1, 7):
        target_img = print_list[a-1]
        if target_img.shape[0] == 3:
            target_img = target_img.transpose(1, 2, 0)
            cmap = None
        else:
            target_img = target_img.transpose(1, 2, 0).squeeze()
            cmap = plt.cm.gray
        plt.subplot(4, 3, a)
        plt.imshow(target_img, cmap)
        plt.xticks([])
        plt.yticks([])
    for a in range(7, 13):
        target_img = print_list[a-1]
        if target_img.shape[0] == 3:
            target_img = target_img.transpose(1, 2, 0)
            cmap = None
        else:
            target_img = target_img.transpose(1, 2, 0).squeeze()
            cmap = plt.cm.gray
        plt.subplot(4, 3, a)
        plt.imshow(target_img, cmap)
        plt.xticks([])
        plt.yticks([])      
    plt.show()    


## Training CycleGAN 
''' 
loss는 크게 4가지로 나누어 짐
- D: Real images들을 1로 분류하기 위한 loss (d_loss_real)
- D: Fake images들을 0로 분류하기 위한 loss (d_loss_fake)
- G: D를 속이는 Fake images들을 만들기 위한 loss (D에서 1로 분류함)(g_loss (1))
- G: 다시 돌아 갔을 때 reconstruction을 위한 cycle loss (g_loss (2))
'''

# trainig 과정에서 생성되는 이미지가 어떻게 변화하는지 볼 수 있도록 
# 데이터를 고정시킴
mnist_test_iter = iter(mnist_testloader)
svhn_test_iter = iter(svhn_testloader)

# 각 도메인별로 2개만 생성
fixed_mnist = next(mnist_test_iter)[0][:2].to(device)
fixed_svhn = next(svhn_test_iter)[0][:2].to(device)


for i in range(iters):
    
    mnist, m_labels = next(iter(mnist_trainloader)) 
    svhn, s_labels = next(iter(svhn_trainloader))
    mnist = mnist.to(device)
    m_labels = m_labels.to(device)
    svhn = svhn.to(device)
    s_labels = s_labels.to(device)
    
    
    #============= Train the discriminator =============#

    # Real images를 통해 D를 트레이닝
    out = D_m(mnist)
    d_loss_mnist = torch.mean((out-1)**2)

    out = D_s(svhn)
    d_loss_svhn = torch.mean((out-1)**2)

    d_real_loss = d_loss_mnist + d_loss_svhn
  
    d_optimizer.zero_grad()
    d_real_loss.backward()
    d_optimizer.step()

    # Fake images = generated images를 통해 D를 트레이닝
    fake_svhn = G_ms(mnist)
    out = D_s(fake_svhn)
    d_loss_mnist = torch.mean(out**2)

    fake_mnist = G_sm(svhn)
    out = D_m(fake_mnist)
    d_loss_svhn = torch.mean(out**2)

    d_fake_loss = d_loss_mnist + d_loss_svhn
  
    d_optimizer.zero_grad()
    d_fake_loss.backward()
    d_optimizer.step()

    #=============== Train the generator ===============#
    # mnist-svhn-mnist cycle loss를 발생 시킴
    fake_svhn = G_ms(mnist)
    out = D_s(fake_svhn)
    recon_mnist = G_sm(fake_svhn)
  
    g_loss = torch.mean((out-1)**2) 
    g_loss += torch.mean((mnist - recon_mnist)**2) 
  
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # svhn-mnist-svhn loss를 발생 시킴
    fake_mnist = G_sm(svhn)
    out = D_m(fake_mnist)
    recon_svhn = G_ms(fake_mnist)

    g_loss = torch.mean((out-1)**2) 
    g_loss += torch.mean((svhn - recon_svhn)**2) 

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # print the log info
    if i % 200 == 0:    
        # fixed image 로 이미지 생성
        mnist_to_svhn = G_ms(fixed_mnist)
        mnist_recon = G_sm(mnist_to_svhn) 

        svhn_to_mnist = G_sm(fixed_svhn)
        svhn_recon = G_ms(svhn_to_mnist)

        print('Step[%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f' 
                  %(i, iters, d_real_loss.item(), d_fake_loss.item(), g_loss.item()))
        plot_images([fixed_mnist.detach().cpu().numpy()*0.5+0.5, mnist_to_svhn.detach().cpu().numpy()*0.5+0.5, mnist_recon.detach().cpu().numpy()*0.5+0.5
                        ,fixed_svhn.detach().cpu().numpy()*0.5+0.5, svhn_to_mnist.detach().cpu().numpy()*0.5+0.5, svhn_recon.detach().cpu().numpy()*0.5+0.5])


    
