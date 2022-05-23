# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:52:03 2022

@author: Kang
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


''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available')



## hyperparameters
batch_size = 64
test_batch_size = 1000
epochs = 50
lr = 0.001

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
                          batch_size = batch_size,
                          shuffle = True)

test_loader = DataLoader(dataset = test_dataset,
                         batch_size = test_batch_size,
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
    plt.imshow(np.transpose(X_train[i], (1,2,0)))
    plt.title('Class: ' + str(y_train[i].item()))
    


''' 6. Convolutional Neural Network (CNN) 모델 설계하기 '''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        # self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(14*14*64, 128)
        self.fc2 = nn.Linear(128, 10)

        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = torch.flatten(x,1)
        # x = x.view(-1, 14 * 14 * 64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x


''' 7. Optimizer, Objective Function 설정하기 '''
model = CNN().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

print(model)    
    

''' 8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(model, train_loader, optimizer, log_interval):
    
    model.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
    
        image = image.to(device)
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
            
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
            
            test_loss += criterion(output, label).item()
            
            prediction = output.max(1, keepdim = True)[1]
            
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset) / test_batch_size)
    test_accuracy = 100.*correct / len(test_loader.dataset)
    
    return test_loss, test_accuracy            
            


''' 10. CNN 학습 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
for epoch in range(1, epochs + 1):
    
    train(model, train_loader, optimizer, log_interval = 200)
    
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))
    
    

# Test 데이터셋에서 배치크기만큼 이미지 추출
dataiter = iter(test_loader)
images, labels = dataiter.next()

# 학습된 CNN 모델을 통해 예측값 출력
images = images.to(device)
output = model(images)

# 10개의 성분을 가지는 예측값 벡터에서 최대의 확률을 가지는 인덱스를 예측값으로 반환
_, preds = torch.max(output, 1)

# 이미지 plot을 위해 텐서에서 numpy 자료구조로 변환
images = images.to(torch.device("cpu"))
images = images.numpy()

# 이미지 plot: 레이블과 예측값이 다르면 적색으로 표시 ()안에 레이블
fig = plt.figure(figsize=(16,16))
for idx in np.arange(36):
    ax = fig.add_subplot(6,6,idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), 
                 color=("green" if preds[idx] == labels[idx] else "red"))
    
    