#!/usr/bin/env python
# coding: utf-8

# In[5]:


# from mnist import MNIST
# from os.path import expanduser
# home = path
# mndata = MNIST(home)

# images, labels = mndata.load_training()
# # or
# images, labels = mndata.load_testing()


# In[6]:


# import os
# from skimage import io
# from os.path import join as pjoin
# path = '/Users/tabaneslami/Desktop/git/data'
# train_path = '/Users/tabaneslami/Desktop/git/data/train'
# test_path = '/Users/tabaneslami/Desktop/git/data/test'
# ct = 0
# for i in X_train:
#     io.imsave(pjoin(train_path, str(ct)+"_"+str(y_train[ct])+".png"), i)
#     ct+=1
# ct = 0
# for i in X_test:
#     io.imsave(pjoin(test_path, str(ct)+"_"+str(y_test[ct])+".png"), i)
#     ct+=1


# In[1]:


import mlflow


# In[2]:


train_path = '/Users/tabaneslami/Desktop/git/data/train'
test_path = '/Users/tabaneslami/Desktop/git/data/test'


# In[15]:


# import pytorch_lightning as pl
# import mlflow.pytorch
# mlflow.pytorch.autolog()
mlflow.end_run()
mlflow.set_tracking_uri("file:///Users/tabaneslami/Documents/mlruns")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

#ex_id = mlflow.create_experiment("Demo PCI-3")
print(ex_id)
mlflow.set_experiment(experiment_name="Demo PCI-3")
# experiment = mlflow.get_experiment_by_name("Demo PCI-3")
# ex_id = experiment.experiment_id
# print(experiment)
batch_size = 34
lr = 0.0003
epochs = 100
with mlflow.start_run():
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
mlflow.end_run()


# In[19]:


mlflow.end_run()


# In[ ]:


from keras.datasets import mnist

#(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# In[ ]:


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target[:,0]
        loss = nn.NLLLoss()
        loss = loss(output, target)#F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target[:,0]
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[ ]:


use_cuda= False
device = torch.device("cuda" if use_cuda else "cpu")
device


# In[ ]:


torch.zeros((10)).shape


# In[ ]:


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


# In[ ]:


training_data, test_data, training_label, test_label = [], [], [], []
ct=0
for i in os.listdir(train_path):
    ct+=1
    if ct==500:
        break
    training_data.append([io.imread(os.path.join(train_path,i))])
    #test_data.append(io.imread(os.path.join(test_path,i)))
    label = int(i.split(".png")[0].split("_")[1])
#     label_vec = np.zeros((1,10))
#     label_vec[0,label]=1
    training_label.append(label)
    #training_data.append(io.imread(train_path))


# In[ ]:


#training_label[0].shape


# In[ ]:


class MnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, arr, lbl, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
        self.transform = transform
        self.data = arr
        self.label = lbl
       # print("insideee", self.label, lbl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform!=None:
            sample = self.transform(self.data[idx])
        else:
            sample =self.data[idx]
        label = (self.label[idx],)
        return torch.FloatTensor(sample), torch.LongTensor(label)


# In[ ]:



dataset = MnistDataset(arr=training_data, lbl=training_label, transform=None)

train_loader = DataLoader(dataset,
                         batch_size=batch_size,
                        )


# In[ ]:


dataset[0][1]


# In[ ]:


model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr)


# In[ ]:


for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    #test(model, device, test_loader)
    optimizer.step()


# In[ ]:


test_path


# In[ ]:


training_data, test_data, training_label, test_label = [], [], [], []
ct=0
for i in os.listdir(test_path):
    ct+=1
    if ct==500:
        break
    test_data.append([io.imread(os.path.join(test_path,i))])
    label = int(i.split(".png")[0].split("_")[1])
    test_label.append(label)


# In[ ]:





# In[ ]:


dataset_test = MnistDataset(arr=test_data, lbl=test_label, transform=None)

test_loader = DataLoader(dataset_test,
                         batch_size=4)


# In[ ]:





# In[ ]:


test(model, device, test_loader)


# In[ ]:




