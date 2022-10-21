from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_pickle(os.path.realpath(os.path.join(os.path.dirname("__file__"),'Image_data.pkl')))
        self.category_map = {
            "Home & Garden": 0,
            "Baby & Kids Stuff": 1,
            "DIY Tools & Materials": 2,
            "Music, Films, Books & Games": 3,
            "Phones, Mobile Phones & Telecoms": 4,
            "Clothes, Footwear & Accessories": 5,
            "Other Goods": 6,
            "Health & Beauty" :7,
            "Sports, Leisure & Travel": 8,
            "Appliances": 9,
            "Computers & Software": 10,
            "Office Furniture & Equipment": 11,
            "Video Games & Consoles": 12
            }

    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = example[-1]
        label = example[0]
        return (torch.tensor(features).float(), torch.tensor(label))
    
    def decode(self, encoded_value):
        numbers = list(self.category_map.values())
        words = list(self.category_map.keys())
        position = numbers.index(encoded_value)
        decoded = words[position]
        return decoded
    
    def encode(self, decoded_value):
        numbers = list(self.category_map.values())
        words = list(self.category_map.keys())
        position = words.index(decoded_value)
        encoded = numbers[position]
        return encoded
    
    def __len__(self):
        return len(self.data)

class resnet50CNN(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__() 
        self.features = models.resnet50(pretrained=True)
        self.freeze()
        input_shape = self.features.fc.in_features
        self.features.fc = (torch.nn.Sequential(
            torch.nn.Linear(input_shape, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, out_size),
            torch.nn.Softmax(dim=1)
        ))
        self.unfreeze()
    
    def forward(self, x):
        x = self.features(x)
        return x
    
    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad=False
    
    def unfreeze(self):
        for param in self.features.fc.parameters():
            param.requires_grad=True

def train(model, epochs=20):
    optimiser = torch.optim.SGD(model.features.fc.parameters(), lr=0.2, momentum=0.1)
    writer = SummaryWriter()
    batch_idx = 0
    for epoch_num, epoch in (enumerate(range(epochs))):
        losses = []
        correct = 0
        for count, batch in tqdm((enumerate(train_samples)), total=len(train_samples)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            prediction = model(features)
            labels2 = labels.tolist()
            pred_lst = prediction.tolist()
            actual_preds = []
            for count,i in enumerate(pred_lst):
                actual_preds.append((i.index(np.max(i))))
            loss = F.cross_entropy(prediction, labels.long())
            for count, i in enumerate(actual_preds):
                if i == labels2[count]:
                    correct += 1
            losses.append(loss.item())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            batch_idx += 1
        print("Epoch: ", {epoch_num}, "\n", "Avrage Loss: ", np.mean(losses), "\n", "Number Correct: ", correct, "\n", "Accuracy: ", {correct / len(train_data)}, "\n")
        writer.add_scalar('Accuracy', (correct / len(train_data)), epoch_num)
        writer.flush()
    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset()

    batch_size = 64

    train_split = 0.8
    val_split = 0.1

    datasize = len(dataset)
    train_size = int(train_split * datasize)
    val_size = int(val_split * datasize)
    test_size = int(datasize - (val_size + train_size))

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_samples = DataLoader(val_data, batch_size=batch_size)
    test_samples = DataLoader(test_data, batch_size=batch_size)

    model = resnet50CNN(13).to(device)
    train(model)
