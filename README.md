# Facebook-Marketplace-s-Recommendation-Ranking-System

> Facebook Marketplace is a platform for buying and selling products on Facebook. This is an implementation of the system behind the marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

## Milestone 1 - Setting Up the Environment

Set up my dev environment, including signing into AWS and setting up this GitHub Repo

## Milestone 2 - Overview of the System

https://youtu.be/1Z5V2VrHTTA

##Milestone 3 - Exploring the Dataset

By connecting to the S3 Bucket, the tabular dataset and image dataset could be downloaded and explored using pandas. The tabular dataset contained the product description, category, product name, location and product id for each product. The product dataset had to be cleaned. This involved removing the emojis from the descriptions, creating a number for each of the categories (which was saved to be used later). Entries with missing data were also removed

The image dataset contained the photo id , product id and image link to all the images for the dataset. Some products had multiple images and so had to be mapped to each product using the product id. The images also were cleaned. This involved resizing the images to 512x512 and making sure that all images were RGB images and no grayscale images were present.

```python
from torchvision import transforms
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(self.repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        
        # Add a dimension to the image
        image = image[None, :, :, :]
        return image
```

## Milestone 4 - Create simple Machine Learning models

Simple machine learning models, such as Regression and simple classification models were created using sklearn. These models were not intended to have a high accuracy, but just to make sure that we can feed in the data to the models sucessfully. The images also had to be encoded using numpy. The encoded images were then merged with the encoded text data to a single dataframe fore ease of use. The text data was encoded using the BERT model and tokenizer.

```python
df = pd.read_csv(os.path.realpath(os.path.join(os.path.dirname("__file__"), 'FACEBOOK_MARKET', 'Products_clean.csv')), lineterminator="\n")
image_df = pd.read_csv(os.path.realpath(os.path.join(os.path.dirname("__file__"), 'FACEBOOK_MARKET', 'Images.csv')), lineterminator="\n")
merged_df = merge_df(df, image_df)

X_train, X_test, y_train, y_test = train_test_split(merged_df["numpy_data"].tolist(), merged_df["Main_Category"].tolist(), test_size=0.33, random_state=42)

clf = svm.SVC()
clf.fit(X_train[::2], y_train[::2])
preds = clf.predict(X_test)

x = 0
for count, i in enumerate(preds):
    if i != y_test[count]:
        x = x + 1

print(len(y_test) - x)
```
##Milestone 5 - Create the vision Model

First, before a Pytorch model could be used, a Pytorch dataset for both the images and text data had to be created. This was my first time using Pytorch and so took quite a while. Eventually these datasets had to be loaded using the Dataloader in the Pytorch library and fed into the text and image models. The Image model was the main focous of this milestone. The model was based on the Resnet50 model, with an additional couple of layers to tune the outputs to 13 output nodes for each class in the dataset. The training took place over 20 epochs with a batch_size of 64. The model had a 25% accuracy after 20 epochs, which initially caused distress as this is very low. However as this will be combined with a text model, and so is not that big of a worry in the end.

```python 
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
```

##Milestone 6 - Build the Text model


