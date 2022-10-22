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

Simple machine learning models, such as Regression and simple classification models were created using sklearn. These models were not intended to have a high accuracy, but just to make sure that we can feed in the data to the models sucessfully. The images also had to be encoded using numpy. The encoded images were then merged with the encoded text data to a single dataframe fore ease of use. The text data was encoded using the BERT model and tokenizer. The text data was encoded with a mac length of 32, which was kept consistent through the whole project.

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
## Milestone 5 - Create the vision Model

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

## Milestone 6 - Build the Text model

Essentially the same as the previous milestone, but applied to the text data. The model this time was completley custom built with Pytorch. This model has a much better accuracy (90%), ans so i was looking foward to seeing the combined model's accuracy in the next milestone. The training took place over 10 epochs with a batch size of 32.

```python
from Datasets import OverallDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Classifier(nn.Module):
    def __init__(self, input_size: int=768, num_classes: int=13):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(128, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, num_classes))
    def forward(self, input):
        x = self.main(input)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    dataset = OverallDataset(max_length=16)
    epochs = 5
    batch_size = 64
    num_classes = 13
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lr = 0.001
    criteria = nn.CrossEntropyLoss()
    
    model = Classifier()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter()

    for epoch in range(epochs):
        for count_1, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            ALL_ACCURACY = []
            actual_preds = []
            test_correct = 0
            labels = batch[1].to(device)
            text = batch[0][0].to(device)
            image = batch[0][1].to(device)
            preds = model(text)
            labels2 = labels.tolist()
            pred_lst = preds.tolist()
            for count,i in enumerate(pred_lst):
                actual_preds.append((i.index(np.max(i))))
            for count, i in enumerate(actual_preds):
                if i == labels2[count]:
                    if count_1 / len(dataloader) > 0.9:
                        test_correct += 1
            loss = criteria(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = torch.sum(torch.argmax(preds, dim=1) == labels).item() / len(labels)
            #print(accuracy)
            ALL_ACCURACY.append(accuracy)
        print(np.mean(np.array(ALL_ACCURACY)))
    torch.save(model.state_dict(), "FINAL_TEXT.pt")
```

## Milestone 7 - Combine the models

To combine the models, the two Pytorch Datasets also has to be combined. To do this, i couldnt call the previously made datasets, as it was causing errors with loading the Data. The combined dataset was created repeating the previous code in the TextDataset and ImageDataset. The combined model consisted of the two previous models, their output was then concatinated resulting in 26 nodes, which has to be reduced to 13. The combined model had an accuracy of around 70% which is a major improvement over the image model and a slight decrease of the text model, which was to be expected.

```python
import torch
import os
import pandas as pd
from lib2to3.pgen2 import token
from transformers import BertTokenizer
from transformers import BertModel
from Clean_Data import Final_df
from encode_and_merge import encode_images
from text_loader_bert import TextDataSet
from model_2 import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class OverallDataset(torch.utils.data.Dataset):
    def __init__(self, max_length=512):
        super().__init__()
        products = encode_images()
        self.all_data = products
        self.BERT_desc = []
        #Assign the product labels and descriptions - LABELS ALREADY ENCODED
        self.labels = products["Main_Category"].to_list()
        self.descriptions = products["product_description"].to_list()
        self.image_data = products["encoded_image_data"].to_list()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.max_length = max_length

        for i in tqdm(self.descriptions, total = len(self.descriptions)):
            encoded = self.tokenizer.batch_encode_plus([i], max_length=self.max_length, padding="max_length", truncation=True)
            encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
            with torch.no_grad():
                desc = self.model(**encoded).last_hidden_state.swapaxes(1,2)
            desc = desc.squeeze(0)
            self.BERT_desc.append(desc)
    
    def __getitem__(self, index):
        label = self.labels[index]
        text = self.BERT_desc[index]
        image = self.image_data[index]
        return ((text, image), label)

    
    def __len__(self):
        return len(self.all_data)

    def decode_label(self, label):
        self.category_map = {
            "Home & Garden": 1,
            "Baby & Kids Stuff": 2,
            "DIY Tools & Materials": 3,
            "Music, Films, Books & Games": 4,
            "Phones, Mobile Phones & Telecoms": 5,
            "Clothes, Footwear & Accessories": 6,
            "Other Goods": 7,
            "Health & Beauty" :8,
            "Sports, Leisure & Travel": 9,
            "Appliances": 10,
            "Computers & Software": 11,
            "Office Furniture & Equipment": 12,
            "Video Games & Consoles": 13
            }
        self.words = list(self.category_map.keys())
        self.numbers = list(self.category_map.values())
        index = self.numbers.index((label.item()+1))
        return self.words[index]



if __name__ == "__main__":
    dataset = OverallDataset(max_length=50)
    dataloaded = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in tqdm(dataloaded, total=len(dataloaded)):
        print(batch[0][1].size())
```
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
