from unittest.util import _MAX_LENGTH
import torch
import os
import pandas as pd
from lib2to3.pgen2 import token
from transformers import BertTokenizer
from transformers import BertModel

class TextDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir="Products_clean.csv", max_length=512):
        #initialise file path
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"{self.root_dir} DONT EXIST")

        #Read in the data
        products = pd.read_csv(self.root_dir, lineterminator="\n")

        #Assign the product labels and descriptions - LABELS ALREADY ENCODED
        self.labels = products["Main_Category"].to_list()
        self.descriptions = products["product_description"].to_list()

        #WOULD BE USEFUL TO HAVE TOTAL NUMBER OF CLASSES
        self.num_classes = len(set(self.labels))

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.max_length = max_length

    def __getitem__(self, index):

        label = self.labels[index]
        label = torch.as_tensor(label)

        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding="max_length", truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            desc = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        desc = desc.squeeze(0)
        return desc, label

    def __len__(self):
        return len(self.labels)
        


if __name__ == '__main__':
    dataset = TextDataSet()
    print(dataset[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
    for count, batch in enumerate(dataloader):
        print(batch[0])
        print(batch[0].size())