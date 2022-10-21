
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
