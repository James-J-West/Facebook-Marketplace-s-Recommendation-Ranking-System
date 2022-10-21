from Datasets import OverallDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from TEXT_MODEL import Classifier
from model_2 import resnet50CNN
import torch.nn.functional as F
from probability import calc_prob

class Combined_model(nn.Module):
    def __init__(self, Text_model, Image_model):
        super(Combined_model, self).__init__()
        self.Text_model = Text_model
        self.Image_model = Image_model
        self.classifier = nn.Sequential(nn.Linear(26, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 13),
                                        nn.Softmax())
        
    def forward(self, x1, x2):
        x1 = self.Text_model(x1)
        x2 = self.Image_model(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("loading model A")
    modelA = Classifier()
    modelA.load_state_dict(torch.load(r"C:\\Users\\james\\Documents\\AI CORE\\FACEBOOK_MARKET\\COMBINED MODEL\\FINAL_TEXT.pt"))
    modelA.eval()
    print("loading model B")
    modelB = resnet50CNN(13)
    modelB.load_state_dict(torch.load(r"C:\\Users\\james\\Documents\\AI CORE\\FACEBOOK_MARKET\\COMBINED MODEL\\19_state.pkl"))
    modelB.eval()
    print("Combining model")
    combined = Combined_model(modelA, modelB).to(device)
    optimizer = optim.Adam(combined.parameters(), lr=0.001)
    writer = SummaryWriter()
    criteria = nn.CrossEntropyLoss()
    print("Loading Dataset")
    dataset = OverallDataset(max_length=32)
    print(len(dataset))
    batch_size = 64
    train_split, test_split = torch.utils.data.random_split(dataset, [10000, 2604])

    train_data = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    print("Data Loaded")
    for epoch in range(10):
        ALL_ACCURACY = []
        for count_1, batch in tqdm(enumerate(train_data), total=len(train_data)):
            actual_preds = []
            probabilities = []
            correct = 0
            labels = torch.tensor(batch[1].to(device).float())
            image_input = batch[0][1].to(device).float()
            text_input = batch[0][0].to(device)
            preds = combined(text_input, image_input)
            labels2 = labels.tolist()
            pred_lst = preds.tolist()
            for count,i in enumerate(pred_lst):
                probabilities.append(calc_prob(torch.as_tensor(i)))
                actual_preds.append((i.index(np.max(i))))
            loss = criteria(preds, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = torch.sum(torch.argmax(preds, dim=1) == labels).item() / len(labels)
            #print(accuracy)
            ALL_ACCURACY.append(accuracy)
            for num, i in enumerate(actual_preds):
                print("PRED:", i, "PROB:", probabilities[num],"LABEL:", labels[num])
        mean = np.mean(np.array(ALL_ACCURACY))
        writer.add_scalar('Accuracy', mean, epoch)
    writer.flush()
    torch.save(combined.state_dict(), "combined_model.pt")




