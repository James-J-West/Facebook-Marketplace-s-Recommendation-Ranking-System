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




 