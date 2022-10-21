from msilib.schema import Class
from text_loader_bert import TextDataSet
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
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, num_classes))
    def forward(self, input):
        x = self.main(input)
        return x

if __name__ == '__main__':
    dataset = TextDataSet(max_length=16)
    n_epochs=5
    batch_size=32
    num_classes = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    lr = 0.001
    criteria = nn.CrossEntropyLoss()

    classifier = Classifier(num_classes=num_classes)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    all_losses = []

    print(len(dataset))
    print(len(dataloader))
    
    writer = SummaryWriter()
    for epoch in (range(n_epochs)):
        accuracy = 0
        Test_correct = 0
        progress = tqdm(enumerate(dataloader), total = len(dataloader))
        number_of_batches = 0
        for count_1, batch in progress:
            actual_preds = []
            data = batch[0].to(device)
            label = batch[1].to(device)
            preds = classifier(data)
            labels2 = label.tolist()
            pred_lst = preds.tolist()
            for count,i in enumerate(pred_lst):
                actual_preds.append((i.index(np.max(i))))
            for count, i in enumerate(actual_preds):
                if i == labels2[count]:
                    if count_1 / len(dataloader) > 0.9:
                        Test_correct += 1
            #Testing
            if count_1 / len(dataloader) > 0.9:
                number_of_batches += 1
                progress.set_description(f"TESTING: Epoch = {epoch+1}/{n_epochs}. Correct = {Test_correct} / {number_of_batches * batch_size}")
                loss = criteria(preds, label)
            else:
                optimizer.zero_grad()
                preds = classifier(data)
                loss = criteria(preds, label)
                loss.backward()
                optimizer.step()
                accuracy = torch.sum(torch.argmax(preds, dim=1) == label).item() / len(label)
                progress.set_description(f"Epoch = {epoch+1}/{n_epochs}. Acc = {round(torch.sum(torch.argmax(preds, dim=1) == label).item() / len(label), 2)}")
            all_losses.append(loss.item())
        Test_percent = Test_correct / (number_of_batches * batch_size)
        print(Test_correct)
        print(Test_percent)
        writer.add_scalar('Test_Accuracy', Test_percent, epoch)

    torch.save(classifier.state_dict(), "FINAL_TEXT.pt")