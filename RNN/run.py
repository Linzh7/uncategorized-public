import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random

torch.set_default_tensor_type(torch.DoubleTensor)

HIDDEN_SIZE = 11
BATCH_SIZE = 256
LAYER = 3
EPOCHS = 15
FIELDS = 128
INPUT_SIZE = 11
USE_GPU = False
NAME = 'BUS'


class BusDataset(Dataset):
    def __init__(self, is_train_set=True):
        # group: 1-29 r error, 29-103 r normal.
        df = pd.read_csv('./full_limit.csv')
        if is_train_set:
            df = pd.concat([df[df['group'] < 22], df[df['group'] > 80]])
        else:
            df = df[(df['group'] > 21) & (df['group'] < 81)]
        ls = np.array(df)
        self.data = ls
        self.len = ls.shape[0]
        self.dataLen = ls.shape[1] - 2

    def __getitem__(self, index):
        if index > self.len - FIELDS:
            # print("Error: index too great")
            return self.__getitem__(random.randint(0, self.len-FIELDS))
        if self.data[index - FIELDS, 0] != self.data[index, 0]:
            # print("Error: not same sample")
            return self.__getitem__(random.randint(0, self.len-FIELDS))
        return self.data[index - FIELDS: index, 2:-1], 1.0 if 1.0 in self.data[index - FIELDS: index, -1] else 2.0

    def __len__(self):
        return self.len

    def getErrorDict(self):
        return {'error': 1.0, 'normal': 2.0}

    def index2label(self, index):
        dic = {1.0: 'error', 2.0: 'normal'}
        return dic[index]

    def getErrorNum(self):
        return 2


def makeTensors(data, label):
    # if(len(data)<101):
    #     return torch.tensor(), torch.tensor()
    # dataLs = []
    # for i in len(data):
    #     dataLs.append(data[i, i+FIELDS, 1:-1])
    # sequenceData = torch.from_numpy(data)
    # sequenceData = torch.from_numpy(data)
    # resultData = torch.from_numpy(label)
    # return createTensor(sequenceData), createTensor(resultData)
    return createTensor(data), createTensor(label)


def createTensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


trainset = BusDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = BusDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

ERROR_TYPES = trainset.getErrorNum()


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=bidirectional, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def __init__hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return createTensor(hidden)

    def forward(self, input):
        # input shape:B * S -> S * B
        # input = input.t()
        batch_size = input.size(0)
        hidden = self.__init__hidden(batch_size)
        # embedding = self.embedding(input)

        # # pack them up
        # gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output


def trainModel():
    total_loss = 0
    for i, (names, types) in enumerate(trainloader, 1):
        inputs, target = makeTensors(names, types)
        output = classifier(inputs)
        loss = criterion(output, target.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')

    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, types) in enumerate(trainloader, 1):
            inputs, target = makeTensors(names, types)
            output = classifier(inputs)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'[Test] Accuracy: {percent}%, {correct} / {total}.')

    return correct / total


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


if __name__ == "__main__":
    classifier = RNNClassifier(INPUT_SIZE, HIDDEN_SIZE, ERROR_TYPES+1, LAYER)
    # classifier = classifier.double()
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = criterion.double()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % EPOCHS)
    acc_list = []
    for epoch in range(1, EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)
        torch.save(classifier, '.\model\{}_{}.pkl'.format(NAME, epoch))
        print("[Save] model saved.")

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()
