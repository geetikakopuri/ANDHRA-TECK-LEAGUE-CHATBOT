import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
file_path = r"C:\Users\asus\Downloads\New folder\intents.json"


with open(file_path, encoding='utf-8') as json_file:
    intents = json.load(json_file)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)  # Convert to a single numpy array
y_train = np.array(y_train, dtype=np.int64)  # Convert to a single numpy array

X_train = torch.tensor(X_train)  # Convert to torch.Tensor with default dtype (float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # Convert to torch.Tensor with dtype torch.long

num_epochs =1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

# Set num_workers to 0 for DataLoader on all platforms
num_workers = 0
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Learning rate scheduler

def train_model():
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(words)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()  # Step the learning rate scheduler after each epoch
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training complete. Model saved to {FILE}')

if __name__ == '__main__':
    train_model()
  