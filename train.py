import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Učitavamo naš json fajl
with open("dictionary.json", "r") as f:
    dictionary = json.load(f)

all_words = []
tags = []
xy = []

# Prolazimo kroz svaku rečenicu za svaki kreirani šablon
for item in dictionary["dictionary"]:
    # Dodajemo sve tagove u listu
    tag = item["tag"]
    tags.append(tag)

    for pattern in item["patterns"]:
        # Tokenizujemo svaku reč rečenice
        w = tokenize(pattern)
        # Dodajemo u našu listu svih reči
        all_words.extend(w)
        # kreiramo niz parova [pattern-tag] koji nam je neophodan za trening chatbot-a
        xy.append((w, tag))

# Stemming i uklanjamo interpukcijske znake
ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Uklanjamo duplikate kreiranjem Set-a
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Kreiranje podataka za treniranje chatbot-a
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words za svaku tokenizovanu rečenicu
    bag = bag_of_words(pattern_sentence, all_words)
    # Kreiramo naš X vektor za treniranje chatbot-a
    X_train.append(bag)

    # Uzimamo odgovarajući index našeg tag-a
    label = tags.index(tag)
    y_train.append(label)  # Class labels

# Numpy array - Grid of values
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):
    # učitavanje dataseta
    def __init__(self):
        self.n_samples = len(X_train)  # Broj uzoraka
        self.x_data = X_train
        self.y_data = y_train

    # pristupanje odredjenom indexu dataseta
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # duzina dataseta
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treniranje modela
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            "Epoch [{0}/{1}], Loss: {2:.4f}".format(epoch + 1, num_epochs, loss.item())
        )


print("Final loss: {:.4f}".format(loss.item()))

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print("training complete. file saved to {}".format(FILE))
