import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Obtain hidden layer representation from the RNN layer
        _, hidden = self.rnn(inputs)
        # Pass the hidden state through the output layer to get scores
        z = self.W(hidden[-1])
        # Sum over output for the final vector and apply softmax
        predicted_vector = self.softmax(z)
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)
        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                # Convert to a tensor, ensuring compatibility
                vectors = torch.from_numpy(np.array(vectors)).view(len(vectors), 1, -1)
                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        # Calculate and store training accuracy for this epoch
        training_accuracy = correct / total
        last_train_accuracy = training_accuracy
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, training_accuracy))

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
            vectors = torch.from_numpy(np.array(vectors)).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        validation_accuracy = correct / total
        last_validation_accuracy = validation_accuracy
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))

        # Stopping condition check
        if validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training stopped to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1
