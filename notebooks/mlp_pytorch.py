import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


def prepare_data(X_train, y_train, X_test, y_test):
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return X_train, y_train, X_test, y_test


@dataclass
class NetParams:
    input_features: int
    hidden_size: int
    num_classes: int
    epochs: int
    learning_rate: float


# creating the network
class Net(nn.Module):
    def __init__(self, params: NetParams):
        super(Net, self).__init__()

        # hyperparameter
        self.params = params
        self.input_features = params.input_features
        self.hidden_size = params.hidden_size
        self.num_classes = params.num_classes
        self.epochs = params.epochs
        self.learning_rate = params.learning_rate

        # net
        self.fc1 = nn.Linear(self.input_features, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def fit(self, X_train, y_train, verbose=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # training
        losses = []
        for i in range(self.epochs):
            y_pred = self.forward(X_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss)
            if verbose:
                print(f"epoch: {i:2}  loss: {loss.item():10.8f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.eval()
        """
        y_pred = self(x)
        return y_pred.argmax().item()
        """
        preds_t = self(X)
        y_pred = torch.argmax(preds_t, axis=1)
        return y_pred.detach().numpy()

    def score(self, X_test, y_test):
        """
        self.eval()
        preds = []
        for x in X_test:
            preds.append(self.predict(x))

        y_pred = torch.LongTensor(preds)
        acc = torch.sum(y_pred == y_test) / X_test.shape[0]
        return float(acc)
        """
        self.eval()
        preds_t = self(X_test)
        y_pred = torch.argmax(preds_t, axis=1)
        y_test_t = torch.Tensor(y_test)
        acc = torch.sum(y_pred == y_test_t) / y_pred.shape[0]
        return float(acc)
