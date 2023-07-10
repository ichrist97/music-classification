from dataclasses import dataclass
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


def prepare_data(X_train, X_test, y_train, y_test):
    # convert to tensors
    X_train_t = Variable(torch.FloatTensor(X_train.to_numpy()))
    X_test_t = Variable(torch.FloatTensor(X_test.to_numpy()))

    y_train_t = Variable(torch.LongTensor(y_train))
    y_test_t = Variable(torch.LongTensor(y_test))

    # reshaping to rows, timestamps, features
    def transform(X):
        return torch.reshape(X, (X.shape[0], 1, X.shape[1]))

    X_train_t_final = transform(X_train_t)
    X_test_t_final = transform(X_test_t)
    return X_train_t_final, X_test_t_final, y_train_t, y_test_t


@dataclass
class LSTMNetParams:
    # hyperparamaters
    num_epochs: int
    learning_rate: float
    dropout: float

    input_size: int  # number of features
    hidden_size: int  # number of features in hidden state
    hidden_layer: int  # number of hidden layers in linear nodes
    num_layers: int  # number of stacked lstm layers
    seq_length: int  # length of series

    num_classes: int  # number of output classes

    tensorboard: bool  # use tensorboard


class LSTMNet(nn.Module):
    def __init__(self, params: LSTMNetParams):
        super(LSTMNet, self).__init__()
        self.params = params
        self.num_classes = params.num_classes  # number of classes
        self.num_layers = params.num_layers  # number of layers
        self.input_size = params.input_size  # input size
        self.hidden_size = params.hidden_size  # hidden state
        self.hidden_layer = params.hidden_layer  # hidden layer size
        self.seq_length = params.seq_length  # sequence length
        self.num_epochs = params.num_epochs
        self.dropout = params.dropout
        self.tensorboard = params.tensorboard

        self.writer = SummaryWriter() if self.tensorboard else None

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # lstm
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_layer)  # fully connected 1
        self.fc = nn.Linear(
            self.hidden_layer, self.num_classes
        )  # fully connected last layer
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # internal state
        x = self.dropout(x)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out

    def fit(self, X_train, y_train, learning_rate=0.01, verbose=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(self.num_epochs):
            self.train()
            # train data
            outputs = self.forward(X_train)  # forward pass
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = criterion(outputs, y_train)
            if self.writer:
                self.writer.add_scalar("loss/train", loss, epoch)
            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop
            if epoch % 100 == 0 and verbose:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        if self.writer:
            self.writer.flush()
            self.writer.close()

    def predict(self, X_test):
        self.eval()
        preds_t = self(X_test)
        y_pred = torch.argmax(preds_t, axis=1)
        return y_pred.detach().numpy()

    def score(self, X_test, y_test):
        self.eval()
        preds_t = self(X_test)
        y_pred = torch.argmax(preds_t, axis=1)
        y_test_t = torch.Tensor(y_test)
        acc = torch.sum(y_pred == y_test_t) / y_pred.shape[0]
        return float(acc)
