import torch
from torch import nn
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(model: nn.Module,
               lossF: nn.Module,
               optim: torch.optim.Optimizer,
               trainDLoader: DataLoader,
               epochs: int,
               accuracy_fn,
               device: torch.device = device):
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch, (X, y) in enumerate(trainDLoader):

            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = lossF(y_pred, y)

            optim.zero_grad()

            loss.backward()
            
            optim.step()


def test_loop(model: nn.Module,
              lossF: nn.Module,
              testDLoader: DataLoader,
              accuracy_fn,
              device: torch.device = device):
    model.eval()
    test_loss = 0
    test_accu = 0
    with torch.inference_mode():
        for X, y in testDLoader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)

            test_loss += lossF(y_pred, y)
            test_accu += accuracy_fn(y, y_pred.argmax(dim=1))
        test_loss /= len(testDLoader)
        test_accu /= len(testDLoader)
        print("Testing loss: ", test_loss, "\nTesting Accu: ",
              test_accu)


