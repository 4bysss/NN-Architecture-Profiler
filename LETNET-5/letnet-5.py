# Basic import
import torch
from torch import nn

# All data-related imports
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from torchvision.transforms.transforms import ToTensor

# Functionaliced train and eval loops
from train_and_eval import train_loop, test_loop

# Some helper functions
from helper_functions import accuracy_fn


# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Prepare both datasets
train_data = datasets.CIFAR10(
    root="dataset",
    train=True,
    transform=ToTensor(),
    target_transform=None,
    download=True
)

test_data = datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=ToTensor(),
    target_transform=None,
    download=True
)


# Prepare batches
BATCH_SIZE = 32

train_data_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data_loader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Prepare our model
class LetNet_5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.Tanh(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.Tanh(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=120,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )

        self.fully_conneted_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=120,
                out_features=84,
            ),
            nn.ReLU()
        )

        self.fully_conneted_2 = nn.Sequential(
            nn.Linear(
                in_features=84,
                out_features=10
            ),
        )

    def forward(self, X):
        out = self.layer_1(X)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.fully_conneted_1(out)
        return self.fully_conneted_2(out)


# Instanciate our model
LetNet_5_model = LetNet_5()


# Train and test loops
EPOCHS = 100  # Provisional
lossF = nn.CrossEntropyLoss()
optiF = torch.optim.Adam(
    params=LetNet_5_model.parameters(),
    lr=0.001
)

train_loop(
    model=LetNet_5_model,
    lossF=lossF,
    optim=optiF,
    trainDLoader=train_data_loader,
    epochs=EPOCHS,
    accuracy_fn=accuracy_fn,
    device=device
)

test_loop(
    model=LetNet_5_model,
    lossF=lossF,
    testDLoader=test_data_loader,
    accuracy_fn=accuracy_fn,
    device=device
)
