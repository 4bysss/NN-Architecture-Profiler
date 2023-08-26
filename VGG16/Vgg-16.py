import torch
from torch import nn
from torch.nn.modules import MaxPool2d, padding
from torch.utils.data import DataLoader, dataloader, dataset

from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

from helper_functions import accuracy_fn
from train_and_eval import test_loop, train_loop

device = "cuda" if torch.cuda.is_available() else "cpu"


transformation = Compose([
    ToTensor(),
    RandomRotation(degrees=20),
    RandomHorizontalFlip()
])


train_data = datasets.CIFAR100(
    root="dataset",
    train=True,
    transform=transformation,
    target_transform=None,
    download=True
)

test_data = datasets.CIFAR100(
    root="dataset",
    train=False,
    transform=ToTensor(),
    target_transform=None,
    download=True
)

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


class Vgg_16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=16
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(
                num_features=32
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=64
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=128
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=256
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        self.layer_6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=256,
                out_features=1024
            ),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(
                in_features=1024,
                out_features=100
            ),
        )

    def forward(self, X):

        out = self.layer_1(X)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        return out
foo = torch.rand(4, 3, 32 ,32)

model = Vgg_16()
foo = model.layer_1(foo)
foo.shape
foo = model.layer_2(foo)
foo.shape
foo = model.layer_3(foo)
foo.shape
foo = model.layer_4(foo)
foo.shape
foo = model.layer_5(foo)
foo.shape
foo = model.layer_6(foo)
foo.shape
lossF = nn.CrossEntropyLoss()
optim = torch.optim.Adam(
    params=model.parameters(),
    lr=0.001
)
'''
train_loop(
    model=model,
    lossF=lossF,
    optim=optim,
    trainDLoader=train_data_loader,
    epochs=10,
    accuracy_fn=accuracy_fn,
    device=device
)
test_loop(
    model=model,
    lossF=lossF,
    testDLoader=test_data_loader,
    accuracy_fn=accuracy_fn,
    device=device
)
'''

