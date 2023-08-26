import torch
from torch import nn
from torch.utils.data import DataLoader


from torchvision import datasets
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ToTensor, Compose, Resize

from helper_functions import accuracy_fn
from train_and_eval import train_loop, test_loop


device = "cuda" if torch.cuda.is_available() else "cpu"

transformation_train = Compose([
    Resize((224, 224)),
    ToTensor(),
    RandomRotation(degrees=20),
    RandomHorizontalFlip()

])

transformation_test = Compose([
    Resize((224, 224)),
    ToTensor(),

])

train_data = datasets.OxfordIIITPet(
    root="dataset",
    split="trainval",
    transform=transformation_train,
    target_transform=None,
    download=True,
)

test_data = datasets.OxfordIIITPet(
    root="dataset",
    split="test",
    transform=transformation_test,
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


class Vgg16_imageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),

        )
        self.FullLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=512*7*7,
                out_features=4096,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4096,
                out_features=4096,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4096,
                out_features=37,
            )
        )

    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return self.FullLayer(out)


model = Vgg16_imageNet()

foo = torch.rand(1, 3, 224, 224)
foo = model.layer1(foo)
foo.shape
foo = model.layer2(foo)
foo.shape
foo = model.layer3(foo)
foo.shape
foo = model.layer4(foo)
foo.shape
foo = model.layer5(foo)
foo.shape
foo = model.FullLayer(foo)
foo.shape

optim = torch.optim.Adam(
    params=model.parameters(),
    lr=0.01,
)

lossF = nn.CrossEntropyLoss()

train_loop(
    model=model,
    lossF=lossF,
    optim=optim,
    trainDLoader=train_data_loader,
    epochs=1,
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
