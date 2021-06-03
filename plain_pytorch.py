import os
import subprocess as sp
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST, read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive

##########################################################################
# using this function because of downloading mnist bug
def patched_download(self):
    """wget patched download method.
    """
    if self._check_exists():
        return

    os.makedirs(self.raw_folder, exist_ok=True)
    os.makedirs(self.processed_folder, exist_ok=True)

    # download files
    for url, md5 in self.resources:
        filename = url.rpartition("/")[2]
        download_root = os.path.expanduser(self.raw_folder)
        extract_root = None
        remove_finished = False

        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        # Use wget to download archives
        sp.run(["wget", url, "-P", download_root])

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        extract_archive(archive, extract_root, remove_finished)

    # process and save as torch files
    print("Processing...")

    training_set = (
        read_image_file(os.path.join(self.raw_folder, "train-images-idx3-ubyte")),
        read_label_file(os.path.join(self.raw_folder, "train-labels-idx1-ubyte")),
    )
    test_set = (
        read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
        read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")),
    )
    with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
        torch.save(training_set, f)
    with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
        torch.save(test_set, f)

    print("Done!")


MNIST.download = patched_download
##########################################################################


# Define my model
# model = nn.Sequential(
#     nn.Linear(28 * 28, 64),
#     nn.ReLU(),
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Dropout(0.1) # if we're overfitting
#     nn.Linear(64, 10)
# )

# Define a more flexible model(add residual connection)
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits


model = ResNet().cuda()

# Define my optimizer
params = model.parameters()
optimizer = optim.SGD(params, lr=1e-2)

# Define my loss
loss = nn.CrossEntropyLoss()

# train, val split
train_data = datasets.MNIST(
    "data", train=True, download=True, transform=transforms.ToTensor()
)
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=10)
val_loader = DataLoader(val, batch_size=10)

# My training and validation loops
nb_epochs = 10
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    print(f"Epoch {epoch +1}")
    model.train()  # effect only certain modules eg) Let Dropout actually drop neurons during training
    for batch in train_loader:
        x, y = batch  # y : label
        x = x.cuda()
        y = y.cuda()

        # x : b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        l = model(x)  # l : logits

        # 2 compute the objective function
        J = loss(l, y)

        # 3 cleaning the gradients
        model.zero_grad()
        # optimizer.zero_grad()
        # params.grad._zero()

        # 4 accumulate the partial derivatives of J wrt(with respect to) params
        J.backward()
        # params.grad._sum(dJ/dparams)

        # 5 step in the opposite direction of the gradient
        optimizer.step()
        # with torch.no_grad(): params = params - eta * params.grad

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f"\t train loss: {torch.tensor(losses).mean():.2f}")
    print(f"\t train accuracy: {torch.tensor(accuracies).mean():.2f}")

    losses = list()
    accuracies = list()
    model.eval()  # effect only certain modules eg) Let Dropout don't drop neurons for evaluation
    for batch in val_loader:
        x, y = batch  # y : label
        x = x.cuda()
        y = y.cuda()

        # x : b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        with torch.no_grad():
            l = model(x)  # l : logits

        # 2 compute the objective function
        J = loss(l, y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f"\t validataion loss: {torch.tensor(losses).mean():.2f}")
    print(f"\t validataion accuracy: {torch.tensor(accuracies).mean():.2f}")
