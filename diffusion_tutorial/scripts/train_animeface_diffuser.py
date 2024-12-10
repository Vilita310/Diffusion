import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "E:/data/huggan-anime-faces"

def download_data():
    if os.path.isdir(DATA_PATH):
        print(f"Data folder already exist at {DATA_PATH}. Skip downloading...")
        return

    os.mkdir(DATA_PATH)
    dataset = load_dataset("huggan/anime-faces", split="train", verification_mode="no_checks")
    print(f"Downloading data to {DATA_PATH}...")
    for i, d in tqdm(enumerate(dataset)):
        d["image"].save(os.path.join(DATA_PATH, f"{i}.jpg"), format="jpeg")
    print(f"Data downloaded.")
    return


class AnimeFaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_names = os.listdir(data_path)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_names[idx])
        image = Image.open(image_path, formats=["jpeg"])
        return self.transform(image)


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.Conv2d(128, 128, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(128, 128, kernel_size=5, padding=2),
                nn.Conv2d(128, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 3:  # For all but the final down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


def train(batch_size=128, n_epochs=3, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloader (you can mess with batch size)
    dataset = AnimeFaceDataset(data_path=DATA_PATH)
    train_dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    # Create the network
    net = BasicUNet()
    net.to(device)

    # Our loss function
    loss_fn = nn.MSELoss()

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # The training loop
    for epoch in range(n_epochs):
        losses = []
        for x in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
            noisy_x = corrupt(x, noise_amount)  # Create our noisy x

            # Get the model prediction
            pred = net(noisy_x)

            # Calculate the loss
            loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print our the average of the loss values for this epoch:
        avg_loss = sum(losses) / len(train_dataloader)
        print(f"Finished epoch {epoch}/{n_epochs}. Average loss for this epoch: {avg_loss:05f}")

    print("Saving model...")
    torch.save(net, "animeface.pth")
    return net


def infer(net, n_steps=5):
    # @markdown Sampling strategy: Break the process into 5 steps and move 1/5'th of the way there each time:
    x = torch.rand(1, 3, 64, 64).to(device)  # Start from random

    for i in tqdm(range(n_steps)):
        with torch.no_grad():  # No need to track gradients during inference
            pred = net(x)  # Predict the denoised x0

        mix_factor = 1 / (n_steps - i)  # How much we move towards the prediction
        x = x * (1 - mix_factor) + pred * mix_factor  # Move part of the way there
        img = x[0].cpu().clip(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        cv2.imshow("", img)
        cv2.waitKey()


def main():
    download_data()
    net = train(n_epochs=40)
    infer(net, n_steps=50)


if __name__ == "__main__":
    main()
