import os

import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import DDPMScheduler, UNet2DModel

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


def train(batch_size=128, n_epochs=3, lr=1e-3):
    # Dataloader (you can mess with batch size)
    dataset = AnimeFaceDataset(data_path=DATA_PATH)
    train_dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Create the network
    net = UNet2DModel(
        sample_size=64,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
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
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get the model prediction
            pred = net(noisy_x, timesteps).sample

            # Calculate the loss
            loss = loss_fn(pred, noise)  # How close is the output to the true 'clean' x?

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
    torch.save(net, "mnist_ddpm.pth")
    return net


def infer(net):
    # @markdown Sampling strategy: Break the process into 5 steps and move 1/5'th of the way there each time:
    x = torch.rand(1, 3, 64, 64).to(device)  # Start from random

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(num_inference_steps=40)

    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():  # No need to track gradients during inference
            residual = net(x, t).sample  # Predict the denoised x0

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

        img = x[0].cpu().clip(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("", img)
        cv2.waitKey()


def main():
    if not os.path.isfile("animeface_ddpm.pth"):
        net = train(n_epochs=10)
    else:
        net = torch.load("animeface_ddpm.pth", map_location=device)
    infer(net)


if __name__ == "__main__":
    main()
