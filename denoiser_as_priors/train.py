"""
Train denoisers model for NMIST toy example. 
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils import add_gaussian_noise, plot_original_noisy_denoised_batches, matplotlib_imshow
from model import save_model
import numpy as np
import os


def train(
    logdir=None,
    epochs=10,
    model=None,
    train_dataloader=None,
    val_dataloader=None,
    optimizer=None,
    loss_fn=None,
    device=None,
):
    """
    model = train(logdir, epochs, device, model, train_dataloader, val_dataloader, optimizer, loss_fn, device)
    Define training loop and train the model
    """
    # Check non-optimal inputs are provided
    if None in [model, train_dataloader, optimizer, loss_fn, device]:
        print(
            "Error: missing input. Please provide model, train_dataloader, optimizer, loss_fn and device."
        )

    # Initialize Tensorboard
    writer = SummaryWriter(log_dir=logdir)
    images, _ = next(iter(train_dataloader))
    writer.add_graph(model, images.to(device))  # Add the model to Tensorboard

    for t in range(epochs):
        size = len(train_dataloader.dataset)
        model.train()  # Sets the model in training mode
        for batch, (X, _) in enumerate(train_dataloader):  # Iterate over batches
            noisy_X = add_gaussian_noise(X)  # noisy sample,
            X, noisy_X = X.to(device), noisy_X.to(device)
            pred = model(noisy_X)  # denoised sample
            pred = pred.view(X.shape)  # reshape from 16x784 to 16x1x28x28
            loss = loss_fn(pred, X)  # MSE loss between denoised and original

            # Backpropagation
            optimizer.zero_grad()  # Set gradients to zero
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            if batch % 50 == 0:  # Record loss every 50 batches
                loss, current = loss.item(), batch * len(X)
                if batch % 250 == 0:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # ... log the running loss
                writer.add_scalar(
                    "training loss", loss, t * len(train_dataloader) + batch
                )
                writer.add_figure(
                    "original, noisy and denoised",
                    plot_original_noisy_denoised_batches(X, noisy_X, pred),
                    global_step=t * len(train_dataloader) + batch,
                )

        if val_dataloader is not None:  # add the test performance to the writer
            val_loss = test(
                dataloader=val_dataloader, model=model, loss_fn=loss_fn, device=device
            )
            writer.add_scalar(
                "validation loss", val_loss, t * len(train_dataloader) + batch
            )
    # Save the parameters of the trained model 
    save_model(model, os.path.join(logdir, "model.pt"))
    return model


@torch.no_grad()
def test(model=None, dataloader=None, loss_fn=None, device=None):
    """
    test_loss = test(model, test_dataloader, loss_fn, device)
    Test a trained model on test dataset.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Set the model in evaluation mode
    test_loss = 0  # init.
    for X, _ in dataloader:
        noisy_X = add_gaussian_noise(X)
        X, noisy_X = X.to(device), noisy_X.to(device)
        pred = model(noisy_X)
        pred = pred.view(X.shape)  # reshape from 16x784 to 16x1x28x28
        test_loss += loss_fn(pred, X).item()
    test_loss /= num_batches  # Mean MSE loss
    return test_loss


if __name__ == "__main__":
    # Sanity check
    from data import load_data
    from model import init_model

    # Load data
    print("Loading data...")
    train_dataloader, test_dataloader = load_data(batch_size=64)
    # Initialize model
    print("Initializing model...")
    model, loss_fn, optimizer = init_model(depth=10)

    # Train model
    print("Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on device {device}")
    model.to(device)
    model = train(
        logdir="runs/test",
        epochs=2,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    print("Done!")
