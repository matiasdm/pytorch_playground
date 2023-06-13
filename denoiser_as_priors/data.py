"""
Define trainign and testing data
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data(logger=None, batch_size=64):
    """
    Load and prepare NMINST dataset for training and evaluating. logger is a logging object, batch_size is the batch size for training and testing (default: 64).
    """
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    if logger is not None:  # Add logging information
        logger.info("[load_data]: Data loaders created")
        logger.info(f"[load_data]: Training data size: {len(training_data)}")
        logger.info(f"[load_data]: Training data size: {len(training_data)}")
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    # Sanity check
    print('Sanity check:: Loading data ... ')
    train, test = load_data()
    for X, y in test:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
    print('Data loaded successfully!')

