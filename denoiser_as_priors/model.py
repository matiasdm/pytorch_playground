"""
Define the denoising model and model utils 
"""

import torch 
from torch import nn


class bfcnn(nn.Module):
    "Bias-Free Convolutional Neural Network block"
    def __init__(self, chin=None, chout=32, kernel_size=3):
        super().__init__()
        self.bn = nn.BatchNorm2d(chin, affine=False)
        self.conv = nn.Conv2d(chin, chout, kernel_size, padding='same', bias=False) 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
        

class Denoiser(nn.Module):
    "Bias free CNN denoiser, with 'depth' number of bfcnn blocks in"
    def __init__(self, depth=10):
        super().__init__()
        self.bfcnn_first = bfcnn(chin=1, chout=32, kernel_size=3)
        self.dfcnn_middle = nn.ModuleList(
            [bfcnn(chin=32, chout=32, kernel_size=3) for i in range(depth)]
            )
        self.bfcnn_last = bfcnn(chin=32, chout=1, kernel_size=3)
        self.dropout = nn.Dropout2d(p=0.5)
        self.depth = depth
        
    def forward(self, x):
        x = self.bfcnn_first(x)  # first bfccn block 1x28x28 -> 32x28x28
        for i in range(self.depth):
            x = self.dfcnn_middle[i](x)  # hidden bfcnn blocks 32x28x28 -> 32x28x28
        x = self.dropout(x)  ## dropout layer to prevent overfitting
        x = self.bfcnn_last(x)  # last bfcnn block 32x28x28 -> 1x28x28
        return x
    
    
def init_model(depth=10):
    """
    model, loss_fn, optimizer = init_model(depth=10)
    ----------------
    Define model architecture and initialize model parameters.
    returns model, loss_fn, optimizer
    """
    model = Denoiser(depth=depth)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer 


def save_model(model=None, path=None):
    """
    Save model parameters
    """
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")
    return 


def load_model(path=None):
    """
    Load model parameters
    """
    model, _, _ = init_model()  # initialize model
    model.load_state_dict(torch.load(path))  # load model parameters
    return model


if __name__ == "__main__":
    # Sanity check
    # Create a new model, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, loss_fn, optimizer = init_model()
    from torchsummary import summary
    summary(model.to(device), (1, 28, 28))
    print('[ok] Model initialized')
    
    # Save a model
    save_model(model=model, path="model.pth")
    print('[ok] Model saved')
    
    # Load a model
    load_model(path="model.pth")
    print('[ok] Model loaded')
    import os
    os.remove('model.pth')  # remove the test model created in this script
    