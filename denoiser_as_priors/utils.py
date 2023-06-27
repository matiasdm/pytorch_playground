"""
util functions
"""


import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = "tight"


# Define tools to visualize examples.
@torch.no_grad()
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


@torch.no_grad()
def plot_original_noisy_denoised_batches(o, n, d):
    # normalize images to range [0,1]
    def normlize_tensor(t):
        t = t - torch.min(t)
        t = t / torch.max(t)
        return t
    o, n, d = normlize_tensor(o), normlize_tensor(n), normlize_tensor(d)
    images = torch.cat((o, n, d), dim=3)
    images = images.cpu()
    fig = show_batch(images)
    return fig


def add_gaussian_noise(img, mean=0, std=None, stocastic=True):
    "If the noise std is not specified, is stocastic = True (default) it is set to be a random number between 0 and .4, otherwise is set to .1"
    if std is None:
        std = np.random.uniform(0, 0.4) if stocastic else 0.1
    noise = torch.randn(img.size()) * std + mean
    noisy_img = img + noise

    dic = {"noise": noise, "noisy_img": noisy_img, "a": 1}
    return noisy_img


def show_batch(batch_of_images):
    imgs = [img for img in batch_of_images]
    if not isinstance(imgs, list):
        imgs = [imgs]

    ncols = int(np.ceil(np.sqrt(len(imgs))))
    nrows = ncols
    fig, axs = plt.subplots(ncols=ncols, nrows=ncols, squeeze=False)
    fig.set_size_inches(10, 10)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i // ncols, np.mod(i, ncols)].imshow(np.asarray(img).squeeze(), cmap="gray")
        axs[i // ncols, np.mod(i, ncols)].set(
            xticklabels=[], yticklabels=[], xticks=[], yticks=[]
        )
    return fig


def img_nomralize(img):
    img = img - img.min()
    img = img / img.max()
    return img


def batch_img_normalize(batch_of_imgs):
    for i in range(len(batch_of_imgs)):
        batch_of_imgs[i] = img_nomralize(batch_of_imgs[i])
    return batch_of_imgs
