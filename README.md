# Pytorch playground

A place to play around with pytorch and random AI experiments. 

## Instalation 

### Requirements
 - pyenv: see https://github.com/pyenv/pyenv#installation 
 
### Using pyenv 
Fix the python version you'll use and you can install it using `pyenv install 3.X.X`. Then you can type `pyenv local 3.X.X` to create a `.python-version` file and fix that version for everyone else.

If another person downloads the repo (having pyenv installed) and run `python --version`, that person should see something like:

```bash
$ python --version
pyenv: version `3.X.X' is not installed (set by /home/.../repo-name/.python-version)
```

Then, you can simply install the required version by running `pyenv install` or `pyenv install <required version>`.

When you have your python version set up correctly, you can create your virtual environment:

```bash
$ python --version  # check version matches the required version
$ python -m venv .venv
```

And that creates the virtualenv!

### Now you can install the remaing python packages 
```bash
$pip install -r requirements.txt
```

## To install from scratch:
From: https://pytorch.org/get-started/pytorch-2.0/#requirements
```bash
$ pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
```

