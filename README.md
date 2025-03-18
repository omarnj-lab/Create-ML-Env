# Setting Up Your Deep Learning Environment
## A Step-by-Step Guide for Windows and macOS

This guide will walk you through setting up a complete development environment for deep learning, including VS Code, Anaconda, CUDA (for GPU support), PyTorch, and other essential libraries.

## Table of Contents
1. [Installing Anaconda](#1-installing-anaconda)
2. [Installing Visual Studio Code](#2-installing-visual-studio-code)
3. [Opening VS Code from Anaconda](#3-opening-vs-code-from-anaconda)
4. [Creating a New Conda Environment](#4-creating-a-new-conda-environment)
5. [Installing CUDA and Checking GPU Availability](#5-installing-cuda-and-checking-gpu-availability)
6. [Installing PyTorch and Essential Libraries](#6-installing-pytorch-and-essential-libraries)
7. [Setting Up Jupyter Notebook](#7-setting-up-jupyter-notebook)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Installing Anaconda

### For Windows
1. Visit [Anaconda's download page](https://www.anaconda.com/products/distribution)
2. Download the Windows installer (64-bit recommended)
3. Run the downloaded `.exe` file
4. During installation:
   - Select "Install for Just Me"
   - Choose the default installation location
   - **Important**: Check "Add Anaconda to my PATH environment variable"
   - Check "Register Anaconda as my default Python"
5. Click "Install" and wait for the installation to complete

### For macOS
1. Visit [Anaconda's download page](https://www.anaconda.com/products/distribution)
2. Download the macOS installer (64-bit recommended)
3. Run the downloaded `.pkg` file
4. Follow the installation prompts
5. During installation, select "Install for me only"
6. Complete the installation

### Verify Anaconda Installation
Open a terminal (Command Prompt on Windows or Terminal on macOS) and type:
```bash
conda --version
```
You should see the version of conda that was installed.

---

## 2. Installing Visual Studio Code

### For Windows
1. Visit the [VS Code download page](https://code.visualstudio.com/download)
2. Click on the Windows download button
3. Run the downloaded `.exe` file
4. Accept the license agreement
5. Choose the installation location
6. In the "Select Additional Tasks" screen:
   - Check all boxes, especially "Add to PATH"
7. Click "Install" and wait for the installation to complete

### For macOS
1. Visit the [VS Code download page](https://code.visualstudio.com/download)
2. Click on the macOS download button
3. Open the downloaded `.zip` file
4. Drag Visual Studio Code to the Applications folder
5. Open VS Code from your Applications folder

### Install Essential VS Code Extensions
1. Open VS Code
2. Press `Ctrl+Shift+X` (Windows) or `Cmd+Shift+X` (macOS) to open Extensions view
3. Search for and install these extensions:
   - Python
   - Jupyter
   - Pylance
   - IntelliCode

---

## 3. Opening VS Code from Anaconda

### From Anaconda Navigator
1. Open Anaconda Navigator (search for it in your start menu or applications)
2. In the Home tab, find VS Code in the list of applications
3. Click "Launch"

### From Command Line
Alternatively, you can open VS Code from the command line:

1. Open Terminal (macOS) or Command Prompt/Anaconda Prompt (Windows)
2. Type the following command:
```bash
code
```

---

## 4. Creating a New Conda Environment

It's best practice to create a separate environment for each project to avoid dependency conflicts.

### Using Command Line (Recommended)
1. Open Terminal (macOS) or Anaconda Prompt (Windows)
2. Create a new environment with Python 3.10:
```bash
conda create -n deeplearning python=3.10
```
3. Activate the environment:
```bash
# On Windows
conda activate deeplearning

# On macOS/Linux
conda activate deeplearning
```

### Using Anaconda Navigator
1. Open Anaconda Navigator
2. Click on the "Environments" tab
3. Click "Create" at the bottom
4. Name your environment (e.g., "deeplearning")
5. Select Python 3.10
6. Click "Create"

---

## 5. Installing CUDA and Checking GPU Availability

### For NVIDIA GPU Users

#### Windows
1. Check your GPU model and ensure it's CUDA-compatible
2. Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. Select Windows, your version, and download type (typically exe local)
4. Download and run the installer
5. Choose "Express Installation"
6. Restart your computer after installation

#### macOS
Note: Most newer Macs with M1/M2 chips don't use NVIDIA GPUs and rely on Metal instead of CUDA.
If you have an older Mac with an NVIDIA GPU:

1. Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select macOS and follow the installation instructions
3. Restart your computer after installation

### Checking GPU Availability

1. Activate your conda environment:
```bash
conda activate deeplearning
```

2. Install PyTorch (temporary installation for testing):
```bash
# For Windows with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For macOS with M1/M2
conda install pytorch torchvision torchaudio -c pytorch
```

3. Test GPU availability in Python:
```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If you see "GPU Available: True", your GPU is correctly set up!

---

## 6. Installing PyTorch and Essential Libraries

1. Ensure your conda environment is activated:
```bash
conda activate deeplearning
```

2. Install PyTorch with GPU support (adjust CUDA version if needed):
```bash
# For Windows with NVIDIA GPU (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For macOS with M1/M2
conda install pytorch torchvision torchaudio -c pytorch

# For CPU only (no GPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

3. Install essential data science and machine learning libraries:
```bash
conda install -c conda-forge numpy pandas matplotlib scikit-learn seaborn

# Deep learning specific libraries
pip install transformers datasets huggingface_hub tensorboard
```

4. Install additional useful libraries:
```bash
# General utilities
pip install tqdm pillow opencv-python

# For NLP
pip install nltk spacy gensim

# For visualization 
pip install plotly

# For experiment tracking
pip install mlflow wandb
```

---

## 7. Setting Up Jupyter Notebook

### Installing Jupyter
1. Ensure your conda environment is activated:
```bash
conda activate deeplearning
```

2. Install Jupyter:
```bash
conda install -c conda-forge jupyterlab notebook
```

### Starting Jupyter Notebook
1. From the terminal with your environment activated:
```bash
jupyter notebook
```
This will open Jupyter in your default web browser.

### Using Jupyter within VS Code
1. Open VS Code
2. Open a folder for your project
3. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS)
4. Type "Python: Select Interpreter" and select your `deeplearning` environment
5. Create a new notebook:
   - Click on the Explorer icon in the side bar
   - Right-click in the Explorer pane
   - Select "New File" and name it with `.ipynb` extension
6. VS Code will open the notebook with Jupyter integration

### Test Your Environment
Create a new notebook and run the following code to test your setup:

```python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import transformers

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## 8. Troubleshooting

### Common Issues and Solutions

#### "conda command not found"
- **Windows**: Reinstall Anaconda and make sure to check "Add Anaconda to my PATH"
- **macOS**: Add conda to your path by running:
  ```bash
  export PATH=~/anaconda3/bin:$PATH
  # Add this line to your ~/.zshrc or ~/.bash_profile to make it permanent
  ```

#### CUDA installation issues
- Make sure your GPU is CUDA-compatible
- Install the appropriate CUDA version for your PyTorch version
- Check NVIDIA driver installation

#### "ImportError: No module named X"
- Make sure your conda environment is activated
- Install the missing package: `pip install X` or `conda install X`

#### Jupyter not showing the right environment
- Install ipykernel in your environment:
  ```bash
  conda install ipykernel
  python -m ipykernel install --user --name=deeplearning
  ```

#### VS Code not finding the Python interpreter
- Manually specify the interpreter path:
  - Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS)
  - Type "Python: Select Interpreter"
  - Click "Enter interpreter path"
  - Navigate to your Anaconda installation and select the Python executable in your environment

### Getting Help
- Check the documentation:
  - [Anaconda Documentation](https://docs.anaconda.com/)
  - [VS Code Documentation](https://code.visualstudio.com/docs)
  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Search for error messages on Stack Overflow
- Join relevant communities on Discord or Reddit

---

Congratulations! You now have a fully set up deep learning environment with VS Code, Anaconda, PyTorch, and all the essential libraries. Happy coding!
