# **Complete Guide to GPU Acceleration on AMD & Intel (Non-NVIDIA Users)**

If you have an **AMD** or **Intel** GPU and want to accelerate your computations without using NVIDIA CUDA, this guide will help you set up the right tools.

## **🔹 1. Understanding GPU Acceleration for AMD & Intel**
Unlike NVIDIA, which uses **CUDA**, AMD and Intel GPUs rely on:
- **AMD:** ROCm (Radeon Open Compute)
- **Intel:** oneAPI (SYCL)
- **Cross-Platform:** OpenCL & Vulkan

### **Check Your GPU Type**
Run this command in Terminal:
```bash
lspci | grep VGA  # For Linux users
dxdiag  # For Windows users (Run in Command Prompt)
```
If you see **AMD** or **Intel** listed, follow the corresponding setup steps below.

---

## **🔹 2. Installing GPU-Accelerated Libraries**

### **For AMD GPUs (ROCm)**
#### ✅ **Install ROCm (Radeon Open Compute) Framework**
```bash
wget https://repo.radeon.com/amdgpu-install/23.10.1/ubuntu/focal/amdgpu-install_23.10.1.50200-1_all.deb
sudo dpkg -i amdgpu-install_23.10.1.50200-1_all.deb
sudo amdgpu-install --usecase=rocm
```
🔹 **Verify Installation:**
```bash
rocminfo  # Check if ROCm is installed
hipcc --version  # Check HIP compiler
```

#### ✅ **Install PyTorch with ROCm**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4
```

#### ✅ **Run PyTorch on AMD GPU**
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```
---

### **For Intel GPUs (oneAPI + SYCL)**
#### ✅ **Install Intel oneAPI (for SYCL Support)**
```bash
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19121/l_BaseKit_p_2022.1.2.194_offline.sh
sh l_BaseKit_p_2022.1.2.194_offline.sh
```
🔹 **Enable oneAPI in Your Environment:**
```bash
source /opt/intel/oneapi/setvars.sh
```

#### ✅ **Install PyTorch for Intel GPUs**
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/oneapi
```

#### ✅ **Run PyTorch on Intel GPU**
```python
import torch

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print("Using device:", device)
```

---

## **🔹 3. Using JAX for High-Performance Computing (Works on AMD & Intel)**
JAX is a great alternative to NumPy that supports **both AMD (ROCm)** and **Intel (oneAPI)**.

### ✅ **Install JAX**
```bash
pip install jax jaxlib
```

### ✅ **Run JAX on AMD or Intel GPU**
```python
import jax.numpy as jnp

# Create a GPU-accelerated array
a = jnp.array([1, 2, 3, 4, 5])
b = jnp.array([10, 20, 30, 40, 50])

# Perform computation
c = a * b
print("Result:", c)
```
---

## **🔹 4. Using Dask for Multi-Core and Parallel Computing (CPU & GPU)**
If your GPU isn’t well-supported, you can still speed up computations using **Dask (multi-core CPU acceleration).**

### ✅ **Install Dask**
```bash
pip install dask
```

### ✅ **Parallel Processing with Dask**
```python
import dask.array as da

# Create a large parallelized array
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# Compute mean in parallel
result = x.mean().compute()
print("Mean:", result)
```

---

## **🔹 5. Comparing Performance on AMD & Intel GPUs**
Want to test your GPU speed? Try this!

### ✅ **Benchmark PyTorch on AMD GPU (ROCm)**
```python
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.randn((5000, 5000), device=device)
b = torch.randn((5000, 5000), device=device)

start_time = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
end_time = time.time()

print(f"✅ ROCm GPU Time: {end_time - start_time:.4f} seconds")
```

### ✅ **Benchmark PyTorch on Intel GPU (oneAPI)**
```python
import torch
import time

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

a = torch.randn((5000, 5000), device=device)
b = torch.randn((5000, 5000), device=device)

start_time = time.time()
c = torch.matmul(a, b)
end_time = time.time()

print(f"✅ OneAPI GPU Time: {end_time - start_time:.4f} seconds")
```
---

## **🔹 6. Summary Table: Best Tools for AMD & Intel GPU Users**
| **Task** | **Best Tool** | **AMD (ROCm)** | **Intel (oneAPI)** | **Cross-Platform** |
|----------|--------------|---------------|----------------|----------------|
| Deep Learning | PyTorch | ✅ | ✅ | ❌ |
| Machine Learning | TensorFlow | ✅ | ❌ | ❌ |
| NumPy-like Computation | JAX | ✅ | ✅ | ✅ |
| Parallel Processing | Dask | ✅ (CPU) | ✅ (CPU) | ✅ |
| Cloud GPU | Google Colab | ❌ (NVIDIA only) | ❌ (NVIDIA only) | ✅ |

---

## **🚀 Conclusion: What Can You Do Next?**
✅ **For Deep Learning on AMD** → Use PyTorch with ROCm  
✅ **For Deep Learning on Intel** → Use PyTorch with oneAPI  
✅ **For High-Performance Computing** → Use JAX with ROCm/oneAPI  
✅ **For Parallel Processing** → Use Dask (CPU-based acceleration)  
✅ **For Cloud GPUs (NVIDIA only)** → Use Google Colab or Paperspace  

Would you like more **advanced projects** or specific code examples? Let me know! 🚀
