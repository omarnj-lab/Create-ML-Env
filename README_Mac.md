# **Complete Guide to GPU Acceleration on Mac (M1, M2, M3 & Intel Macs)**  

Mac laptops, especially Apple Silicon (M1, M2, and M3), have powerful GPUs, but they do **not** support CUDA (which is exclusive to NVIDIA). Instead, they use **Metal Performance Shaders (MPS)** for GPU acceleration.

## **🔹 1. Checking Your Mac GPU Compatibility**
First, check your Mac model and whether it supports **Metal (Apple’s GPU API)**.

### **Run this in Terminal:**  
```bash
system_profiler SPHardwareDataType | grep "Chip"
```
- **Apple Silicon (M1, M2, M3)** → Supports **MPS** (Metal Performance Shaders)
- **Intel Macs** → Limited support for Metal, recommended for CPU workloads

### **Check GPU support in Python:**  
```python
import torch

if torch.backends.mps.is_available():
    print("✅ MPS (Metal GPU) is available!")
else:
    print("❌ MPS is not supported.")
```

---

## **🔹 2. Installing GPU-Accelerated Libraries on Mac**
### **Install PyTorch with Metal (for deep learning)**
```bash
pip install torch torchvision torchaudio
```

### **Install JAX (for high-performance computing)**
```bash
pip install jax jaxlib
```
🔹 JAX **automatically** detects the Metal backend on Mac.

### **Install TensorFlow with MPS (for AI/ML)**
Apple provides a Metal-optimized TensorFlow package:
```bash
pip install tensorflow-macos tensorflow-metal
```

### **Install Dask (for parallel computing)**
```bash
pip install dask
```

---

## **🔹 3. Using PyTorch with Metal on Mac**
### **Example: Matrix Multiplication on Mac GPU**
```python
import torch

# Select Metal GPU
device = torch.device("mps")

# Create large tensors on the GPU
a = torch.randn((1000, 1000), device=device)
b = torch.randn((1000, 1000), device=device)

# Perform matrix multiplication on GPU
c = torch.matmul(a, b)

print("✅ Computation done on:", c.device)
```

---

## **🔹 4. Running TensorFlow on Apple Silicon (M1/M2/M3)**
### **Example: Simple Neural Network Training on GPU**
```python
import tensorflow as tf

# Verify GPU usage
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Create and train a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create dummy data
import numpy as np
x_train = np.random.rand(1000, 32)
y_train = np.random.randint(0, 10, 1000)

# Train on Metal GPU
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

---

## **🔹 5. Using JAX for High-Performance Computing**
### **Example: GPU-Accelerated NumPy on Mac**
```python
import jax.numpy as jnp
from jax import random

# Generate random arrays on GPU
key = random.PRNGKey(0)
a = random.normal(key, (1000, 1000))
b = random.normal(key, (1000, 1000))

# Perform matrix multiplication on GPU
c = jnp.dot(a, b)

print("✅ Computation completed on Metal GPU")
```

---

## **🔹 6. Running Parallel Workloads on Mac with Dask**
### **Example: Parallel Computing with Dask**
```python
import dask.array as da

# Create a large array using Dask
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# Compute the mean using parallel processing
result = x.mean().compute()

print("✅ Parallel computation done using Dask:", result)
```

---

## **🔹 7. Benchmarking Mac GPU Performance**
Want to test your Mac’s GPU speed? Try this!

### **PyTorch Benchmarking**
```python
import torch
import time

device = torch.device("mps")

# Create large tensors on Metal GPU
a = torch.randn((5000, 5000), device=device)
b = torch.randn((5000, 5000), device=device)

# Time the matrix multiplication
start_time = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # Ensure all operations finish
end_time = time.time()

print(f"✅ Metal GPU Time: {end_time - start_time:.4f} seconds")
```

### **TensorFlow Benchmarking**
```python
import tensorflow as tf
import time

# Generate random tensors
a = tf.random.normal((5000, 5000))
b = tf.random.normal((5000, 5000))

# Time matrix multiplication
start_time = time.time()
c = tf.matmul(a, b)
end_time = time.time()

print(f"✅ TensorFlow (Metal GPU) Time: {end_time - start_time:.4f} seconds")
```

---

## **🔹 8. Cloud GPU Options for Mac Users**
If you need **more powerful GPUs (e.g., NVIDIA CUDA)**, use cloud services:
- **Google Colab** (Free Tesla T4 GPU)
- **Paperspace Gradient** (Affordable A100 GPUs)
- **Lambda Labs** (High-performance GPU instances)

Example: Running a cloud GPU on **Google Colab**
```python
!nvidia-smi  # Check available NVIDIA GPUs
```

---

## **🔹 9. Summary Table: Best Tools for Mac GPU**
| **Task** | **Best Tool** | **Mac Support** |
|----------|--------------|----------------|
| Deep Learning | PyTorch (MPS) | ✅ Metal |
| Machine Learning | TensorFlow (MPS) | ✅ Metal |
| NumPy-like Computation | JAX | ✅ Metal |
| Parallel Processing | Dask | ✅ CPU |
| Cloud GPU | Google Colab | ✅ NVIDIA |

---

## **🚀 Conclusion: What Can You Do Next?**
🔹 **For Deep Learning** → Use PyTorch or TensorFlow with Metal  
🔹 **For High-Performance Computing** → Use JAX with Metal  
🔹 **For Parallel Computing** → Use Dask (CPU-based)  
🔹 **For NVIDIA GPUs** → Use Google Colab or cloud services  

