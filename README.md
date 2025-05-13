
# Mesh Reconstruction from 3D Gaussian Splatting

This repository contains the implementation of **Mesh Reconstruction from 3D Gaussian Splatting**, a master's thesis project that reconstructs explicit 3D mesh surfaces from 3D Gaussians by combining:
- **GaussianShader**'s physically-based appearance modeling
- **RaDe-GS**'s differentiable rasterization and geometric regularization

The method uses a hybrid 3D Gaussian representation as a surface prior and optimizes it for photo-consistency, normal fidelity, and mesh extractability.

---

## 🧠 Project Summary

Introduces a differentiable pipeline for:
1. Optimizing Gaussian parameters (position, scale, rotation, BRDF appearance, opacity) from multi-view images.
2. Rendering views using closed-form splatting and physically-based shading.
3. Extracting high-quality triangle meshes by projecting optimized Gaussians into a tetrahedral field and running Marching Tetrahedra.

---

## 📦 Installation Guide

Tested on **Ubuntu 20.04+, CUDA 11.8, Python 3.8, PyTorch 2.1+**

### 1. Setup Conda environment


conda create -y -n meshfrom3dgs python=3.8
conda activate glare

### 2. Setup Conda environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

cd submodules/tetra_triangulation
conda install -y cmake
conda install -y -c conda-forge gmp cgal

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
      -DCMAKE_CXX_FLAGS="-I/usr/local/cuda-11.8/include" .
make
pip install -e .
cd ../..

git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
sudo apt-get install -y build-essential cmake
pip install .
cd ..
