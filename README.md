# LF-PINE
Physics-informed explicit modelling for high-resolution reconstruction of light-field microscopy
# Environments
## Recommended System Configuration
* a NVIDIA-A100-SXM4 / NVIDIA-RTX3090 GPU or better
* Windows 10 or 11 / Linux Ubuntu 16 or higher version operating system
* 32GB RAM or more
* 512GB disk space or more
* 2pSAM captured data or Scanning Light Field Microscopy captured data
## Tested System Configuration
* a NVIDIA-A100-SXM4 GPU and NVIDIA-RTX3090 GPU
* Linux Ubuntu 22.04.5 LTS
* 128GB RAM
* 16TB disk space
* 2pSAM captured data & Scanning Light Field Microscopy captured data
## Preparation
### Download LF-PINE source code
Download our code using
```
cd ~
git clone https://github.com/Joey1998hub/LF-PINE.git
```
The download time is usually within 10 minutes (depending on your network speed).

### Create LF-PINE anaconda environment
```
conda create -n LF-PINE python=3.9
conda activate LF-PINE
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tifffile tensorboard numpy
```
Our repo is built mainly using PyTorch, so installing torch has higher priority. You can refer to the [PyTorch](https://pytorch.org/) official guide to install torch based on your machine and driver version.

## Using LF-PINE
### Reconstruction Directly:
* Place the input LFs (Light Fields) folder (containing TIFF format files) into `./Projections/` and the PSFs (Point Spread Functions) folder (containing TIFF format files) into `./PSF/` .
* Set the file paths of LFs and PSFs in the `reconLF.py` file according to your local directory structure.
* Run `reconLF.py` to perform reconstruction.
* Reconstruction results will be saved to `./Results/`.

#### Demo Results
A simulated light field image of Fly Neurons based on the 2pSAM system, with an input resolution of 512×512×13 (X×Y×Angle, pixel) and an output resolution of 512×512×39 (X×Y×Z, pixel). The training and reconstruction process is expected to take 80 seconds on a computing platform equipped with the NVIDIA-A100-SXM4 GPU.

### Reconstruction with Digital Adaptive Optics (DAO):
* Download the required Zernike basis and PSFs in phase space via the link: https://drive.google.com/drive/folders/1BUlpoeEX6Y9qSb2VCT8l-owbe4nqWXAp?usp=sharing
* Place phase space PSFs folder (`/PSFEx_zoom4_31dz1_N13`) and Zernike basis folder (`/Phase`) into `./PSF/` .
* Place the input LFs (Light Fields) folder (containing TIFF format files) into `./Projections/`.
* Set the file paths of LFs and PSFs in the `reconLF_DAO.py` file according to your local directory structure.
* Run `reconLF_DAO.py` to perform reconstruction.
* If you want to perform reconstruction without DAO, set `polynum = 0`.
* Reconstruction results will be saved to `./Results/`.

#### Demo Results
A simulated light field image of USAF with aberration introduced (RMS=1.0λ) based on the 2pSAM system, with an input resolution of 512×512×13 (X×Y×Angle, pixel) and an output resolution of 512×512×31 (X×Y×Z, pixel).
