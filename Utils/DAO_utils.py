from torch.fft import ifft2,fftshift
import torch;import torch.nn.functional as F
import os;import sys
sys.path.append(os.getcwd())
import Utils.utils as ut
import tifffile as tf
import numpy as np

def genCircle(matrix_size,radius):
    matrix_shape = (matrix_size, matrix_size)  

    x, y = torch.meshgrid(torch.arange(matrix_shape[0]), torch.arange(matrix_shape[1]))

    center_x, center_y = matrix_shape[0] // 2, matrix_shape[1] // 2

    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

    circle_matrix = torch.where(distance <= radius, torch.tensor(1), torch.tensor(0)).float()

    circle_matrix.requires_grad = False

    return circle_matrix

def calPupilSize(zoomFactor=8):

    FOV_zoom8_um = 86
    during_dxy = 50

    FOV = FOV_zoom8_um*1e3*8/zoomFactor
    pupil_xysize = round(FOV/during_dxy)

    return pupil_xysize

def generate_psf_ab(psf_fft,fplane,pupilsize,psfsize,amp=1e8):
    # same as DAOSLIMIT

    fplane = torch.exp(fplane)

    padding = int((pupilsize-psf_fft.shape[1])/2)
    psf_fft_ab = F.pad(psf_fft*fplane,(padding,padding,padding,padding))

    psf_ab = ifft2(psf_fft_ab)
    psf_ab = torch.pow(torch.abs(psf_ab),2)
    psf_ab = torch.pow(torch.abs(psf_ab),2)
    psf_ab = psf_ab*amp
    # psf_ab = psf_ab/torch.sum(psf_ab)
    # psf_ab = ut.upsample(psf_ab,size=psfsize)

    for i in range(psf_ab.shape[0]): psf_ab[i,...] = fftshift(psf_ab[i,...])

    crop_start = int((pupilsize-psfsize)/2)
    psf_ab = psf_ab[:,crop_start:crop_start+psfsize,crop_start:crop_start+psfsize]

    return psf_ab

def generate_psf_ab_1P(psf_fft,fplane):
    # same as DAOSLIMIT

    fplane = fftshift(torch.exp(fplane))
    psf_fft_ab = psf_fft*fplane

    psf_ab = ifft2(psf_fft_ab)
    psf_ab = torch.abs(psf_ab)
    # psf_ab = ut.upsample(psf_ab,size=psfsize)

    # for i in range(psf_ab.shape[0]): psf_ab[i,...] = fftshift(psf_ab[i,...])

    return psf_ab

def load_psfEx(PSF_dir,index):

    psfs_fft = []
    postfix = os.listdir(PSF_dir)[0].split('.')[1]
    psf_real_path = os.path.join(PSF_dir,'psf_'+str(index)+'_real.'+postfix)
    psf_imag_path = os.path.join(PSF_dir,'psf_'+str(index)+'_imag.'+postfix)
    psf_real = torch.from_numpy(tf.imread(psf_real_path).astype(np.float32))
    psf_imag = torch.from_numpy(tf.imread(psf_imag_path).astype(np.float32))
    psf_fft = torch.zeros([psf_real.shape[0],psf_real.shape[1],psf_real.shape[2]],dtype=torch.cfloat)
    psf_fft.real=psf_real ; psf_fft.imag=psf_imag
    psfs_fft.append(psf_fft)
    print('Load PSFEx:',psf_real_path,'PSF max:',torch.max(psf_real).item())
    print('Load PSFEx:',psf_imag_path,'PSF max:',torch.max(psf_imag).item())

    psfs_fft = torch.stack(psfs_fft,dim=0).squeeze()
    return psfs_fft