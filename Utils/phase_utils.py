from torch.fft import ifft2,fftshift
import torch;import torch.nn.functional as F

def genCircle(matrix_size,radius):
    matrix_shape = (matrix_size, matrix_size)

    x, y = torch.meshgrid(torch.arange(matrix_shape[0]), torch.arange(matrix_shape[1]))

    center_x, center_y = matrix_shape[0] // 2, matrix_shape[1] // 2

    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

    circle_matrix = torch.where(distance <= radius, torch.tensor(1), torch.tensor(0)).float()

    circle_matrix.requires_grad = False

    return circle_matrix

def generate_psf_ab(psf_fft,fplane,pupilsize,psfsize,amp=1e9):
    # same as DAOSLIMIT

    fplane = torch.exp(fplane)

    padding = int((pupilsize-psf_fft.shape[-1])/2)
    psf_fft_ab = F.pad(psf_fft*fplane,(padding,padding,padding,padding))

    psf_ab = ifft2(psf_fft_ab)
    psf_ab = torch.pow(torch.abs(psf_ab),2)
    psf_ab = torch.pow(torch.abs(psf_ab),2)
    psf_ab = psf_ab*amp

    for i in range(psf_ab.shape[0]):
        for j in range(psf_ab.shape[1]):
            psf_ab[i,j,...] = fftshift(psf_ab[i,j,...])

    crop_start = int((pupilsize-psfsize)/2)
    psf_ab = psf_ab[...,crop_start:crop_start+psfsize,crop_start:crop_start+psfsize]

    return psf_ab
