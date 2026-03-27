import torch
import torch.nn.functional as F
import os
import sys
import tifffile as tf
import numpy as np

sys.path.append(os.getcwd())
from Utils.DAO_utils import load_psfEx, generate_psf_ab
from Utils.utils import upsample

PSF_dir = 'PSF/PSFEx_zoom4_31dz1_N13'
pupilsize = 1024
psfsize = 1024
Nnum = 13
psf_res = 786  
polynum = 36   

rms_values = np.arange(0.1, 1.1, 0.1)

for RMS in rms_values:
    RMS_rounded = round(RMS, 1)

    phase_path = f'PSF/Phase/zk7/Phase_PSFRes{psf_res}_Polynum{polynum}_RMS{RMS_rounded}/phase.tif'
    save_dir = f'PSF/zk7/PSF_zoom4_31dz1_N13_1024_Phase_PSFRes{psf_res}_Polynum{polynum}_RMS{RMS_rounded}'
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    phase = torch.from_numpy(tf.imread(phase_path))
    
    for angle in range(Nnum):
        psf_fft = load_psfEx(PSF_dir, angle + 1)
        z_res, psf_res_actual, _ = psf_fft.shape
        
        fplane = torch.zeros([psf_res_actual, psf_res_actual], dtype=torch.cfloat)
        phase_upsampled = upsample(phase, size=psf_res_actual)  # 上采样到实际PSF分辨率
        fplane.imag = -phase_upsampled*2*torch.pi
        
        psf_ab = generate_psf_ab(psf_fft, fplane, pupilsize, psfsize)
        
        psf_name = f'psf_{angle + 1}.tiff'
        save_path = os.path.join(save_dir, psf_name)
        tf.imwrite(save_path, psf_ab.cpu().numpy())
        print(f'RMS={RMS_rounded} - Saved: {save_path}')