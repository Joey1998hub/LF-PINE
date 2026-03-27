import torch
import utils as ut
import tifffile as tf
import numpy as np
import os

psf_res = 786
polynum = 36
rms_values = np.arange(0.1, 1.1, 0.1)

zernike = tf.imread('./PSF/Phase/zernike_kMax128_poly105.tif')
zernike = torch.from_numpy(zernike).float()
polymode = list(range(polynum)) 
zernike_base  = ut.upsample(zernike[polymode,...], size=psf_res)

for RMS in rms_values:
    RMS_rounded = round(RMS, 1)

    zernike_coef = torch.randn(size=[polynum,1,1])
    zernike_coef[0,...] = 0  
    
    folder_name = f"PSF/Phase/zk7/Phase_PSFRes{psf_res}_Polynum{polynum}_RMS{RMS_rounded}"
    current_rms = torch.sum(zernike_coef**2).sqrt()
    zernike_coef = (RMS / current_rms) * zernike_coef 
    
    phase = torch.sum(zernike_base * zernike_coef, dim=0)
    
    txt_path = os.path.join(folder_name, "zernike_coef.txt")
    phase_path = os.path.join(folder_name, "phase.tif")
    
    os.makedirs(folder_name, exist_ok=True)
    tf.imwrite(phase_path, phase.numpy())
    np.savetxt(txt_path, torch.squeeze(zernike_coef).numpy(), fmt="%.4f")
    
    print(f"RMS={RMS_rounded} Saved: {folder_name}")