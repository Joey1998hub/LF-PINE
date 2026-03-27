import tifffile as tf
import numpy as np
import os

uv_res = 13
norm_op = 0
crop_xsize = 128
crop_zsize = 39
PSF_dir = './PSF/PSF_zoom4_61dz1_N13'

save_dir = PSF_dir+'_edit'
if not os.path.exists(save_dir): os.mkdir(save_dir)

for i in range(int(uv_res)):
    psf_path = os.path.join(PSF_dir,'psf_'+str(i+1)+'.tiff')
    psf = tf.imread(psf_path)
    psf = psf.astype(np.float32)
    print('Load PSF:',psf_path)
    z_res,x_res,y_res = psf.shape

    if crop_xsize == 0: crop_xstart = 0; crop_xend = x_res
    else: crop_xstart = int((x_res-crop_xsize)/2); crop_xend = crop_xstart+crop_xsize

    if crop_zsize == 0: crop_zstart = 0; crop_zend = z_res
    else: crop_zstart = int((z_res-crop_zsize)/2); crop_zend = crop_zstart+crop_zsize

    psf = psf[crop_zstart:crop_zend,crop_xstart:crop_xend,crop_xstart:crop_xend]

    if i == 0: psf_max = np.max(psf)
    if norm_op == 1:  psf = (1e-3/psf_max)*psf
    
    save_path = os.path.join(save_dir,'psf_'+str(i+1)+'.tif')
    tf.imwrite(save_path,psf)
    print('Saved :',save_path,'PSF max:',np.max(psf))